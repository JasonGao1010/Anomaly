"""Aggregate single-scan geometry distributions and write reproducible CSV tables."""

from __future__ import annotations

from collections import Counter
import csv
import json
import os
from pathlib import Path

import numpy as np

from .data import _atomic_json
from .evaluate import diagnostic_bin
from .profile import QUANTILES, describe


def summarize_geometry(output, extraction):
    """Aggregate exact probe counts on their existing common-valid point sets."""
    from contextlib import ExitStack
    from zipfile import ZipFile, ZIP_DEFLATED

    from .geometry import (GROUP_DTYPE, csv_rows, group_metrics, merge_groups,
                           normal_threshold, subtract_groups, threshold_counts)
    from .protocol import load_protocol

    output = Path(output)
    directory = output / "tables"
    directory.mkdir(exist_ok=True)
    sequences = tuple(load_protocol().public_sequence_ids)
    locations = [("train", 206), ("train", 201), *[("val", seq) for seq in sequences]]
    summaries = {where: json.loads((output / "counts" / where[0] / str(where[1]) / "summary.json").read_text())
                 for where in locations}
    keys = summaries[("train", 206)]["keys"]
    if any(set(summary["keys"]) != set(keys) for summary in summaries.values()):
        raise ValueError("Probe cohorts differ between normal sources and validation sequences")
    indices = {where: {key: str(i) for i, key in enumerate(summary["keys"])}
               for where, summary in summaries.items()}
    tables = {name: [] for name in ("metrics", "normal_transfer", "coverage", "confusion", "frames")}
    pooled_coverage, pooled_confusion, candidates = Counter(), Counter(), []
    for seq in sequences:
        summary = summaries[("val", seq)]
        tables["frames"].extend(summary["frames"])
        candidates.extend(summary["candidates"])
        for key, counts in summary["coverage"].items():
            for i, count in enumerate(counts):
                pooled_coverage[key, i] += count
        for key, counts in summary["confusion"].items():
            for i, count in enumerate(counts):
                pooled_confusion[key, i] += count

    def coverage_row(scope, sequence, key, counts):
        cohort, stratum = key.rsplit("|", 1)
        normal, anomaly, covered_normal, covered_anomaly = map(int, counts)
        return dict(scope=scope, sequence=sequence, cohort=cohort, stratum=stratum,
                    all_normal=normal, all_anomaly=anomaly, covered_normal=covered_normal,
                    covered_anomaly=covered_anomaly,
                    normal_coverage=100 * covered_normal / normal if normal else None,
                    anomaly_coverage=100 * covered_anomaly / anomaly if anomaly else None)

    for scope, seq, coverage, confusion in [
        *[("sequence", seq, summaries[("val", seq)]["coverage"], summaries[("val", seq)]["confusion"])
          for seq in sequences],
        ("pooled", "all", {key: [pooled_coverage[key, i] for i in range(4)] for key, _ in pooled_coverage},
         {key: [pooled_confusion[key, i] for i in range(3)] for key, _ in pooled_confusion}),
    ]:
        tables["coverage"].extend(coverage_row(scope, seq, key, counts) for key, counts in sorted(coverage.items()))
        for key, counts in sorted(confusion.items()):
            cohort, model, semantic = key.rsplit("|", 2)
            normal, fp = map(int, counts[:2])
            tables["confusion"].append(dict(scope=scope, sequence=seq, cohort=cohort, model=model,
                                           semantic=int(semantic), normal=normal, fp=fp,
                                           FPR=100 * fp / normal if normal else None))

    def metric_row(scope, sequence, key, groups, threshold):
        cohort, model = key.rsplit("|", 1)
        result = group_metrics(groups)
        point = result.get("recall_at_fpr_limit") or {}
        fixed = threshold_counts(groups, threshold)
        return dict(scope=scope, sequence=sequence, cohort=cohort, model=model,
                    normal=fixed["normal"], anomaly=fixed["anomaly"], AP=result["AP"],
                    AUROC=result["AUROC"], FPR95=result["FPR95"], R1=point.get("recall"),
                    threshold1=point.get("threshold"), FPR1=point.get("FPR"),
                    threshold_train206=threshold, FPR_train206=fixed["FPR"], recall_train206=fixed["recall"])

    def checked_counts(archives, where, key):
        group = archives[where][indices[where][key]]
        count, anomaly = int(group["count"].sum()), int(group["positive"].sum())
        cohort = key.rsplit("|", 1)[0]
        coverage = summaries[where]["coverage"][cohort + "|all"]
        if [count - anomaly, anomaly] != coverage[2:]:
            raise ValueError(f"Exact score counts differ from existing cohort coverage: {where}, {key}")
        return group

    # Archive members are read by one comparison key; complete archives never enter RAM.
    with ExitStack() as stack:
        archives = {where: stack.enter_context(np.load(
            output / "counts" / where[0] / str(where[1]) / "groups.npz", allow_pickle=False)) for where in locations}
        pooled_archive = stack.enter_context(ZipFile(output / "counts" / "pooled.npz", "w", ZIP_DEFLATED, allowZip64=True))
        for key_index, key in enumerate(keys):
            train = checked_counts(archives, ("train", 206), key)
            threshold = normal_threshold(train)
            cohort, model = key.rsplit("|", 1)
            for seq in (206, 201):
                group = train if seq == 206 else checked_counts(archives, ("train", seq), key)
                fixed = threshold_counts(group, threshold)
                if fixed["anomaly"]:
                    raise ValueError("Normal-only transfer summaries contain anomaly labels")
                all_normal = summaries[("train", seq)]["coverage"][cohort + "|all"][0]
                tables["normal_transfer"].append(dict(sequence=seq, cohort=cohort, model=model,
                    all_normal=all_normal, normal=fixed["normal"],
                    normal_coverage=100 * fixed["normal"] / all_normal if all_normal else None,
                    threshold_train206=threshold, fp=fixed["fp"], FPR_train206=fixed["FPR"]))
            del train, group
            uncompressed = sum(archives[("val", seq)].zip.getinfo(indices[("val", seq)][key] + ".npy").file_size
                               for seq in sequences)
            # Cache at most 450 MB of one-key inputs, allowing sorting and subtraction workspace.
            cache = {} if uncompressed <= 450_000_000 else None
            pooled = np.empty(0, GROUP_DTYPE)
            for seq in sequences:
                group = checked_counts(archives, ("val", seq), key)
                tables["metrics"].append(metric_row("sequence", seq, key, group, threshold))
                if cache is not None:
                    cache[seq] = group
                else:
                    if 4 * (pooled.nbytes + group.nbytes) > 2_500_000_000:
                        raise MemoryError("One exact probe group exceeds the bounded aggregation workspace")
                    pooled = merge_groups([pooled, group])
            if cache is not None:
                pooled = merge_groups(list(cache.values()))
            tables["metrics"].append(metric_row("pooled", "all", key, pooled, threshold))
            for seq in sequences:
                group = cache[seq] if cache is not None else checked_counts(archives, ("val", seq), key)
                remaining = subtract_groups(pooled, group)
                tables["metrics"].append(metric_row("leave_one_out", seq, key, remaining, threshold))
                del remaining
            with pooled_archive.open(str(key_index) + ".npy", "w", force_zip64=True) as member:
                np.lib.format.write_array(member, pooled, allow_pickle=False)
            del pooled, cache, group

    for name, rows in tables.items():
        csv_rows(directory / (name + ".csv"), rows)
    _atomic_json(directory / "candidates.json", candidates)
    _atomic_json(output / "counts" / "keys.json", dict(keys=keys, sequences=list(sequences)))
    val_frames = [row for row in extraction["frames"] if row["partition"] == "val"]
    official_frames = [row for row in val_frames if row["eligible"]]
    denominators = dict(sequences=len(sequences), scanned_frames=len(val_frames), eligible_frames=len(official_frames),
                        normal=sum(row["official_normal"] for row in official_frames),
                        anomaly=sum(row["official_anomaly"] for row in official_frames))
    pooled_rows = [row for row in tables["coverage"] if row["scope"] == "pooled" and row["stratum"] == "all"]
    if any([row["all_normal"], row["all_anomaly"]] != [denominators["normal"], denominators["anomaly"]]
           for row in pooled_rows):
        raise ValueError("Extracted official point denominators differ from aggregated coverage")
    result = dict(format="stu-conditional-geometry-probes", normal_fit="train/206", normal_transfer="train/201",
                  scope="Public development val19; each comparison uses its existing common-valid subset, not full_STU",
                  full_STU=False, independent_test=False, official_input=denominators,
                  metric_units="percent, except raw score thresholds", sequences=list(sequences),
                  threshold_source="train/206 normal scores; complete ties with empirical FPR at most 1 percent",
                  cohorts={row["cohort"]: row for row in pooled_rows},
                  tables={name: len(rows) for name, rows in tables.items()}, candidates=len(candidates),
                  pooled_groups=dict(archive="counts/pooled.npz", keys="counts/keys.json"),
                  leave_one_out="Subtract each original sequence's exact score counts; retain every remaining original point",
                  aggregation="One comparison key at a time; input cache at most 450 MB; streamed pooled archive")
    _atomic_json(output / "summary.json", result)
    return result


def _geometry_fonts():
    from matplotlib import font_manager
    from matplotlib.ft2font import FT2Font

    paths = ("/mnt/c/Windows/Fonts/simsun.ttc", "/mnt/c/Windows/Fonts/times.ttf")
    for path, expected in zip(paths, ("SimSun", "Times New Roman"), strict=True):
        if FT2Font(path).family_name != expected:
            raise ValueError("The requested font file does not contain the required actual font")
        font_manager.fontManager.addfont(path)
    return tuple(font_manager.FontProperties(fname=path) for path in paths)


def _geometry_pdf(path, pages):
    import re
    import subprocess

    report = subprocess.run(["pdffonts", str(path)], check=True, capture_output=True, text=True).stdout
    rows = [line for line in report.splitlines()[2:] if line.strip()]
    names = {line.split()[0].split("+")[-1] for line in rows}
    if names != {"SimSun", "TimesNewRomanPSMT"} or any(
        "TrueType" not in line or not re.search(r"\byes\s+yes\s+yes\s+\d+\s+\d+$", line) for line in rows
    ):
        raise RuntimeError("Geometry PDF did not embed exclusively the required TrueType fonts")
    info = subprocess.run(["pdfinfo", str(path)], check=True, capture_output=True, text=True).stdout
    if not re.search(rf"^Pages:\s+{pages}$", info, flags=re.MULTILINE):
        raise RuntimeError("Geometry PDF has an unexpected page count")
    return sorted(names)


def sampling_increment(output, reference=None):
    """Compare added sampling conditioning on exactly the sampling cohort's points.

    This is a supplementary development comparison after the main readout. Scores,
    reference conditions and tie handling remain unchanged; only C_direction's
    existing counts lose points outside the already defined sampling cohort.
    """
    import resource
    import time

    from .geometry import (as_data, comparisons, csv_rows, group_metrics, merge_groups,
                           score_groups, subtract_groups)
    from .probes import DIRECT, GEOMETRY, SAMPLING, _cells, fit_reference, score_reference
    from .protocol import load_protocol

    output = Path(output)
    started = time.monotonic()
    metadata = json.loads((output / "reference.json").read_text())
    if reference is None:
        normal = np.concatenate([np.load(path, allow_pickle=False)
                                 for path in sorted((output / "features/train/206").glob("*.npy"))])
        if np.any(normal["target"] != 0) or len(np.unique(normal["frame"])) != 449:
            raise ValueError("Sampling comparison requires the same normal train/206 reference")
        reference = fit_reference(as_data(normal))
        del normal
    if reference["metadata"] != metadata or metadata["source"] != "train/206 normal":
        raise ValueError("Sampling comparison cannot change the fitted normal reference")
    supported = {mode: np.array([int(cell) for cell, values in metadata["cells"][mode].items()
                                 if values["trusted_all"]], np.int32)
                 for mode in ("direction", "sampling")}
    global_valid = {mode: all(metadata["global_fields"][name]["trusted"] for name in (*GEOMETRY, *conditions))
                    for mode, conditions in (("direction", DIRECT), ("sampling", SAMPLING))}
    sequences = tuple(load_protocol().public_sequence_ids)
    groups = {model: {} for model in ("C_direction", "C_sampling")}
    removed_counts, rows = np.zeros(2, np.int64), []

    def counts(group):
        positive = int(group["positive"].sum())
        return np.array([int(group["count"].sum()) - positive, positive], np.int64)

    def metric_row(scope, sequence, model, group):
        metrics = group_metrics(group)
        point = metrics.get("recall_at_fpr_limit") or {}
        return dict(scope=scope, sequence=sequence, model=model, AP=metrics["AP"],
                    AUROC=metrics["AUROC"], FPR95=metrics["FPR95"], R1=point.get("recall"),
                    normal=metrics["normal_count"], anomaly=metrics["anomaly_count"])

    for sequence in sequences:
        parts, expected = [], {mode: np.zeros(2, np.int64) for mode in supported}
        for path in sorted((output / "features/val" / str(sequence)).glob("*.npy")):
            values = np.load(path, allow_pickle=False, mmap_mode="r")
            cells = _cells(as_data(values), reference["edges"])
            geometry_valid = np.logical_and.reduce([np.isfinite(values[name]) for name in GEOMETRY])
            masks = {}
            for mode, conditions in (("direction", DIRECT), ("sampling", SAMPLING)):
                masks[mode] = (geometry_valid & global_valid[mode]
                               & np.logical_and.reduce([np.isfinite(values[name]) for name in conditions])
                               & np.isin(cells[mode], supported[mode]))
                expected[mode] += np.bincount(values["target"][masks[mode]], minlength=2)
            if np.any(masks["sampling"] & ~masks["direction"]):
                raise ValueError("The sampling cohort must remain a subset of the direction cohort")
            excluded = masks["direction"] & ~masks["sampling"]
            if not excluded.any():
                continue
            selected = values[excluded]
            scores, individual, _ = score_reference(as_data(selected), reference)
            actual = {mode: finite for mode, _, finite in comparisons(scores, individual)
                      if mode in masks}
            if not actual["direction"].all() or actual["sampling"].any():
                raise ValueError("Fast exclusion identities disagree with the original score coverage")
            parts.append(score_groups(scores["C_direction"], selected["target"]))
        directory = output / "counts/val" / str(sequence)
        keys = json.loads((directory / "summary.json").read_text())["keys"]
        with np.load(directory / "groups.npz", allow_pickle=False) as saved:
            direction = saved[str(keys.index("direction|C_direction"))]
            sampling = saved[str(keys.index("sampling|C_sampling"))]
        if not np.array_equal(counts(direction), expected["direction"]) or not np.array_equal(counts(sampling), expected["sampling"]):
            raise ValueError("Fast cohort masks disagree with the existing exact score counts")
        removed = merge_groups(parts)
        direction = subtract_groups(direction, removed)
        if not np.array_equal(counts(direction), counts(sampling)):
            raise ValueError("Sampling increment methods do not cover the same normal/anomaly points")
        removed_counts += counts(removed)
        groups["C_direction"][sequence], groups["C_sampling"][sequence] = direction, sampling
        for model in groups:
            rows.append(metric_row("sequence", sequence, model, groups[model][sequence]))
        print(json.dumps(dict(stage="sampling_increment", sequence=sequence,
                              excluded_normal=int(counts(removed)[0]), excluded_anomaly=int(counts(removed)[1]),
                              seconds=round(time.monotonic() - started, 2))), flush=True)
    for model, sequence_groups in groups.items():
        pooled = merge_groups(list(sequence_groups.values()))
        rows.append(metric_row("pooled", "all", model, pooled))
        for sequence in sequences:
            remaining = subtract_groups(pooled, sequence_groups[sequence])
            rows.append(metric_row("leave_one_out", sequence, model, remaining))
    csv_rows(output / "tables/sampling_increment.csv", rows)
    return dict(rows=rows, excluded_normal=int(removed_counts[0]), excluded_anomaly=int(removed_counts[1]),
                seconds=time.monotonic() - started,
                max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                supplementary_development_comparison=True)


def plot_geometry(output):
    """Plot common-cohort diagnostics, using the actual required embedded fonts."""
    import warnings

    import matplotlib as mpl
    mpl.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import TwoSlopeNorm
    from .probes import GEOMETRY

    output = Path(output)
    with (output / "tables" / "metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        metrics = list(csv.DictReader(stream))
    with (output / "tables" / "coverage.csv").open(encoding="utf-8-sig", newline="") as stream:
        coverage = list(csv.DictReader(stream))
    indexed = {(row["scope"], row["sequence"], row["cohort"], row["model"]): row for row in metrics}
    covered = {(row["cohort"], row["stratum"]): row for row in coverage if row["scope"] == "pooled"}
    if len(indexed) != len(metrics):
        raise ValueError("Plot inputs contain duplicate metric identities")
    cn, en = _geometry_fonts()
    modes = ("range", "direction", "sampling")
    mode_names = dict(range="距离", direction="距离与射线方向", sampling="再加入邻距尺度")
    feature_names = ("局部表面残差", "表面厚度", "法向变化", "最小方差占比", "线性度", "平面度")
    controls = ("A_range", "A_direct", "A_sampling")
    colors = ("#4b5563", "#3176ab", "#bf6435")
    pngs = ("overview.png", "sequences.png", "coverage.png", "features.png")

    def number(row, field):
        return float(row[field]) if row[field] not in (None, "") else np.nan

    def ap(cohort, model, scope="pooled", sequence="all"):
        return number(indexed[(scope, str(sequence), cohort, model)], "AP")

    def coverage_percent(row, label):
        total = int(row["all_" + label])
        return 100 * int(row["covered_" + label]) / total if total else np.nan

    def style(axis):
        axis.spines[["top", "right"]].set_visible(False)
        for label in (*axis.get_xticklabels(), *axis.get_yticklabels()):
            label.set_fontproperties(en)
            label.set_fontsize(10)
        axis.xaxis.get_offset_text().set_fontproperties(en)
        axis.yaxis.get_offset_text().set_fontproperties(en)

    def save(fig, pdf, name):
        fig.savefig(output / name, dpi=190, facecolor="white")
        pdf.savefig(fig, facecolor="white")
        plt.close(fig)

    settings = {"font.family": "Times New Roman", "pdf.fonttype": 42,
                "ps.fonttype": 42, "axes.unicode_minus": False, "font.size": 10}
    with warnings.catch_warnings(record=True) as caught, mpl.rc_context(settings), PdfPages(output / "plots.pdf") as pdf:
        warnings.simplefilter("always")
        fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.6))
        fig.subplots_adjust(left=.065, right=.98, bottom=.18, top=.72, wspace=.3)
        fig.suptitle("共同有效子集中的总体区分能力", fontproperties=cn, fontsize=17, y=.96)
        for axis, mode, control in zip(axes, modes, controls):
            values = np.array([ap(mode, model) for model in (control, "B_geometry", "C_" + mode)])
            ceiling = max(.01, float(np.nanmax(values)) * 1.25) if np.isfinite(values).any() else 1.
            axis.bar(np.arange(3), values, color=colors, width=.65)
            axis.set_ylim(0, ceiling)
            axis.set_xticks(np.arange(3), ("A", "B", "C"))
            axis.set_ylabel("AP (%)", fontproperties=en)
            axis.set_title(mode_names[mode], fontproperties=cn, fontsize=13, pad=36)
            for index, value in enumerate(values):
                axis.text(index, value + .02 * ceiling if np.isfinite(value) else .02 * ceiling,
                          f"{value:.4f}" if np.isfinite(value) else "NA", ha="center", fontproperties=en)
            row = covered[(mode, "all")]
            for label, x, chinese in (("normal", .02, "正常"), ("anomaly", .52, "异常")):
                value = coverage_percent(row, label)
                axis.text(x, 1.035, chinese, transform=axis.transAxes, fontproperties=cn, fontsize=10)
                axis.text(x + .16, 1.035, f"{value:.1f}%" if np.isfinite(value) else "NA",
                          transform=axis.transAxes, fontproperties=en, fontsize=10)
            style(axis)
        fig.text(.5, .055, "图上比例表示共同覆盖率；每组仅比较共同有效点，不代表完整基准成绩。",
                 ha="center", fontproperties=cn, fontsize=10)
        save(fig, pdf, pngs[0])

        sequences = sorted({int(row["sequence"]) for row in metrics if row["scope"] == "sequence"})
        differences = np.array([[ap(mode, "C_" + mode, "sequence", seq) - ap(mode, "B_geometry", "sequence", seq)
                                 for mode in modes] for seq in sequences])
        span = max(.1, float(np.nanmax(np.abs(differences)))) if np.isfinite(differences).any() else 1.
        fig, axis = plt.subplots(figsize=(8.2, max(5.2, .29 * len(sequences) + 2.2)))
        fig.subplots_adjust(left=.15, right=.85, bottom=.12, top=.89)
        fig.suptitle("共同有效子集中的逐序列条件化变化", fontproperties=cn, fontsize=17, y=.965)
        image = axis.imshow(np.ma.masked_invalid(differences), aspect="auto", cmap="RdBu_r",
                            norm=TwoSlopeNorm(vmin=-span, vcenter=0, vmax=span))
        axis.set_xticks(np.arange(3), [mode_names[mode] for mode in modes])
        axis.set_yticks(np.arange(len(sequences)), [str(seq) for seq in sequences])
        axis.set_ylabel("序列", fontproperties=cn)
        for row in range(len(sequences)):
            for col in range(3):
                value = differences[row, col]
                axis.text(col, row, f"{value:+.4f}" if np.isfinite(value) else "NA", ha="center", va="center",
                          color="white" if abs(value) > .55 * span else "#171d23", fontproperties=en, fontsize=10)
        bar = fig.colorbar(image, ax=axis, fraction=.045, pad=.055)
        bar.set_label("平均精确率的条件化增益（百分点）", fontproperties=cn)
        style(axis)
        for label in axis.get_xticklabels():
            label.set_fontproperties(cn)
        style(bar.ax)
        fig.text(.5, .035, "逐序列使用该对照的共同有效点；正值表示条件参考得分的平均精确率更高。",
                 ha="center", fontproperties=cn, fontsize=10)
        save(fig, pdf, pngs[1])

        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharey=True)
        fig.subplots_adjust(left=.075, right=.98, bottom=.21, top=.79, wspace=.18)
        fig.suptitle("共同有效子集相对官方评价点的距离分组覆盖", fontproperties=cn, fontsize=17, y=.96)
        for axis, label in zip(axes, ("normal", "anomaly")):
            for mode, color in zip(modes, colors):
                rows = [covered[(mode, "range_" + str(i))] for i in range(4)]
                axis.plot(np.arange(4), [coverage_percent(row, label) for row in rows],
                          marker="o", linewidth=1.6, markersize=5, color=color, label=mode_names[mode])
            axis.set_title("正常" if label == "normal" else "异常", fontproperties=cn, fontsize=13)
            axis.set_xticks(np.arange(4), ("[2.5, 10)", "[10, 20)", "[20, 35)", "[35, 50]"))
            axis.set_xlabel("距离（米）", fontproperties=cn)
            axis.set_ylim(0, 105)
            axis.grid(axis="y", alpha=.2)
            axis.legend(prop=cn, loc="best", frameon=False)
            style(axis)
        axes[0].set_ylabel("覆盖率（百分比）", fontproperties=cn)
        fig.text(.5, .055, "每个比例的分母为相应距离内的全部官方正常点或异常点。",
                 ha="center", fontproperties=cn, fontsize=10)
        save(fig, pdf, pngs[2])

        fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.2), sharey=True)
        fig.subplots_adjust(left=.18, right=.98, bottom=.16, top=.76, wspace=.15)
        fig.suptitle("共同有效子集中的单个几何量区分能力", fontproperties=cn, fontsize=17, y=.96)
        for axis, mode, color in zip(axes, ("direction", "sampling"), colors[1:]):
            # Each B score comes from the very same feature/mode cohort as its C partner.
            values = np.array([[ap(f"feature/{feature}/{mode}", model) for model in ("B_geometry", "C_" + mode)]
                               for feature in GEOMETRY])
            ceiling = max(.01, float(np.nanmax(values)) * 1.25) if np.isfinite(values).any() else 1.
            for column, (name, shade) in enumerate((("无条件几何", colors[0]), ("条件几何", color))):
                locations = np.arange(len(GEOMETRY)) + (column - .5) * .3
                axis.barh(locations, values[:, column], height=.28, color=shade, label=name)
                for location, value in zip(locations, values[:, column]):
                    axis.text(value + .015 * ceiling if np.isfinite(value) else .015 * ceiling, location,
                              f"{value:.4f}" if np.isfinite(value) else "NA", va="center", fontproperties=en, fontsize=9)
            axis.set_xlim(0, ceiling)
            axis.set_yticks(np.arange(len(GEOMETRY)), feature_names)
            axis.set_xlabel("AP (%)", fontproperties=en)
            axis.set_title(mode_names[mode], fontproperties=cn, fontsize=13, pad=42)
            axis.legend(prop=cn, loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, frameon=False)
            style(axis)
            for label in axis.get_yticklabels():
                label.set_fontproperties(cn)
        axes[0].invert_yaxis()
        fig.text(.5, .05, "每一对柱均使用对应特征与条件下的共同有效点。",
                 ha="center", fontproperties=cn, fontsize=10)
        save(fig, pdf, pngs[3])
    if any("Glyph" in str(warning.message) or "font" in str(warning.message).lower() for warning in caught):
        raise RuntimeError("Rendered geometry figures reported a missing glyph or font warning")
    names = _geometry_pdf(output / "plots.pdf", 4)
    return dict(pdf="plots.pdf", pages=4, pngs=list(pngs), embedded_fonts=names)


def geometry_cases(output, data_root, reference=None):
    """Inspect preselected normal false-positive neighborhoods after full scoring."""
    import warnings

    import matplotlib as mpl
    mpl.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Rectangle

    from .evaluate import evaluation_targets
    from .geometry import as_data, comparisons, csv_rows
    from .probes import GEOMETRY, fit_reference, score_reference
    from .protocol import load_protocol
    from .scene import STUSequence, LabelMode

    output = Path(output)
    # Only the final all-sequence summary owns these case candidates.
    summary = json.loads((output / "summary.json").read_text())
    candidates = json.loads((output / "tables/candidates.json").read_text())
    if summary["candidates"] != len(candidates):
        raise ValueError("Final case-candidate count differs from the completed summary")
    methods = ("C_direction", "C_sampling")
    candidates = sorted((row for row in candidates if row["method"] in methods),
                        key=lambda row: (-row["fp"], row["sequence"], row["frame"], row["slot"]))
    selected, used_frames, used_sequences = [], set(), set()
    # Two rounds give each method at most two cases; no visual reselection is used.
    for _ in range(2):
        for method in methods:
            available = [row for row in candidates if row["method"] == method
                         and (row["sequence"], row["frame"]) not in used_frames]
            diverse = [row for row in available if row["sequence"] not in used_sequences]
            if not available:
                continue
            row = (diverse or available)[0]
            selected.append(row)
            used_frames.add((row["sequence"], row["frame"]))
            used_sequences.add(row["sequence"])
    rules = dict(methods=list(methods), maximum_per_method=2,
                 frames="two rounds, direction then sampling; prefer an unused sequence; descending frame FP, then sequence/frame/slot; no repeated frame",
                 anchor="one-metre sensor-coordinate grid with most distinct FP coordinates; lexicographic cell tie-break",
                 display="all actual returns within a 6 m cube about the selected cell centre; no display subsampling",
                 interpretation="preselected extreme development cases, not a representative sample or an object taxonomy")
    if not selected:
        result = dict(rules=rules, cases=[], pages=0, reason="no eligible normal false-positive candidates")
        _atomic_json(output / "cases.json", result)
        return result
    if reference is None:
        normal = np.concatenate([np.load(path, allow_pickle=False)
                                 for path in sorted((output / "features/train/206").glob("*.npy"))])
        if np.any(normal["target"] != 0) or len(np.unique(normal["frame"])) != 449:
            raise ValueError("Case reference requires all 449 cached normal train/206 frames")
        reference = fit_reference(as_data(normal))
        del normal
    if reference["metadata"] != json.loads((output / "reference.json").read_text()):
        raise ValueError("Reconstructed normal reference differs from the scoring reference")
    thresholds = json.loads((output / "thresholds.json").read_text())["values"]
    cn, en = _geometry_fonts()
    settings = {"font.family": "Times New Roman", "pdf.fonttype": 42,
                "ps.fonttype": 42, "axes.unicode_minus": False, "font.size": 10}
    details, table = [], []
    with warnings.catch_warnings(record=True) as caught, mpl.rc_context(settings), PdfPages(output / "cases.pdf") as pdf:
        warnings.simplefilter("always")
        for index, case in enumerate(selected, 1):
            seq, frame_id, method, cohort = (case[key] for key in ("sequence", "frame", "method", "cohort"))
            source = STUSequence.open(data_root, protocol=load_protocol(), partition="val",
                                      sequence_id=seq, label_mode=LabelMode.REQUIRED)[frame_id]
            xyz = source.xyzi[:, :3]
            target = evaluation_targets(xyz, source.labels.semantic)
            path = output / "features/val" / str(seq) / f"{frame_id:06d}.npy"
            values = np.load(path, allow_pickle=False)
            slots = values["source_slot"]
            if (not np.all(values["frame"] == frame_id)
                    or not np.array_equal(slots, np.flatnonzero(target >= 0))
                    or not np.array_equal(values["target"], target[slots])
                    or np.count_nonzero(target == 1) < 5):
                raise ValueError("Case cached point identities differ from the raw official point set")
            scores, individual, condition_cells = score_reference(as_data(values), reference)
            _, _, covered = next(item for item in comparisons(scores, individual) if item[0] == cohort)
            threshold = thresholds[cohort + "|" + method]
            fp_rows = np.flatnonzero(covered & (values["target"] == 0) & (scores[method] >= threshold)) if threshold is not None else np.array([], int)
            if len(fp_rows) != case["fp"] or not len(fp_rows):
                raise ValueError("Reconstructed case FP count differs from the final candidate")
            anchor = np.flatnonzero(slots == case["slot"])
            if len(anchor) != 1 or float(scores[method][anchor[0]]) != case["score"] or anchor[0] not in fp_rows:
                raise ValueError("Recorded candidate slot or score changed during case reconstruction")
            fp_slots = slots[fp_rows]
            unique_fp = np.unique(xyz[fp_slots], axis=0)
            cells, cell_counts = np.unique(np.floor(unique_fp).astype(np.int64), axis=0, return_counts=True)
            cell = cells[int(np.argmax(cell_counts))]
            center = cell.astype(float) + .5
            display = np.zeros(len(xyz), bool)
            display[source.real_slots] = np.all(np.abs(xyz[source.real_slots] - center) <= 3, axis=1)
            relative = xyz - center
            supported = np.zeros(len(xyz), bool)
            supported[slots[covered]] = True
            false = np.zeros(len(xyz), bool)
            false[fp_slots] = True
            anomaly = target == 1
            local_fp = display & false
            cell_fp = false & np.all(np.floor(xyz) == cell, axis=1)
            local_unique = np.unique(xyz[local_fp], axis=0).astype(np.float64)
            centred = local_unique - local_unique.mean(axis=0)
            covariance = centred.T @ centred / len(centred)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            maximum = float(eigenvalues[-1])
            normal_z = float(abs(eigenvectors[2, 0])) if eigenvalues[1] > 1e-12 else None
            fig, axes = plt.subplots(3, 2, figsize=(11.5, 11.8))
            fig.subplots_adjust(left=.07, right=.97, bottom=.11, top=.88, wspace=.15, hspace=.3)
            styles = ((display & ~supported & ~anomaly, "#d0d0d0", "未评价或未覆盖", 3),
                      (display & supported & (target == 0) & ~false, "#808080", "正常未误报", 4),
                      (local_fp, "#ad4a15", "正常误报", 7),
                      (display & anomaly, "#126aa4", "官方异常回波", 14))
            for row, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
                left, right = axes[row]
                left.scatter(relative[display & ~anomaly, a], relative[display & ~anomaly, b],
                             s=4, c="#888888", linewidths=0, rasterized=True)
                left.scatter(relative[display & anomaly, a], relative[display & anomaly, b],
                             s=14, c="#126aa4", linewidths=0, rasterized=True)
                for take, color, label, size in styles:
                    right.scatter(relative[take, a], relative[take, b], s=size, c=color,
                                  label=label, linewidths=0, rasterized=True)
                for axis in (left, right):
                    axis.add_patch(Rectangle((-.5, -.5), 1, 1, fill=False, edgecolor="#191919", linewidth=.8))
                    axis.set(xlim=(-3, 3), ylim=(-3, 3), aspect="equal")
                    axis.set_xlabel(f"{'xyz'[a]} (m)", fontproperties=en)
                    axis.set_ylabel(f"{'xyz'[b]} (m)", fontproperties=en)
                    axis.grid(alpha=.15)
                    for tick in (*axis.get_xticklabels(), *axis.get_yticklabels()):
                        tick.set_fontproperties(en)
            axes[0, 0].set_title("实际可见结构与官方异常位置", fontproperties=cn, fontsize=12)
            axes[0, 1].set_title("固定正常训练阈值下的误报", fontproperties=cn, fontsize=12)
            fig.suptitle("正常误报附近的真实可见结构", fontproperties=cn, fontsize=17, y=.972)
            fig.text(.5, .935, f"{index} | val/{seq}/{frame_id:06d} | {method} | t={threshold:.7g} | FP={len(fp_rows)}",
                     ha="center", fontproperties=en)
            handles, labels = axes[0, 1].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .045), ncol=4, prop=cn, frameon=False)
            fig.text(.5, .023, "坐标原点为网格中心；方框为一米网格投影；同尺度显示，缺测点不视为正常预测。",
                     ha="center", fontproperties=cn, fontsize=10)
            name = f"case_{index}.png"
            fig.savefig(output / name, dpi=180, facecolor="white")
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)
            local_rows = fp_rows[display[fp_slots]]
            feature_rows = []
            for feature in GEOMETRY:
                maximum_rows = local_rows[individual[method][feature][local_rows] == scores[method][local_rows]]
                identities = condition_cells[cohort][maximum_rows]
                tails = dict(lower_tail=0, upper_tail=0, equal_tail=0)
                # Resolve direction against the same complete-tie conditional reference.
                for identity in np.unique(identities):
                    ordered = reference["conditional"][cohort][int(identity)][feature]
                    observed = values[feature][maximum_rows[identities == identity]]
                    lower = np.searchsorted(ordered, observed, side="right")
                    upper = len(ordered) - np.searchsorted(ordered, observed, side="left")
                    tails["lower_tail"] += int(np.count_nonzero(lower < upper))
                    tails["upper_tail"] += int(np.count_nonzero(upper < lower))
                    tails["equal_tail"] += int(np.count_nonzero(lower == upper))
                if sum(tails.values()) != len(maximum_rows):
                    raise ValueError("Case tail directions do not account for every co-maximum point")
                feature_rows.append(dict(feature=feature,
                    observed_median=float(np.median(values[feature][local_rows])),
                    rarity_median=float(np.median(individual[method][feature][local_rows])),
                    co_maximum_points=len(maximum_rows), **tails))
            span = np.ptp(local_unique, axis=0).tolist()
            row = dict(case=index, sequence=seq, frame=frame_id, cohort=cohort, model=method,
                       figure=name, threshold_train206=threshold, frame_fp=len(fp_rows),
                       local_actual=int(display.sum()), local_normal=int(np.sum(display & (target == 0))),
                       local_covered_normal=int(np.sum(display & supported & (target == 0))),
                       local_fp=int(local_fp.sum()), local_anomaly=int(np.sum(display & anomaly)),
                       local_unique_fp=len(local_unique), cell_unique_fp=int(cell_counts.max()),
                       co_maximum_features=",".join(item["feature"] for item in feature_rows if item["co_maximum_points"]),
                       fp_span_x_m=span[0], fp_span_y_m=span[1], fp_span_z_m=span[2],
                       fp_plane_rms_m=float(np.sqrt(max(0., eigenvalues[0]))), fp_plane_abs_normal_z=normal_z,
                       fp_linearity=float((eigenvalues[2] - eigenvalues[1]) / maximum) if maximum > 0 else None,
                       fp_planarity=float((eigenvalues[1] - eigenvalues[0]) / maximum) if maximum > 0 else None,
                       centre_x_m=float(center[0]), centre_y_m=float(center[1]), centre_z_m=float(center[2]),
                       nearest_official_anomaly_m=float(np.linalg.norm(xyz[anomaly] - center, axis=1).min()))
            table.append(row)
            details.append(dict(**row, feature_cache=str(path.relative_to(output)),
                                selected_cell=cell.tolist(), selected_cell_fp_slots=np.flatnonzero(cell_fp).tolist(),
                                local_fp_bounds_m=[local_unique.min(axis=0).tolist(), local_unique.max(axis=0).tolist()],
                                local_fp_covariance_eigenvalues_m2=eigenvalues.tolist(), feature_rarity=feature_rows))
    if any("Glyph" in str(warning.message) or "font" in str(warning.message).lower() for warning in caught):
        raise RuntimeError("Rendered geometry cases reported a missing glyph or font warning")
    names = _geometry_pdf(output / "cases.pdf", len(details))
    csv_rows(output / "tables/cases.csv", table)
    result = dict(rules=rules, pages=len(details), pdf="cases.pdf", embedded_fonts=names,
                  geometry="PCA describes distinct visible local FP coordinates, not object identity or true normal surfaces",
                  feature_rarity="co-maximum counts include all ties and may overlap; lower/upper/equal tail uses inclusive normal counts in the same condition cell; score components, not causal attribution",
                  cases=details)
    _atomic_json(output / "cases.json", result)
    return result


def source_geometry_cases(output, reference_root="results/geometry", data_root="/home/jasongao/Data/STU"):
    """Describe original normal sources in fixed direction conditions, then inspect six anchors."""
    from collections import defaultdict
    import warnings

    import matplotlib as mpl
    mpl.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    from .geometry import as_data, csv_rows
    from .probes import _cells, fit_reference, tail_score
    from .profile import GEOMETRY_PARAMETERS, observed_geometry
    from .protocol import load_protocol
    from .scene import STUSequence, LabelMode

    output, reference_root = Path(output), Path(reference_root)
    artifacts = ["normal_cases.json", "normal_cases.pdf", "normal.csv",
                 *(f"normal_{i}.png" for i in range(1, 7))]
    if any((output / name).exists() for name in artifacts):
        raise FileExistsError("Original-source geometry artifacts already exist")
    if any(not (reference_root / "features/train" / source).is_dir() for source in ("206", "201")):
        raise FileNotFoundError("Original-source feature caches were removed; explicit re-extraction is required")
    output.mkdir(parents=True, exist_ok=True)
    train_paths = sorted((reference_root / "features/train/206").glob("*.npy"))
    train_chunks = [np.load(path, allow_pickle=False, mmap_mode="r") for path in train_paths]
    starts = np.r_[0, np.cumsum([len(chunk) for chunk in train_chunks])]
    normal = np.concatenate(train_chunks)
    del train_chunks
    reference = fit_reference(as_data(normal))
    if reference["metadata"] != json.loads((reference_root / "reference.json").read_text()):
        raise ValueError("Original-source analysis changed the existing train/206 reference")
    thresholds = json.loads((reference_root / "thresholds.json").read_text())["values"]
    features = ("normal_change", "surface_residual")
    counts, seen_frames, seen_cells = defaultdict(Counter), defaultdict(lambda: defaultdict(set)), defaultdict(lambda: defaultdict(set))
    candidates, cache_frames = defaultdict(list), {}
    names = ("cached", "finite", "supported", "lower", "central", "upper")
    quantiles = {int(cell): {feature: np.quantile(fields[feature], [.1, .9])
                            for feature in features if len(fields[feature])}
                 for cell, fields in reference["conditional"]["direction"].items()}

    def accumulate(key, values, status, cells, frame):
        masks = (np.ones(len(values), bool), np.isfinite(values), status >= 0,
                 status == 0, status == 1, status == 2)
        for name, mask in zip(names, masks, strict=True):
            number = int(mask.sum())
            counts[key][name] += number
            if number:
                seen_frames[key][name].add(frame)
                seen_cells[key][name].update(int(c) for c in np.unique(cells[mask]) if c >= 0)

    for sequence, expected_frames in ((206, 449), (201, 682)):
        if sequence == 206:
            frames = ((int(path.stem), normal[start:end]) for path, start, end
                      in zip(train_paths, starts[:-1], starts[1:], strict=True))
        else:
            paths = sorted((reference_root / "features/train/201").glob("*.npy"))
            frames = ((int(path.stem), np.load(path, allow_pickle=False)) for path in paths)
        visited, empty = set(), []
        for frame, values in frames:
            if frame in visited or np.any(values["frame"] != frame) or np.any(values["target"] != 0):
                raise ValueError("Original normal cache has repeated frames or nonnormal labels")
            visited.add(frame)
            if not len(values):
                empty.append(frame)
                continue
            cells = _cells(as_data(values), reference["edges"])["direction"]
            status = {feature: np.full(len(values), -1, np.int8) for feature in features}
            ranks = {feature: np.full(len(values), np.nan) for feature in features}
            for cell in np.unique(cells):
                indices = np.flatnonzero(cells == cell)
                fields = reference["conditional"]["direction"].get(int(cell), {})
                for feature in features:
                    ordered = fields.get(feature, ())
                    valid = indices[np.isfinite(values[feature][indices])]
                    if len(ordered):
                        observed = values[feature][valid]
                        low, high = quantiles[int(cell)][feature]
                        status[feature][valid] = np.where(observed < low, 0, np.where(observed > high, 2, 1))
                        # Midranks handle ties without declaring a constant normal value extreme.
                        ranks[feature][valid] = (np.searchsorted(ordered, observed, side="left")
                                                + np.searchsorted(ordered, observed, side="right")) / (2 * len(ordered))
                    accumulate((sequence, feature, int(cell)), values[feature][indices],
                               status[feature][indices], cells[indices], frame)
                common = indices[np.isfinite(ranks[features[0]][indices]) & np.isfinite(ranks[features[1]][indices])]
                if not len(common):
                    continue
                angle, residual = (ranks[feature][common] for feature in features)
                for kind, eligible, priority, raw in (
                    ("low_both", (angle <= .1) & (residual <= .1), np.maximum(angle, residual), values["normal_change"][common]),
                    ("high_normal_change", angle >= .9, -angle, -values["normal_change"][common]),
                    ("high_surface_residual", residual >= .9, -residual, -values["surface_residual"][common]),
                ):
                    choices = np.flatnonzero(eligible)
                    if not len(choices):
                        continue
                    chosen = choices[np.lexsort((values["source_slot"][common[choices]], raw[choices], priority[choices]))[0]]
                    index = int(common[chosen])
                    candidates[sequence, kind].append(dict(sequence=sequence, frame=frame,
                        slot=int(values["source_slot"][index]), direction_cell=int(cell), kind=kind,
                        priority=float(priority[chosen]), tie_value=float(raw[chosen]),
                        **{feature: float(values[feature][index]) for feature in features},
                        **{feature + "_rank": float(ranks[feature][index]) for feature in features},
                        **{name: float(values[name][index]) for name in ("range", "ray_z", "scale")}))
            for feature in features:
                accumulate((sequence, feature, "all"), values[feature], status[feature], cells, frame)
        if len(visited) != expected_frames:
            raise ValueError("Original-source cache does not contain its complete expected frame set")
        cache_frames[str(sequence)] = dict(total=len(visited), nonempty=len(visited) - len(empty),
                                            empty=len(empty), empty_frames=empty)
    del normal
    rows = []
    direction_bins = len(reference["edges"]["ray_z"]) - 1
    for key in sorted(counts, key=lambda item: (item[0], item[1], str(item[2]))):
        sequence, feature, cell = key
        count = counts[key]
        band = quantiles.get(cell, {}).get(feature)
        row = dict(sequence=sequence, feature=feature, conditioning="direction", cell=cell,
                   reference_p10=float(band[0]) if band is not None else None,
                   reference_p90=float(band[1]) if band is not None else None,
                   **{name + "_points": count[name] for name in names},
                   **{name + "_frames": len(seen_frames[key][name]) for name in names},
                   **{name + "_cells": len(seen_cells[key][name]) for name in names},
                   geometry_missing_points=count["cached"] - count["finite"],
                   reference_uncovered_points=count["finite"] - count["supported"],
                   **{name + "_percent_of_supported": 100 * count[name] / count["supported"] if count["supported"] else None
                      for name in ("lower", "central", "upper")})
        if cell != "all" and cell >= 0:
            r, d = divmod(cell, direction_bins)
            row.update(range_low=float(reference["edges"]["range"][r]), range_high=float(reference["edges"]["range"][r+1]),
                       ray_z_low=float(reference["edges"]["ray_z"][d]), ray_z_high=float(reference["edges"]["ray_z"][d+1]))
        rows.append(row)
    csv_rows(output / "normal.csv", rows)
    sources = {sequence: STUSequence.open(data_root, protocol=load_protocol(), partition="train",
                                          sequence_id=sequence, label_mode=LabelMode.REQUIRED) for sequence in (206, 201)}
    selected = []
    for sequence in (206, 201):
        used_frames, used_cells, used_positions = set(), set(), set()
        for kind in ("low_both", "high_normal_change", "high_surface_residual"):
            ranked = sorted(candidates[sequence, kind], key=lambda row: (row["priority"], row["tie_value"], row["frame"], row["slot"]))
            available = [row for row in ranked if row["frame"] not in used_frames]
            ordered = [row for row in available if row["direction_cell"] not in used_cells]
            ordered += [row for row in available if row["direction_cell"] in used_cells]
            for candidate in ordered:
                source = sources[sequence][candidate["frame"]]
                center = source.xyzi[candidate["slot"], :3].astype(float)
                world = center @ source.lidar_pose[:3, :3].T + source.lidar_pose[:3, 3]
                position = tuple(np.floor(world).astype(int))
                if position not in used_positions:
                    selected.append(dict(candidate, center=center.tolist(), world_cell=list(map(int, position))))
                    used_frames.add(candidate["frame"])
                    used_cells.add(candidate["direction_cell"])
                    used_positions.add(position)
                    break
            else:
                raise ValueError("No distinct original-source candidate satisfies a declared case type")
    rules = dict(reference="unchanged train/206 normal reference", conditioning="range and ray_z only",
                 selection="three cases per original source: both midranks <= .1, normal_change >= .9, surface_residual >= .9; prioritize distinct frame, then unused direction cell; rank/raw value/frame/slot order; distinct metre cells in each source's world coordinates",
                 scope="all existing cached normal samples in 449 train/206 and 682 train/201 frames; up to 8192 range-valid slots sampled before normal-label filtering per frame, not all raw points",
                 bands="below p10, inclusive p10-p90, above p90 of the fixed train/206 condition-specific normal reference",
                 limitations="extreme anchors establish observed examples only; frames and world cells do not establish independent objects or environments; unseen structures may be absent from the sampled cache")
    _atomic_json(output / "normal_cases.json", dict(rules=rules, cache_frames=cache_frames, selected=selected, rendered=False))
    cn, en = _geometry_fonts()
    titles = {"low_both": "法向变化与局部残差均较小", "high_normal_change": "法向变化较大", "high_surface_residual": "局部残差较大"}
    settings = {"font.family": "Times New Roman", "pdf.fonttype": 42, "ps.fonttype": 42,
                "axes.unicode_minus": False, "font.size": 10}
    details = []
    with warnings.catch_warnings(record=True) as caught, mpl.rc_context(settings), PdfPages(output / "normal_cases.pdf") as pdf:
        warnings.simplefilter("always")
        for number, case in enumerate(selected, 1):
            source = sources[case["sequence"]][case["frame"]]
            xyz, actual = source.xyzi[:, :3], source.real_slots
            checked = observed_geometry(source.xyzi[actual], actual, query_slots=np.array([case["slot"]], np.int32), workers=1)
            if any(float(checked[feature][0]) != case[feature] for feature in (*features, "range", "ray_z", "scale")):
                raise ValueError("Original-source anchor geometry differs from its saved cache")
            if load_protocol().semantic_class_map.get(int(source.labels.semantic[case["slot"]]), 255) == 255:
                raise ValueError("Selected original anchor is no longer a valid normal point")
            center = np.array(case["center"])
            unique, first = np.unique(xyz[actual], axis=0, return_index=True)
            representative = actual[first]
            distance = np.linalg.norm(unique.astype(float) - center, axis=1)
            valid = np.flatnonzero((distance > 0) & (distance <= GEOMETRY_PARAMETERS["radius_m"]))
            support = valid[np.lexsort((representative[valid], distance[valid]))[:GEOMETRY_PARAMETERS["neighbors"]]]
            if len(support) != int(checked["neighbor_count"][0]):
                raise ValueError("Displayed original-source support differs from the authoritative neighborhood")
            display = actual[np.all(np.abs(xyz[actual] - center) <= 3, axis=1)]
            local = xyz[display].astype(float) - center
            support_xyz = unique[support].astype(float)
            support_relative = support_xyz - center
            centered = support_xyz - support_xyz.mean(axis=0)
            eigen, vectors = np.linalg.eigh(centered.T @ centered / len(centered))
            fixed_probes = {}
            for feature in features:
                ordered = reference["conditional"]["direction"][case["direction_cell"]][feature]
                score = float(tail_score(np.array([case[feature]], np.float32), ordered)[0])
                threshold = thresholds[f"feature/{feature}/direction|C_direction"]
                fixed_probes[feature] = dict(score=score, threshold=threshold,
                                            false_positive=threshold is not None and score >= threshold)
            fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.6))
            fig.subplots_adjust(left=.065, right=.98, bottom=.22, top=.75, wspace=.25)
            for axis, (a, b) in zip(axes, ((0, 1), (0, 2), (1, 2)), strict=True):
                axis.scatter(local[:, a], local[:, b], s=3, c="#999999", linewidths=0, rasterized=True, label="实际可见回波")
                axis.scatter(support_relative[:, a], support_relative[:, b], s=20, c="#bd611d", linewidths=0,
                             rasterized=True, label="几何支持邻域")
                axis.scatter([0], [0], s=65, marker="x", c="#126aa4", linewidths=1.7, label="选定正常回波")
                axis.set(xlim=(-3, 3), ylim=(-3, 3), aspect="equal")
                axis.set_xlabel(f"{'xyz'[a]} (m)", fontproperties=en)
                axis.set_ylabel(f"{'xyz'[b]} (m)", fontproperties=en)
                axis.grid(alpha=.15)
                for tick in (*axis.get_xticklabels(), *axis.get_yticklabels()):
                    tick.set_fontproperties(en)
            fig.suptitle(titles[case["kind"]], fontproperties=cn, fontsize=17, y=.97)
            fig.text(.5, .875, f"{number} | train/{case['sequence']}/{case['frame']:06d} | slot={case['slot']} | cell={case['direction_cell']}", ha="center", fontproperties=en)
            fig.text(.5, .815, f"normal_change={case['normal_change']:.6g} rad | surface_residual={case['surface_residual']:.6g} m | range={case['range']:.3f} m", ha="center", fontproperties=en)
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .085), ncol=3, prop=cn, frameon=False)
            fig.text(.5, .038, "坐标原点为选定回波；仅描述真实可见结构，候选类型不等于物体类别。", ha="center", fontproperties=cn)
            name = f"normal_{number}.png"
            fig.savefig(output / name, dpi=180, facecolor="white")
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)
            details.append(dict(case, case=number, figure=name, actual_display_returns=len(display),
                                display_unique_positions=len(np.unique(xyz[display], axis=0)), support_positions=len(support),
                                support_slots=representative[support].astype(int).tolist(),
                                support_bounds_m=[support_xyz.min(axis=0).tolist(), support_xyz.max(axis=0).tolist()],
                                support_covariance_eigenvalues_m2=eigen.tolist(), support_abs_normal_z=float(abs(vectors[2, 0])),
                                support_plane_rms_m=float(np.sqrt(max(0., eigen[0]))),
                                fixed_direction_probes=fixed_probes,
                                raw_geometry_exactly_reproduced=True))
    if any("Glyph" in str(warning.message) or "font" in str(warning.message).lower() for warning in caught):
        raise RuntimeError("Original-source case rendering reported a missing glyph or font warning")
    fonts = _geometry_pdf(output / "normal_cases.pdf", len(details))
    result = dict(rules=rules, cache_frames=cache_frames, cases=details, rendered=True, pages=len(details), embedded_fonts=fonts,
                  statistics="normal.csv", source_totals=[row for row in rows if row["cell"] == "all"])
    _atomic_json(output / "normal_cases.json", result)
    return result


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def finite_json(value):
    if isinstance(value, dict):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [finite_json(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return "-inf" if value < 0 else "inf" if value > 0 else None
    return value


def physical_unit(meta):
    factor, key = meta["factor"], meta["metric"]
    if key == "background_intensity_variance":
        return "原始强度单位的平方"
    if "intensity" in key or factor in ("C01", "C03"):
        return "原始强度单位"
    if key in (
        "azimuth",
        "elevation",
        "azimuth_span",
        "elevation_span",
        "azimuth_gap",
    ):
        return "弧度"
    if any(x in key for x in ("distance", "ground_height")) or key in (
        "x",
        "y",
        "z",
        "length",
        "width",
        "height",
    ):
        return "米"
    if "fraction" in key or key in (
        "aspect",
        "linearity",
        "planarity",
        "scattering",
    ):
        return "比例或比值"
    return "类别编码" if meta["categorical"] else "数量"


def summarize(
    meta,
    x,
    c,
    w,
    *,
    sequence_weights=None,
    sequence_mean=None,
    sequence_bins=None,
    sequence_count=1,
):
    result = {}
    variants = [
        (
            "observation_equal",
            c,
            meta["total"] / meta["n"] if meta["n"] else None,
            meta["bin_counts"],
        ),
        (
            "frame_equal",
            w,
            meta["frame_total"] / meta["valid_frames"]
            if meta["valid_frames"]
            else None,
            meta["bin_frame_weights"],
        ),
    ]
    if sequence_weights is not None:
        variants.append(
            (
                "sequence_equal",
                sequence_weights,
                sequence_mean / sequence_count if sequence_count else None,
                sequence_bins,
            )
        )
    for view, weights, mean, bins in variants:
        row = describe(meta, x, weights, mean=mean, bin_weights=bins)
        row.update(
            sequence_count=sequence_count,
            value_unit=physical_unit(meta),
            weighting=view,
            status="已统计"
            if meta["n"]
            else "没有适用观测"
            if not meta["denominator"]
            else "估计不可靠或缺失",
        )
        if meta["categorical"]:
            universe = set(map(int, x)) | {
                int((a + b) / 2)
                for a, b in zip(meta["bins"][:-1], meta["bins"][1:], strict=True)
                if np.isfinite(a + b)
            }
            count_map = dict(zip(map(int, x), map(int, c)))
            row["category_counts"] = {
                str(v): count_map.get(v, 0) for v in sorted(universe)
            }
            row["categories"] = {
                str(v): row["categories"].get(str(v), 0.0 if weights.sum() else None)
                for v in sorted(universe)
            }
        elif not meta["resolution"]:
            row["unique_count"] = len(x)
            row["repeated_value_fraction"] = (
                float(c[c > 1].sum() / c.sum()) if c.sum() else None
            )
            row["minimum_count"] = int(c[0]) if len(c) else 0
            row["maximum_count"] = int(c[-1]) if len(c) else 0
            top = np.argsort(-c, kind="stable")[:10]
            row["most_frequent_values"] = [
                dict(value=float(x[i]), count=int(c[i])) for i in top
            ]
        result[view] = row
    return result


def aggregate_profile(output):
    output = Path(output)
    spec = json.loads((output / "spec.json").read_text())
    merged = {}
    result = dict(sequences={}, series={})
    frames = []
    instances = []
    for seq in spec["sequences"]:
        directory = output / str(seq)
        coverage = json.loads((directory / "summary.json").read_text())
        if coverage.get("format") != "stu-frame-profile":
            raise ValueError(
                "profile data must be regenerated with the current single-scan source"
            )
        result["sequences"][str(seq)] = dict(coverage=coverage, series={})
        for name, target in (
            ("frames", frames),
            ("instances", instances),
        ):
            target.extend(read_rows(directory / f"{name}.jsonl"))
        with np.load(directory / "histograms.npz") as saved:
            for meta, x, c, w in saved_distributions(saved):
                key = meta["key"]
                assert int(c.sum()) == meta["n"]
                np.testing.assert_allclose(
                    w.sum(), meta["valid_frames"], atol=1e-9, rtol=1e-12
                )
                result["sequences"][str(seq)]["series"][key] = summarize(
                    meta, x, c, w, sequence_count=int(meta["n"] > 0)
                )
                if key not in merged:
                    combined = {
                        k: v
                        for k, v in meta.items()
                        if k
                        not in (
                            "n",
                            "denominator",
                            "frames",
                            "valid_frames",
                            "total",
                            "frame_total",
                            "minimum",
                            "maximum",
                            "bin_counts",
                            "bin_frame_weights",
                        )
                    }
                    combined.update(
                        n=0,
                        denominator=0,
                        frames=0,
                        valid_frames=0,
                        total=0.0,
                        frame_total=0.0,
                        minimum=None,
                        maximum=None,
                        bin_counts=np.zeros(len(meta["bins"]) - 1, np.int64),
                        bin_frame_weights=np.zeros(len(meta["bins"]) - 1),
                        counts=Counter(),
                        frame_weights=Counter(),
                        sequence_weights=Counter(),
                        sequence_bins=np.zeros(len(meta["bins"]) - 1),
                        sequence_mean=0.0,
                        sequence_count=0,
                    )
                    merged[key] = combined
                item = merged[key]
                for field in (
                    "n",
                    "denominator",
                    "frames",
                    "valid_frames",
                    "total",
                    "frame_total",
                ):
                    item[field] += meta[field]
                item["bin_counts"] += meta["bin_counts"]
                item["bin_frame_weights"] += meta["bin_frame_weights"]
                if meta["n"]:
                    item["minimum"] = (
                        meta["minimum"]
                        if item["minimum"] is None
                        else min(item["minimum"], meta["minimum"])
                    )
                    item["maximum"] = (
                        meta["maximum"]
                        if item["maximum"] is None
                        else max(item["maximum"], meta["maximum"])
                    )
                    item["sequence_count"] += 1
                    item["sequence_mean"] += meta["total"] / meta["n"]
                    item["sequence_bins"] += np.asarray(meta["bin_counts"]) / meta["n"]
                    item["sequence_weights"].update(
                        dict(zip(x.tolist(), (c / meta["n"]).tolist()))
                    )
                item["counts"].update(dict(zip(x.tolist(), c.tolist())))
                item["frame_weights"].update(dict(zip(x.tolist(), w.tolist())))
        print(json.dumps(dict(event="aggregated_sequence", sequence=seq)), flush=True)
    for key, item in merged.items():
        x = np.array(sorted(item["counts"]))
        c = np.array([item["counts"][v] for v in x], np.int64)
        w = np.array([item["frame_weights"][v] for v in x])
        s = np.array([item["sequence_weights"][v] for v in x])
        meta = {
            k: v
            for k, v in item.items()
            if k
            not in (
                "counts",
                "frame_weights",
                "sequence_weights",
                "sequence_mean",
                "sequence_bins",
                "sequence_count",
            )
        }
        result["series"][key] = summarize(
            meta,
            x,
            c,
            w,
            sequence_weights=s,
            sequence_mean=item["sequence_mean"],
            sequence_bins=item["sequence_bins"],
            sequence_count=item["sequence_count"],
        )
    result.update(definitions=spec, frames=frames, instances=instances)
    result["totals"] = {
        key: sum(r["coverage"][key] for r in result["sequences"].values())
        for key in (
            "frames",
            "slots",
            "visible",
            "zero_slots",
            "ignore",
            "normal",
            "anomaly",
            "anomaly_in_range",
            "unknown_instance_points",
            "official_frames",
            "official_anomaly_points",
            "official_normal_points",
            "instance_frame_count",
        )
    }
    result["totals"]["sequences"] = len(result["sequences"])
    _atomic_json(output / "summary.json", finite_json(result))
    return result


def saved_distributions(saved):
    for i, meta in enumerate(json.loads(str(saved["catalog"]))):
        x, c, w = saved[f"x{i}"], saved[f"c{i}"], saved[f"w{i}"]
        yield meta, x, c, w
        if meta["metric"] == "background_intensity_std":
            # Nonnegative std -> variance is monotone; reuse every observed value and weight.
            variance = x * x
            edges = np.asarray(meta["bins"])
            derived = dict(
                meta,
                key=meta["key"].replace("_std|", "_variance|"),
                metric="background_intensity_variance",
                bins=edges * abs(edges),
                minimum=meta["minimum"] ** 2 if meta["n"] else None,
                maximum=meta["maximum"] ** 2 if meta["n"] else None,
                total=float(variance @ c),
                frame_total=float(variance @ w),
            )
            yield derived, variance, c, w


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(finite_json(value), ensure_ascii=False)
                    if isinstance(value, (dict, list, tuple, np.ndarray))
                    else value
                    for key, value in row.items()
                }
            )


def write_tables(output, result, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    frame_rows = []
    for frame in result["frames"]:
        row = {
            key: value for key, value in frame.items() if not key.endswith("_intensity")
        }
        for group in ("normal", "anomaly", "ignore"):
            distribution = frame[f"{group}_intensity"]
            for q in QUANTILES:
                row[f"{group}_intensity_p{int(q * 100):02d}"] = (
                    distribution["quantiles"][str(q)] if distribution else None
                )
        frame_rows.append(row)
    tables = {
        "frames": frame_rows,
        "instances": result["instances"],
        "sequences": [values["coverage"] for values in result["sequences"].values()],
        "totals": [result["totals"]],
    }
    continuous, categorical, bins = [], [], []
    populations = [("all", result["series"])] + [
        (seq, values["series"]) for seq, values in result["sequences"].items()
    ]
    for sequence, series in populations:
        for key, views in sorted(series.items()):
            for weighting, data in views.items():
                row = dict(
                    sequence=sequence,
                    factor=data["factor"],
                    metric=data["metric"],
                    scope=data["scope"],
                    group=data["group"],
                    unit=data["unit"],
                    weighting=weighting,
                    n=data["n"],
                    denominator=data["denominator"],
                    frames=data["frames"],
                    valid_frames=data["valid_frames"],
                    sequence_count=data["sequence_count"],
                    missing_fraction=data["missing_fraction"],
                    empty_frame_fraction=data["empty_frame_fraction"],
                    minimum=data["minimum"],
                    maximum=data["maximum"],
                    mean=data["mean"],
                )
                if data["categorical"]:
                    for category, fraction in data["categories"].items():
                        categorical.append(
                            dict(
                                row,
                                category=category,
                                fraction=fraction,
                                count=data["category_counts"][category],
                            )
                        )
                else:
                    for q in QUANTILES:
                        estimate = data["quantiles"].get(str(q), {})
                        for field in ("value", "lower", "upper"):
                            row[f"p{int(q * 100):02d}_{field}"] = estimate.get(field)
                    continuous.append(row)
                for index, fraction in enumerate(data["bin_fraction"]):
                    bins.append(
                        dict(
                            sequence=sequence,
                            factor=data["factor"],
                            metric=data["metric"],
                            scope=data["scope"],
                            group=data["group"],
                            unit=data["unit"],
                            weighting=weighting,
                            n=data["n"],
                            bin=index,
                            lower=data["bins"][index],
                            upper=data["bins"][index + 1],
                            fraction=fraction,
                        )
                    )
    tables.update(continuous=continuous, categorical=categorical, bins=bins)
    matched = {f"{i}_{j}": [] for i in range(4) for j in range(4)}
    for row in result["frames"]:
        group = diagnostic_bin(
            row["anomaly_in_range"], row["anomaly_in_range_distance_median"]
        )
        if group is not None:
            matched[group].append(row)
    tables["count_distance"] = [
        dict(
            group=group,
            count_bin=int(group[0]),
            range_bin=int(group[2]),
            sequences=len({r["sequence"] for r in rows}),
            frames=len(rows),
            anomaly_points=sum(r["anomaly_in_range"] for r in rows),
        )
        for group, rows in matched.items()
    ]
    reliability = Counter(row["ground_status"] for row in result["instances"])
    tables["reliability"] = [
        dict(ground_status=status, instance_frames=count)
        for status, count in sorted(reliability.items())
    ]
    for name, rows in tables.items():
        write_csv(directory / f"{name}.csv", rows)
    write_csv(
        directory / "index.csv",
        [
            dict(
                file=f"{name}.csv",
                rows=len(rows),
                source=os.path.relpath(Path(output) / "summary.json", directory),
            )
            for name, rows in tables.items()
        ],
    )
    totals = result["totals"]
    (directory / "method.md").write_text(
        "# 单帧真实几何统计\n\n"
        "研究对象为 STU 公开验证集的当前扫描。每个原始扫描只计一次，几何统计不读取模型或预测。\n\n"
        f"本次覆盖 {totals['sequences']} 条序列、{totals['frames']} 帧、"
        f"{totals['visible']} 个实际回波，其中异常回波 {totals['anomaly']} 个。"
        f"官方范围内异常不少于 5 点的帧共 {totals['official_frames']} 个，"
        f"对应 {totals['official_anomaly_points']} 个异常点和 {totals['official_normal_points']} 个正常点。\n\n"
        "逐帧画像保留忽略标签、范围外回波和不满足官方帧门槛的扫描；"
        "这些描述性统计不能直接作为官方指标的分母。"
        "官方评价只使用当前传感器坐标中距离为 2.5 至 50 米、语义非零的点，语义 2 为异常。\n\n"
        "点数与距离联合表包含所有符合官方门槛的原始帧，包括序列开头的帧。"
        "距离分箱为 [2.5,10)、[10,20)、[20,35)、[35,50] 米；"
        "异常点数分箱为 [5,20)、[20,100)、[100,500)、至少 500。\n\n"
        "最近正常点距离和半径 0.5 米的正常邻域使用当前扫描的真实语义标签。"
        "同实例最近邻及半径 0.25 米的邻居数使用当前帧的实例标识。"
        "这些依赖真值的量只用于诊断，不作为模型输入。\n\n"
        "实例可见尺寸和协方差形态至少要求 10 个不同坐标点。"
        "实例标识为零的异常点保留在点级统计中，但不假定其真实实例归属。"
        "尺寸描述已观测点的跨度，不等于物体完整尺寸。\n\n"
        "地面代理使用语义 40、44、48、49、60 的当前扫描回波。"
        "实例中心水平 2 米内至少 20 个支持点；稳健拟合后要求均方根残差不超过 0.05 米、"
        "斜率不超过 20 度、最小水平协方差特征值至少 0.01，且中心位于支持凸包内。"
        "不可靠高度保留缺失和具体原因，不能填零。\n\n"
        "连续分布分别保留观测等权、有效帧等权和有效序列等权结果。"
        "分位数取加权经验分布的逆函数，不平均各序列分位数。"
        "强度保留原始值；局部强度方差由完整标准差分布逐值平方后计算。"
        "CSV 为 UTF-8，空单元格表示缺失或不适用；CSV 本身不保存字体。\n\n"
        "再生成命令："
        "`PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
        ".venv/bin/python -m src.profile --data-root /home/jasongao/Data/STU "
        f"--output {directory.parent.as_posix()} --workers 12`。"
        "进程数应依据运行时资源重新确定。\n",
        encoding="utf-8",
    )
