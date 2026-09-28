"""Plot three matched measured runs, or explicitly request the illustrative preview."""

import argparse
import json
from pathlib import Path
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from scipy.stats import t


ROOT = Path(__file__).resolve().parent
FONT_DIR = Path.home() / ".local/share/fonts/windows-report"
for name in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
    font_manager.fontManager.addfont(FONT_DIR / name)
if font_manager.FontProperties(fname=FONT_DIR / "times.ttf").get_name() != "Times New Roman":
    raise RuntimeError("The required Times New Roman font is unavailable")

plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.65,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "savefig.facecolor": "white",
})

METADATA = {
    "Title": "Illustrative SERVE preview — hypothetical, not measured",
    "Subject": "User-authorized simulated display data; not experimental evidence",
    "Keywords": "illustrative, hypothetical, simulated, not measured, SERVE preview",
}
METHODS = ("Appearance only", "Separate objectives", "SERVE")
COLORS = ("#667181", "#2479A7", "#BE4438")
MARKERS = ("o", "s", "D")

# Explicitly supplied hypothetical curves. No samples or uncertainty are invented.
FPR = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
CONFIDENT_RECALL = np.array([
    [7.5, 15.8, 22.1, 31.7, 48.2],
    [26.4, 48.1, 62.7, 74.8, 86.8],
    [49.6, 70.8, 81.6, 89.7, 95.3],
])
OBJECT_BINS = ("1–4", "5–9", "10–19", "20–49", "50–99", "100+")
OBJECT_RECALL = np.array([
    [10, 18, 30, 46, 59, 71],
    [25, 40, 57, 70, 80, 88],
    [43, 59, 72, 83, 90, 94],
])


def style_axes(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#D8DCE1", linewidth=0.55, alpha=0.8)
    ax.tick_params(length=3, pad=3)
    ax.set_ylim(0, 103)
    ax.set_yticks(np.arange(0, 101, 20))


def save(fig, path, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata=metadata)
    plt.close(fig)
    fonts = subprocess.run(["pdffonts", str(path)], check=True, text=True, capture_output=True).stdout.splitlines()[2:]
    # PDF inspection checks the embedded fonts, beyond the plotting configuration.
    if not fonts or any(row.split()[-5] != "yes" or
                        not any(name in row.split()[0] for name in ("TimesNewRoman", "Cmmi", "Cmr"))
                        for row in fonts):
        raise RuntimeError("The PDF must embed Times New Roman text and default mathematical fonts")


def comparisons(paths):
    runs = [json.loads(Path(path).read_text()) for path in paths]
    if len(runs) != 3 or {row.get("seed") for row in runs} != {206, 307, 409}:
        raise ValueError("Measured figures require exactly one comparison for seeds 206, 307 and 409")
    keys = ("manifest_sha256", "split", "official_commit", "normal_points", "unknown_points", "points", "scans",
            "baseline_confidence_threshold")
    for key in keys:
        if key not in runs[0] or any(row.get(key) != runs[0][key] for row in runs):
            raise ValueError("Comparisons must share the same evaluation population and definition: " + key)
    if runs[0]["baseline_confidence_threshold"] != .9:
        raise ValueError("The paper's confident-unknown subset uses appearance confidence >= 0.9")
    if not runs[0]["manifest_sha256"] or runs[0]["split"] not in ("val", "test"):
        raise ValueError("Measured figures require an identified held-out evaluation population")
    methods = ("semantic", "separate", "joint")
    fprs, confidence, objects, actual_fpr = None, [], [], []
    population = runs[0]["instance_coverage"]
    counts = np.asarray(population["instance_counts"])
    if counts.shape != (6,) or (counts < 0).any() or not counts.sum():
        raise ValueError("Instance coverage requires six nonnegative size-bin populations")
    if population["size_bins"] != ["1-4", "5-9", "10-19", "20-49", "50-99", "100+"]:
        raise ValueError("Instance-size bins differ from the paper")
    for run in runs:
        conditions = run.get("conditions", {})
        if conditions.get("matched") is not True or conditions.get("differences"):
            raise ValueError("Only completed matched comparisons can supply a measured figure")
        coverage = run["instance_coverage"]
        if any(coverage[key] != population[key] for key in ("size_bins", "instance_counts", "unassigned_anomaly_points")):
            raise ValueError("Comparisons use different scan-instance populations")
        if (run["confident_unknown_points"] < 0
                or run["points"] != run["normal_points"] + run["unknown_points"]):
            raise ValueError("The compared point populations are empty or inconsistent")
        confident_rows, object_rows, observed_fprs = [], [], []
        for method in methods:
            operating = run["methods"][method]["operating_points"]
            requested = np.asarray([row["requested_fpr"] for row in operating], dtype=float)
            if fprs is None:
                fprs = requested
                if not len(fprs) or not np.isfinite(fprs).all() or (fprs <= 0).any() or (fprs > 1).any() or (np.diff(fprs) <= 0).any():
                    raise ValueError("Normal FPR budgets must be positive and strictly increasing")
                if np.count_nonzero(np.isclose(fprs, .01, atol=1e-12, rtol=0)) != 1:
                    raise ValueError("The instance panel needs a measured 1% normal-FPR operating point")
            if not np.array_equal(requested, fprs):
                raise ValueError("All methods and seeds must share requested FPR budgets")
            recall = np.asarray([row["confident_unknown_recall"] for row in operating], dtype=float)
            recalled = np.asarray([row["confident_unknown_recalled"] for row in operating])
            count = run["confident_unknown_points"]
            expected_recall = recalled / count if count else np.full(len(recalled), np.nan)
            if ((recalled < 0).any() or (recalled > count).any()
                    or not np.allclose(recall, expected_recall, atol=1e-12, rtol=0, equal_nan=True)):
                raise ValueError("Confident-unknown recalls do not match their observed point counts")
            candidates = [row for row in coverage["methods"][method] if np.isclose(row["requested_fpr"], .01, atol=1e-12, rtol=0)]
            if len(candidates) != 1:
                raise ValueError("Instance coverage needs exactly one measured 1% operating point")
            row = candidates[0]
            detected, object_recall = np.asarray(row["detected"]), np.asarray(row["recall"], dtype=float)
            expected = np.divide(detected, counts, out=np.full(6, np.nan), where=counts > 0)
            if (detected.shape != (6,) or (detected < 0).any() or (detected > counts).any()
                    or not np.allclose(object_recall, expected, atol=1e-12, rtol=0, equal_nan=True)):
                raise ValueError("Instance recalls do not match their observed instance counts")
            confident_rows.append(recall)
            object_rows.append(object_recall)
            observed_fprs.append([row["actual_fpr"] for row in operating])
        confidence.append(confident_rows)
        objects.append(object_rows)
        actual_fpr.append(observed_fprs)
    statistics = dict(seeds=[run["seed"] for run in runs], sources=[str(Path(path).resolve()) for path in paths],
                      population={key: runs[0][key] for key in keys}, methods=list(methods),
                      requested_fpr=fprs.tolist(), actual_fpr=actual_fpr,
                      confident_unknown_points=[run["confident_unknown_points"] for run in runs],
                      instance_counts=counts.tolist(), unassigned_anomaly_points=population["unassigned_anomaly_points"],
                      instance_complete=population["unassigned_anomaly_points"] == 0,
                      instance_population="officially evaluated anomaly points with a positive instance ID",
                      reduction="arithmetic mean and sample standard deviation across seeds; units are percentage points")
    for key, values in (("confident_unknown_recall", confidence), ("instance_coverage", objects)):
        values = np.asarray(values) * 100
        summary = {}
        for name, reduced in (("mean", values.mean(0)), ("std", values.std(0, ddof=1))):
            # Empty instance bins stay undefined, never become zero recall.
            serialized = reduced.astype(object)
            serialized[~np.isfinite(reduced)] = None
            summary[name] = serialized.tolist()
        statistics[key] = summary
    return statistics


def results(path, fpr, confident_recall, object_recall, metadata, *, deviations=None, unassigned_anomaly_points=0):
    note = ("Eligible frames contain ≥5 anomaly points in total; individual objects may contain 1–4."
            "\nScore ties can make the achieved false-positive rate lower than the requested budget.")
    if unassigned_anomaly_points:
        note += f"\nObject coverage uses positive instance IDs; {unassigned_anomaly_points:,} anomaly points have no usable ID."
    if not np.isfinite(confident_recall).all() or not np.isfinite(object_recall).all():
        note += "\nUndefined seed-level recalls remain blank; available seeds are not averaged on their own."
    footer = .14 * note.count("\n")
    height = 3.0 + footer
    fig, axes = plt.subplots(1, 2, figsize=(7.0, height))
    fig.subplots_adjust(left=0.078, right=0.985, bottom=(.72 + footer) / height,
                        top=1 - .69 / height, wspace=0.27)
    for row, (name, color, marker) in enumerate(zip(METHODS, COLORS, MARKERS)):
        axes[0].plot(fpr, confident_recall[row], label=name, color=color,
                     marker=marker, markersize=4.0, linewidth=1.55,
                     markeredgecolor="white", markeredgewidth=0.45)
        axes[1].plot(np.arange(len(OBJECT_BINS)), object_recall[row], color=color,
                     marker=marker, markersize=4.0, linewidth=1.55,
                     markeredgecolor="white", markeredgewidth=0.45)
        if deviations is not None:
            axes[0].errorbar(fpr, confident_recall[row], yerr=deviations[0][row], color=color, fmt="none", capsize=2, linewidth=.7)
            axes[1].errorbar(np.arange(len(OBJECT_BINS)), object_recall[row], yerr=deviations[1][row], color=color, fmt="none", capsize=2, linewidth=.7)
    for ax in axes:
        style_axes(ax)
    axes[0].set_xscale("log")
    axes[0].set_xticks(fpr, [f"{value:g}" for value in fpr])
    axes[0].minorticks_off()
    axes[0].set_xlim(fpr.min() * .8, fpr.max() * 1.22)
    axes[0].set_xlabel("Normal false-positive budget (%)", labelpad=5)
    axes[0].set_ylabel("Unknown-point recall (%)", labelpad=4)
    axes[0].set_title("(a) Confident unknown subset", loc="left", pad=9)
    axes[1].set_xticks(np.arange(len(OBJECT_BINS)), OBJECT_BINS)
    axes[1].set_xlim(-0.3, 5.3)
    axes[1].set_xlabel("Evaluated points per anomaly object", labelpad=5)
    axes[1].set_ylabel("Object recall at 1% FPR budget (%)", labelpad=4)
    axes[1].set_title("(b) Objects in eligible frames", loc="left", pad=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.52, 1.003), columnspacing=2.2, handlelength=2.2)
    fig.text(0.5, .105 / height, note,
             ha="center", fontsize=7.7, color="#485361")
    save(fig, path, metadata)


# Hypothetical Student-t distributions over log range, not density over metres.
# Each displayed class has one active component; no real scan/prediction is used.
CASES = (
    dict(title="(a) Normal road", observed=12.1, means=(17.0, 12.0),
         scales=(0.035, 0.025), appearance=(0.70, 0.55)),
    dict(title="(b) Normal car", observed=20.0, means=(20.0, 20.0),
         scales=(0.025, 0.25), appearance=(0.80, 0.80)),
    dict(title="(c) Confused unknown", observed=18.0, means=(25.0, 12.0),
         scales=(0.025, 0.035), appearance=(0.90, 0.10)),
)


def cases():
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.35), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.33, top=0.71, wspace=0.14)
    ranges = np.geomspace(8, 30, 1200)
    class_colors = ("#2479A7", "#CA8C25")
    common_bound = 2 / (np.pi * np.sqrt(3) * 0.001)
    for panel, (ax, case) in enumerate(zip(axes, CASES)):
        densities = []
        for name, color, mean, scale in zip(("Car", "Road"), class_colors,
                                             case["means"], case["scales"]):
            density = t.pdf(np.log(ranges), df=3, loc=np.log(mean), scale=scale)
            observed_density = float(t.pdf(np.log(case["observed"]), df=3,
                                           loc=np.log(mean), scale=scale))
            densities.append(observed_density)
            ax.plot(ranges, density, color=color, linewidth=1.55, label=name)
            ax.fill_between(ranges, density, color=color, alpha=0.06)
        ax.axvline(case["observed"], color="#4B5259", linewidth=1.0,
                   linestyle=(0, (3, 2)), label="Observed range")
        ax.scatter([case["observed"]] * 2, densities, s=19, c=class_colors,
                   edgecolors="white", linewidths=0.4, zorder=5)
        ax.text(0.97, 0.96, f"Observed: {case['observed']:g} m",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.3,
                color="#4B5259")
        ax.set_title(case["title"], loc="left", pad=27)
        ax.set_xlim(8, 30)
        ax.set_ylim(-0.35, 18.4)
        ax.set_xticks([10, 15, 20, 25, 30])
        ax.set_yticks([0, 5, 10, 15])
        ax.set_xlabel("Range (m)", labelpad=4)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#D8DCE1", linewidth=0.55, alpha=0.8)
        ax.tick_params(length=3, pad=3)
        # This is illustrative class support, not a calibrated class probability.
        joint = np.asarray(case["appearance"]) * densities / common_bound
        ax.text(0.0, 1.055,
                f"Appearance (car, road): ({case['appearance'][0]:.2f}, {case['appearance'][1]:.2f})",
                transform=ax.transAxes, fontsize=7.2, ha="left", va="bottom")
        choice = "Road" if joint[1] > joint[0] else "Car"
        label = (f"Best joint support: {choice}" if panel < 2
                 else "Best joint support is small")
        ax.text(0.5, -0.42, label, transform=ax.transAxes,
                ha="center", fontsize=8, fontweight="bold", color="#374452")
    axes[0].set_ylabel(r"Density with respect to $d(\log r)$", labelpad=5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.51, 1.006), columnspacing=2.3, handlelength=2.6)
    fig.text(0.5, 0.097,
             "Common-class multiplication  •  Broad predictions pay a density cost  •  Low joint support raises the rejection score",
             ha="center", fontsize=7.7, color="#485361")
    fig.text(0.5, 0.027,
             "Log-range Student-t densities (3 degrees of freedom); two of 19 classes displayed.",
             ha="center", fontsize=7.7, color="#485361")
    save(fig, ROOT / "preview_cases.pdf", METADATA)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--illustrative", action="store_true", help="explicitly regenerate hypothetical preview figures")
    mode.add_argument("--comparisons", nargs=3, type=Path, metavar="JSON", help="matched comparison.json for seeds 206, 307 and 409")
    parser.add_argument("--output", type=Path, help="measured results PDF; defaults to figures/results.pdf")
    parser.add_argument("--error-bars", action="store_true", help="show measured sample standard deviations")
    args = parser.parse_args()
    if args.illustrative:
        if args.output is not None or args.error_bars:
            parser.error("--output and --error-bars apply only to measured comparisons")
        results(ROOT / "preview_results.pdf", FPR, CONFIDENT_RECALL, OBJECT_RECALL, METADATA)
        cases()
    else:
        output = args.output or ROOT / "results.pdf"
        if output.resolve() in {ROOT / "preview_results.pdf", ROOT / "preview_cases.pdf"}:
            parser.error("measured figures must not overwrite illustrative preview figures")
        summary = comparisons(args.comparisons)
        serialized = json.dumps(summary, ensure_ascii=False, allow_nan=False)
        metadata = dict(Title="SERVE measured matched comparisons", Subject=serialized,
                        Keywords="measured; three-seed mean and sample standard deviation; SERVE")
        confidence, objects = summary["confident_unknown_recall"], summary["instance_coverage"]
        deviations = ([np.asarray(confidence["std"], dtype=float), np.asarray(objects["std"], dtype=float)]
                      if args.error_bars else None)
        results(output, np.asarray(summary["requested_fpr"]) * 100,
                np.asarray(confidence["mean"], dtype=float), np.asarray(objects["mean"], dtype=float),
                metadata, deviations=deviations, unassigned_anomaly_points=summary["unassigned_anomaly_points"])
        print(serialized)
