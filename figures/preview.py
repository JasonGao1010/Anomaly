"""Authorized hypothetical figures; every value is illustrative, never measured."""

from pathlib import Path

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


def results():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    fig.subplots_adjust(left=0.078, right=0.985, bottom=0.24, top=0.77, wspace=0.27)
    for row, (name, color, marker) in enumerate(zip(METHODS, COLORS, MARKERS)):
        axes[0].plot(FPR, CONFIDENT_RECALL[row], label=name, color=color,
                     marker=marker, markersize=4.0, linewidth=1.55,
                     markeredgecolor="white", markeredgewidth=0.45)
        axes[1].plot(np.arange(len(OBJECT_BINS)), OBJECT_RECALL[row], color=color,
                     marker=marker, markersize=4.0, linewidth=1.55,
                     markeredgecolor="white", markeredgewidth=0.45)
    for ax in axes:
        style_axes(ax)
    axes[0].set_xscale("log")
    axes[0].set_xticks(FPR, ["0.1", "0.5", "1", "2", "5"])
    axes[0].minorticks_off()
    axes[0].set_xlim(0.08, 6.1)
    axes[0].set_xlabel("Normal false-positive rate (%)", labelpad=5)
    axes[0].set_ylabel("Unknown-point recall (%)", labelpad=4)
    axes[0].set_title("(a) Confident unknown subset", loc="left", pad=9)
    axes[1].set_xticks(np.arange(len(OBJECT_BINS)), OBJECT_BINS)
    axes[1].set_xlim(-0.3, 5.3)
    axes[1].set_xlabel("Evaluated points per anomaly object", labelpad=5)
    axes[1].set_ylabel("Object recall at 1% FPR (%)", labelpad=4)
    axes[1].set_title("(b) Objects in eligible frames", loc="left", pad=9)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.52, 1.003), columnspacing=2.2, handlelength=2.2)
    fig.text(0.5, 0.035,
             "Eligible frames contain ≥5 anomaly points in total; individual objects may contain 1–4.",
             ha="center", fontsize=7.7, color="#485361")
    fig.savefig(ROOT / "preview_results.pdf", metadata=METADATA)
    plt.close(fig)


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
    fig.savefig(ROOT / "preview_cases.pdf", metadata=METADATA)
    plt.close(fig)


if __name__ == "__main__":
    results()
    cases()
