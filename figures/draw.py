"""Draw the paper's architecture and source observations without running a model.

Run from the repository root: python figures/draw.py
The fixed training example was selected without model scores: both insertion
roles have at least ten valid returns at median range 20--30 m, followed by
lexicographic scene/frame/variant/instance ordering. Placements are independent.
"""

from pathlib import Path
import subprocess
import sys
import tempfile

import ijson
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import point_targets, read_nuscenes

TOKEN = "955bce79c1aa4f7d80e8060d5c255a49"
FILES = (None, f"{TOKEN}_anomaly_r3.npz", f"{TOKEN}_control.npz")
GRAY, ORANGE, TEAL = "#90999F", "#D55E00", "#008877"


def records():
    path = ROOT / "results/data/sequence/train.json"
    with path.open("rb") as stream:
        mapping = list(ijson.items(stream, "mapping.item", use_float=True))
    selected = {}
    with path.open("rb") as stream:
        for row in ijson.items(stream, "records.item", use_float=True):
            if row["token"] != TOKEN:
                continue
            delta = Path(row["delta"]).name if "delta" in row else None
            if delta in FILES:
                if delta in selected:
                    raise ValueError("ambiguous source example")
                selected[delta] = row
            if len(selected) == len(FILES):
                break
    return mapping, [selected[name] for name in FILES]


def main():
    # Embed the requested text font and retain standard mathematical glyphs.
    matplotlib.rcParams.update({
        "font.family": "Times New Roman", "font.size": 8.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "cm",
        "axes.linewidth": .55, "xtick.major.width": .5,
        "ytick.major.width": .5, "xtick.major.size": 2,
        "ytick.major.size": 2, "savefig.facecolor": "white",
    })
    mapping, chosen = records()
    frames = [read_nuscenes(record, mapping) for record in chosen]
    inserted_masks = []
    fig, axes = plt.subplots(2, 3, figsize=(5.5, 3.05),
                             gridspec_kw={"height_ratios": [4, 1]})
    fig.subplots_adjust(left=.092, right=.985, bottom=.24, top=.86,
                        wspace=.22, hspace=.30)
    titles = ("(a) Original scan", "(b) Auxiliary anomaly", "(c) Normal insertion")
    stats = []
    for col, (record, frame) in enumerate(zip(chosen, frames)):
        xyz, target = frame.xyzi[:, :3], point_targets(frame)
        inserted = np.zeros(len(xyz), dtype=bool)
        changed_ignore = inserted.copy()
        if "delta" in record:
            with np.load(record["delta"], allow_pickle=False) as delta:
                slots, labels = delta["slots"], delta["labels"]
                inserted[slots[labels > 0]] = True
                changed_ignore[slots[labels == 0]] = True
        inserted_masks.append(inserted)
        # One common crop is applied to all panels; every retained point is drawn.
        crop = ((xyz[:, 0] >= 0) & (xyz[:, 0] <= 8)
                & (xyz[:, 1] >= -24) & (xyz[:, 1] <= -16)
                & (xyz[:, 2] >= -3) & (xyz[:, 2] <= -1))
        if np.any(inserted & ~crop):
            raise ValueError("the shared crop truncates an inserted object")
        color, marker = (ORANGE, "o") if col == 1 else (TEAL, "^")
        for row, vertical in enumerate((1, 2)):
            ax = axes[row, col]
            background = crop & ~inserted & ~changed_ignore
            ax.scatter(xyz[background, 0], xyz[background, vertical],
                       s=2, c=GRAY, linewidths=0, zorder=1)
            ignored = crop & changed_ignore
            ax.scatter(xyz[ignored, 0], xyz[ignored, vertical],
                       s=8, c=GRAY, marker="x", linewidths=.5, zorder=2)
            ax.scatter(xyz[inserted, 0], xyz[inserted, vertical],
                       s=12, c=color, marker=marker, linewidths=0, zorder=3)
            ax.set_xlim(0, 8)
            ax.set_xticks([0, 4, 8])
            ax.set_ylim((-24, -16) if row == 0 else (-3, -1))
            ax.set_yticks([-24, -20, -16] if row == 0 else [-3, -1])
            ax.set_aspect("equal", adjustable="box")
            ax.spines[["top", "right"]].set_visible(False)
            if col == 0:
                ax.set_ylabel("LiDAR y (m)" if row == 0 else "LiDAR z (m)", labelpad=2)
            else:
                ax.tick_params(labelleft=False)
            if row == 0:
                ax.set_title(titles[col], fontsize=8.5, pad=15)
                ax.tick_params(labelbottom=False)
            else:
                ax.set_xlabel("LiDAR x (m)", labelpad=1)
        valid = inserted & (target >= 0)
        count = int(valid.sum())
        distance = float(np.median(np.linalg.norm(xyz[valid], axis=1))) if count else None
        detail = "Same receiving scan" if not count else f"{count} returns; {distance:.1f} m"
        axes[0, col].text(.5, 1.04, detail, ha="center", va="bottom",
                          transform=axes[0, col].transAxes, fontsize=8)
        stats.append((count, distance))
    handles = [Line2D([], [], color=GRAY, marker=".", linestyle="", label="Background"),
               Line2D([], [], color=GRAY, marker="x", markersize=4, linestyle="", label="Ignored occlusion"),
               Line2D([], [], color=ORANGE, marker="o", markersize=4, linestyle="", label="Auxiliary anomaly"),
               Line2D([], [], color=TEAL, marker="^", markersize=4, linestyle="", label="Inserted normal")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.52,.005),
               ncol=2, frameon=False, fontsize=8, handletextpad=.3, columnspacing=1.2)
    out = Path(__file__).resolve().parent / "data.pdf"
    fig.savefig(out, metadata={"Title": "Paired nuScenes source observations",
                              "Author": "Anonymous authors"})
    plt.close(fig)
    print(f"{out}: scene-0042, token={TOKEN}, inserted_counts_and_ranges={stats}")

    # Scenes show measured source data, not model predictions. Generate them in
    # the TeX build directory so the final vector figure remains self-contained.
    with tempfile.TemporaryDirectory(prefix="ajae-method.") as build:
        subprocess.run([
            "/usr/bin/python3", "-c",
            "import sys,cairosvg; from pathlib import Path; "
            "s=Path(sys.argv[1]).read_text().replace('currentColor','#D86B2B'); "
            "cairosvg.svg2pdf(bytestring=s.encode(),write_to=sys.argv[2])",
            str(ROOT / "figures/assets/flame.svg"), str(Path(build) / "flame.pdf"),
        ], check=True)
        # The same camera and crop preserve spatial context across observations.
        eye = np.array([12., 7., 9.])
        forward = np.array([0., -20., -1.8]) - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0., 0., 1.])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)

        def project(xyz):
            shifted = xyz - eye
            depth = shifted @ forward
            return shifted @ right / depth, shifted @ up / depth, depth

        with PdfPages(Path(build) / "observations.pdf") as pdf:
            for index in (0, 1, 0):
                xyz = frames[index].xyzi[:, :3]
                u, v, depth = project(xyz)
                crop = ((np.abs(xyz[:, 0]) <= 25) & (xyz[:, 1] >= -45)
                        & (xyz[:, 1] <= 5) & (xyz[:, 2] >= -3.5)
                        & (xyz[:, 2] <= 8) & (depth > 1))
                order = np.flatnonzero(crop)[np.argsort(-depth[crop])]
                inserted = inserted_masks[index]
                fig = plt.figure(figsize=(32 / 25.4, 22 / 25.4))
                ax = fig.add_axes([0, 0, 1, 1])
                ax.scatter(u[order], v[order], s=.32, c="#697782", linewidths=0)
                ax.scatter(u[inserted], v[inserted], s=1.3, c=ORANGE, linewidths=0)
                ax.set_xlim(-.85, .85)
                ax.set_ylim(-.57, .5)
                ax.set_aspect("equal")
                ax.set_axis_off()
                pdf.savefig(fig)
                plt.close(fig)
        run = subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error",
                              str(ROOT / "figures/method.tex")],
                             cwd=build, capture_output=True, text=True)
        if run.returncode:
            raise RuntimeError(run.stdout[-3000:])
        if "Missing character" in run.stdout or "Font Warning" in run.stdout:
            raise RuntimeError(run.stdout[-3000:])
        (ROOT / "figures/method.pdf").write_bytes((Path(build) / "method.pdf").read_bytes())
    print(f"{ROOT / 'figures/method.pdf'}: training and inference; source-data scenes")


if __name__ == "__main__":
    main()
