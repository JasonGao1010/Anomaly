"""Draw the motivation and current network: python figures/draw.py.

Layer stacks and attention blocks follow ML Visuals conventions; see
assets/LICENSES.md. The input is a real STU training scan; density curves are schematic.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np

OUT = Path(__file__).resolve().parent
INK, BLUE, TEAL, PURPLE, RED = "#263747", "#377EAB", "#20867A", "#8065A8", "#B95245"
OBSERVED = "#F00000"


def motivation(root):
    # A labeled normal car motivates class-specific, rather than ground-only, checks.
    source = root / "train/206"
    xyz = np.fromfile(source / "velodyne/000128.bin", "<f4").reshape(-1, 4)[:, :3]
    packed = np.fromfile(source / "labels/000128.label", "<u4")
    if len(xyz) != len(packed) or not np.isfinite(xyz).all():
        raise ValueError("invalid motivation scan or labels")
    semantic = packed & 65535
    car = packed == ((64 << 16) | 10)
    points = xyz[car]
    center = np.median(points, axis=0)
    center[2] = np.quantile(points[:, 2], .65)
    target_id = np.flatnonzero(car)[np.argmin(np.sum((points-center)**2, axis=1))]
    target = xyz[target_id].astype(float)
    horizontal = np.linalg.norm(xyz[:, :2], axis=1)
    h_target = np.linalg.norm(target[:2])
    azimuth = np.arctan2(xyz[:, 1], xyz[:, 0])
    az_target = np.arctan2(target[1], target[0])
    angle = np.arctan2(np.sin(azimuth-az_target), np.cos(azimuth-az_target))

    # This plane is a geometric illustration from road labels, not a model prediction.
    road = (semantic == 40) & (horizontal > 3) & (horizontal < 20) & (abs(angle) < .55)
    design = np.column_stack((xyz[road, :2], np.ones(road.sum())))
    plane = np.linalg.lstsq(design, xyz[road, 2], rcond=None)[0]
    slope = plane[:2] @ (target[:2]/h_target)
    h_road = plane[2] / (target[2]/h_target-slope)
    if not h_road > h_target > 0:
        raise ValueError("selected ray does not meet the road beyond the normal car")

    fig = plt.figure(figsize=(5.5, 2.65))
    fig.text(.025, .94, "(a) A normal car above the road", fontsize=8, weight="bold")
    fig.text(.515, .94, "(b) Two explanations of one return", fontsize=8, weight="bold")
    view = fig.add_axes([.015, .23, .46, .66])
    crop = ((xyz[:, 0] > -2.5) & (xyz[:, 0] < 6) &
            (xyz[:, 1] > 3.5) & (xyz[:, 1] < 13) &
            (xyz[:, 2] > -3.5) & (xyz[:, 2] < 1.5))
    # Orthographic view of the original points, colored with dataset annotations.
    az, el = np.deg2rad([-72, 24])
    eye = np.array([np.cos(az)*np.cos(el), np.sin(az)*np.cos(el), np.sin(el)])
    right = np.cross(-eye, [0., 0., 1.]); right /= np.linalg.norm(right)
    up = np.cross(right, -eye)
    projection = np.stack((right, up), axis=1)
    ids = np.flatnonzero(crop)
    ids = ids[np.argsort(xyz[ids] @ eye, kind="stable")]
    colors = np.full(len(xyz), "#BCC5CB", dtype="<U7")
    colors[semantic == 40], colors[car] = TEAL, BLUE
    view.scatter(*(xyz[ids] @ projection).T, c=colors[ids], s=.8,
                 linewidths=0, rasterized=True)
    observed = target @ projection
    view.scatter(*observed, s=19, c=OBSERVED, ec="white", lw=.6, zorder=5)
    view.annotate("Observed return", observed, xytext=(.02, .88),
                  textcoords="axes fraction", fontsize=7.5, color=OBSERVED,
                  arrowprops={"arrowstyle":"-", "color":OBSERVED, "lw":.65})
    view.set_aspect("equal"); view.set_axis_off()
    for x, name, color in ((.05, "Car", BLUE), (.19, "Road", TEAL), (.34, "Other", "#8E9BA5")):
        fig.text(x, .22, "● "+name, color=color, fontsize=7.5)
    fig.text(.25, .15, "Real scan · colors from annotations", ha="center", fontsize=7)

    # Both side views use the identical measured ray and normal car, with no learned scores.
    xlim = (-.3, h_road+1.3)
    ground_x = np.linspace(2, xlim[1], 80)
    ground_z = slope*ground_x+plane[2]
    near_car = car & (abs(angle) < .10)
    for row, (bottom, title) in enumerate(((.55, "Road continuation"), (.25, "Car surface"))):
        ax = fig.add_axes([.52, bottom, .465, .255])
        ax.set(xlim=xlim, ylim=(-3.6, .5)); ax.set_axis_off()
        ax.text(0, 1.02, title, transform=ax.transAxes, fontsize=7.7,
                color=TEAL if row == 0 else BLUE, weight="bold")
        ax.plot(ground_x, ground_z, color=TEAL, ls="--", lw=.85)
        ax.scatter(0, 0, marker="s", s=17, c=INK)
        ax.text(.3, .03, "LiDAR", fontsize=6.8)
        if row == 1:
            ax.scatter(horizontal[near_car], xyz[near_car, 2], s=1.5, c=BLUE,
                       linewidths=0, rasterized=True)
        ax.plot([0, h_target], [0, target[2]], color=OBSERVED, lw=.9)
        ax.scatter(h_target, target[2], s=18, c=OBSERVED, ec="white", lw=.5, zorder=5)
        if row == 0:
            z_road = h_road*target[2]/h_target
            ax.plot([h_target, h_road], [target[2], z_road], color=INK, ls=":", lw=.7)
            ax.scatter(h_road, z_road, s=23, facecolors="white", ec=TEAL, lw=.85, zorder=4)
            ax.annotate("Farther return", (h_road, z_road), xytext=(h_road-.3, -.48),
                        ha="center", fontsize=7, color=TEAL,
                        arrowprops={"arrowstyle":"-", "color":TEAL, "lw":.6})
        else:
            ax.annotate("Earlier return", (h_target, target[2]), xytext=(h_target+2.5, -.43),
                        ha="center", fontsize=7, color=BLUE,
                        arrowprops={"arrowstyle":"-", "color":BLUE, "lw":.6})
    fig.text(.75, .15, "Geometric illustration of class hypotheses", ha="center", fontsize=7)
    fig.text(.5, .055, "Which known class explains both the shape and the measured return?",
             ha="center", fontsize=8, weight="bold")
    fig.savefig(OUT / "motivation.pdf", dpi=600, metadata={
        "Title": "Normal class explanations for a real LiDAR return",
        "Author": "Anonymous authors"})
    plt.close(fig)
    print(f"Motivation: 206/000128, car 64, point {target_id}; "
          f"horizontal range {h_target:.3f} m, road intersection {h_road:.3f} m")


def scene(root):
    # Fix the middle training frame independently of labels or model predictions.
    path = root / "train/206/velodyne/000224.bin"
    xyzi = np.fromfile(path, dtype="<f4").reshape(-1, 4)
    if not len(xyzi) or not np.isfinite(xyzi).all():
        raise ValueError(f"invalid original STU scan: {path}")
    xyz = xyzi[:, :3]
    distance = np.linalg.norm(xyz, axis=1)
    actual = np.any(xyz != 0, axis=1)
    # A display crop only: preserve every measured point inside these bounds.
    visible = actual & (distance >= 2.5) & (distance <= 35) & (xyz[:, 2] >= -3) & (xyz[:, 2] <= 8)
    xyz = xyz[visible]
    azimuth, elevation = np.deg2rad([180, 30])
    eye = np.array([np.cos(azimuth)*np.cos(elevation),
                    np.sin(azimuth)*np.cos(elevation), np.sin(elevation)])
    right = np.cross(-eye, [0., 0., 1.])
    right /= np.linalg.norm(right)
    up = np.cross(right, -eye)
    # Orthographic projection with far-to-near drawing preserves scene geometry.
    xyz = xyz[np.argsort(xyz @ eye, kind="stable")]
    uv = np.column_stack((xyz @ right, xyz @ up))
    cmap = LinearSegmentedColormap.from_list("height", ["#345c9b", "#1884a8", "#49a487", "#bea240", "#bd653a"])
    norm = Normalize(-2.1, 5, clip=True)
    print(f"STU 206/000224: {actual.sum():,} actual returns; {len(xyz):,} in display crop")
    return uv, xyz[:, 2], cmap, norm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stu-root", type=Path, default=Path("/home/jasongao/Data/STU"))
    args = parser.parse_args()
    # Register the actual regular, bold, italic, and bold-italic font files.
    fonts = Path.home() / ".local/share/fonts/windows-report"
    for name in ("times.ttf", "timesbd.ttf", "timesi.ttf", "timesbi.ttf"):
        font_manager.fontManager.addfont(fonts / name)
    font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 7,
                         "mathtext.fontset": "cm", "pdf.fonttype": 42,
                         "ps.fonttype": 42, "svg.fonttype": "none",
                         "text.color": INK, "axes.unicode_minus": False})
    motivation(args.stu_root)
    uv, height, cmap, norm = scene(args.stu_root)
    colors = cmap(norm(height))
    # Export a readable standalone view as well as the compact architecture inset.
    view = plt.figure(figsize=(5.5, 4.5))
    view_ax = view.add_axes([.025, .15, .95, .72])
    view_ax.scatter(*uv.T, s=.38, c=colors, linewidths=0)
    view_ax.set_aspect("equal")
    view_ax.set_axis_off()
    view.suptitle("A real LiDAR scan", y=.96, fontsize=11)
    view.text(.5, .91, f"2.5–35 m range; -3–8 m height; {len(uv):,} returns",
              ha="center", fontsize=7)
    color_ax = view.add_axes([.31, .10, .38, .025])
    bar = view.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                       cax=color_ax, orientation="horizontal", extend="both")
    bar.set_label("Height in the LiDAR frame (m)", fontsize=8)
    bar.set_ticks([-2, 0, 2, 5])
    bar.ax.tick_params(labelsize=7, length=2)
    view.savefig(OUT / "scene.png", dpi=320,
                 metadata={"Source":"STU/train/206/velodyne/000224.bin",
                           "Description":"Original returns; display range 2.5–35 m, height -3–8 m; no model output."})
    plt.close(view)
    fig = plt.figure(figsize=(5.5, 3.42))
    ax = fig.add_axes([.008, .008, .984, .984])
    ax.set(xlim=(0, 16), ylim=(0, 10))
    ax.set_axis_off()

    def label(x, y, value, size=6.6, **kw):
        ax.text(x, y, value, ha=kw.pop("ha", "center"), va="center", fontsize=size, **kw)

    def box(x, y, w, h, value="", color=BLUE, fill="#F0F6FA", size=6.5, **kw):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0.03,rounding_size=0.10",
                     ec=color, fc=fill, lw=.65, **kw))
        if value:
            label(x + w / 2, y + h / 2, value, size)

    def wire(points, color=INK, dashed=False, arrow=True):
        # Straight segments keep information paths distinguishable at print scale.
        for k in range(len(points) - 1):
            last = k == len(points) - 2
            ax.add_patch(FancyArrowPatch(points[k], points[k + 1],
                         arrowstyle="-|>" if last and arrow else "-",
                         mutation_scale=6, lw=.65, color=color,
                         linestyle=(0, (3, 2)) if dashed else "-",
                         shrinkA=0, shrinkB=0, zorder=3))

    def tensor(x, y, w, h, color=BLUE, planes=3, rows=5):
        for k in reversed(range(planes)):
            dx, dy = .10 * k, .09 * k
            ax.add_patch(Rectangle((x + dx, y + dy), w, h,
                                  ec=color, fc="white", lw=.55, zorder=4))
            ax.add_patch(Rectangle((x + dx, y + dy), w, h,
                                  ec="none", fc=color, alpha=.18, zorder=4))
            for j in range(1, rows):
                ax.plot([x + dx, x + dx + w], [y + dy + h*j/rows]*2,
                        color=color, lw=.3, zorder=4)
            ax.plot([x + dx + w/2]*2, [y + dy, y + dy + h], color=color, lw=.3, zorder=4)

    # The encoders operate independently; only class vectors are shared.
    box(2.05, 6.10, 9.22, 3.40, color="#ADCBDD", fill="#F5F9FC")
    box(2.05, 2.00, 9.22, 3.02, color="#A9CFC5", fill="#F3F9F6")
    label(2.23, 9.16, "(a) Appearance evidence", 7.8, ha="left", weight="bold")
    label(2.23, 4.69, "(b) Return evidence", 7.8, ha="left", weight="bold")

    center = (uv.min(0) + uv.max(0)) / 2
    scale = min(1.68 / np.ptp(uv[:, 0]), 1.77 / np.ptp(uv[:, 1]))
    inset = (uv - center) * scale + [.87, 5.76]
    ax.scatter(*inset.T, s=.016, c=colors, linewidths=0, rasterized=True)
    label(.88, 6.93, "Single scan", 7.2, weight="bold")
    label(.88, 4.76, r"$X$", 9)
    wire([(1.73, 5.75), (1.85, 5.75), (1.85, 7.80), (2.32, 7.80)])
    wire([(1.85, 5.75), (1.85, 3.40), (2.27, 3.40)])

    # The backbone is shown as a conventional encoder/decoder layer stack.
    for x, y, w, h in [(2.38,7.32,.34,.86),(2.90,7.48,.37,.55),
                       (3.45,7.53,.4,.44),(4.01,7.36,.34,.76)]:
        tensor(x,y,w,h,planes=2,rows=3)
    for start,end in [(2.82,2.89),(3.37,3.43),(3.95,4.00)]:
        wire([(start,7.78),(end,7.78)])
    label(3.41, 8.60, "LitePT-S", 7.2)
    label(3.41, 7.00, "voxel to point", 5.7)
    box(2.38, 6.25, 1.97, .57, "Point detail\nMLP", size=5.8)
    wire([(1.85,6.54),(2.36,6.54)])
    ax.add_patch(Circle((4.87,7.76),.20,ec=BLUE,fc="white",lw=.65))
    label(4.87,7.76,"C",6.3)
    wire([(4.48,7.76),(4.66,7.76)])
    wire([(4.36,6.55),(4.87,6.55),(4.87,7.55)])
    tensor(5.38,7.28,.39,.85,planes=3)
    wire([(5.08,7.76),(5.36,7.76)])
    label(5.68,8.63,r"$\mathbf{h}_i$",8.5)
    label(5.68,6.91,"48-D",6.1)

    # Four appearance modes per class; colors denote different normal classes.
    box(6.43,7.10,2.13,1.19,color=BLUE,fill="white")
    for row,col in enumerate((BLUE,TEAL,PURPLE)):
        for k in range(4):
            ax.add_patch(Circle((6.77+.47*k,7.34+.32*row),.085,ec=col,fc=col,lw=.3))
    label(7.50,8.79,"Class centers",6.8)
    label(7.50,8.42,"4 modes / class",5.4)
    wire([(5.98,7.76),(6.40,7.76)])
    tensor(9.31,7.12,.50,1.15,planes=1,rows=7)
    wire([(8.61,7.76),(9.27,7.76)])
    label(9.56,8.64,r"$a_{ic}$",8.5)
    label(9.56,6.71,"19 class\nsupports",5.9)

    # Shared class vectors are parameters, with no full-scan feature input.
    box(5.63,5.36,3.72,.47,r"Shared class vectors $\mathbf{q}_c$",PURPLE,"#F3EFF8",6.3)
    wire([(7.50,5.85),(7.50,6.45),(7.50,7.07)],color=PURPLE)
    wire([(6.80,5.33),(6.80,4.10)],color=PURPLE)

    # Cell-local encoding precedes neighbor selection and cross-cell attention.
    tensor(2.30,2.95,.50,.83,TEAL,planes=1,rows=4)
    label(2.55,2.43,"Angular\ncells",5.9)
    box(3.35,2.93,1.43,.97,"Cell MLP\nmean + max",TEAL,"#E1F0EA",6.0)
    wire([(2.84,3.40),(3.30,3.40)])
    label(4.06,2.40,"Per cell",5.6)
    # The crossed central token is removed before attention, including all returns.
    for row in range(3):
        for col in range(3):
            x,y=5.06+.20*col,3.10+.20*row
            ax.add_patch(Rectangle((x,y),.18,.18,ec=TEAL,fc="#DFEFE8",lw=.3))
    ax.add_patch(Rectangle((5.26,3.30),.18,.18,ec=RED,fc="white",lw=.55,zorder=5))
    ax.plot([5.26,5.44],[3.30,3.48],color=RED,lw=.65,zorder=5)
    ax.plot([5.26,5.44],[3.48,3.30],color=RED,lw=.65,zorder=5)
    wire([(4.83,3.40),(5.01,3.40)])
    label(5.36,2.39,"Target\nexcluded",5.2,color=RED)
    box(6.13,3.10,1.50,1.09,color=TEAL,fill="#DFEEE7")
    box(6.04,3.01,1.50,1.09,"Cross\nattention\n+ FFN",TEAL,"#C5E0D4",5.8)
    wire([(5.68,3.40),(6.00,3.40)])
    label(6.80,2.44,"2 layers\n3 heads",5.5)
    label(8.00,4.58,"class + angle queries",5.3,color=PURPLE)
    box(7.94,3.03,1.30,1.02,"Student-t\nmixture\nhead",TEAL,"#DFEEE7",5.6)
    wire([(7.58,3.40),(7.90,3.40)])
    label(8.59,2.47,"3 components\nper class",5.6)
    label(8.59,4.27,r"$\mu,\sigma,w$",7.5)

    # Density curves are schematic. The target measurement enters only here.
    wire([(9.29,3.40),(9.56,3.40)])
    t=np.linspace(0,1,100)
    for mu,col in [(.29,BLUE),(.55,TEAL),(.73,PURPLE)]:
        p=sum(w/scale*(1+((t-mean)/scale)**2/3)**-2
              for mean,scale,w in [(mu-.16,.07,.20),(mu,.12,.65),(mu+.14,.05,.15)])
        ax.plot(9.62+1.14*t,3.04+.09*p,color=col,lw=.65)
    ax.plot([10.31,10.31],[3.01,3.94],color=OBSERVED,lw=1.15,zorder=5)
    label(10.58,3.98,r"$z_i$",7.8,color=OBSERVED)
    ax.plot([9.62,10.76],[3.02,3.02],color=INK,lw=.45)
    label(10.21,4.28,"Evaluate density",5.8)
    label(10.85,2.55,r"$p_{ic}$",8.0)
    wire([(.84,4.52),(.84,1.52),(10.31,1.52),(10.31,2.73)],color=OBSERVED)
    label(4.10,1.51,r"Measured log range $z_i$",6.8,color=OBSERVED,
          bbox={"fc":"white","ec":"none","pad":1.5})

    # Evidence combines within a class before any class is selected.
    wire([(9.86,7.76),(11.60,7.76),(11.60,6.05),(11.76,6.05)])
    wire([(10.84,3.45),(11.40,3.45),(11.40,5.56),(11.76,5.56)])
    label(13.47,8.80,"(c) Joint decision",7.6,weight="bold")
    box(11.81,5.18,2.03,1.27,color=PURPLE,fill="#F1ECF6")
    label(12.84,6.16,"Same class",6.3)
    label(12.84,5.78,r"$v_{ic}=a_{ic}(p_{ic}/M)^{\kappa_i}$",6.0)
    label(12.84,5.40,r"$E_{ic}=-\log v_{ic}$",7.0)
    box(14.40,6.65,1.40,1.18,"Semantic\nlabel\n"+r"$\arg\min_c E_{ic}$",PURPLE,"#F7F4FA",6.0)
    box(14.40,4.14,1.40,1.18,"Unknown\nscore\n"+r"$\min_c E_{ic}$",PURPLE,"#F7F4FA",6.0)
    wire([(13.88,5.96),(14.13,5.96),(14.13,7.23),(14.35,7.23)])
    wire([(14.13,5.96),(14.13,4.73),(14.35,4.73)])

    # Dashed paths are normal-data training objectives, retained in the caption.
    box(11.99,2.70,1.72,.66,"Class loss\n"+r"$\mathcal{L}_{\rm joint}$",RED,"#FCF1EC",6.0)
    wire([(12.84,5.15),(12.84,3.40)],color=RED,dashed=True)
    box(14.27,2.70,1.52,.66,"Normal label\n"+r"$Y_i$",RED,"#FCF1EC",5.5)
    wire([(14.23,3.03),(13.76,3.03)],color=RED,dashed=True)
    box(11.99,.23,1.72,.75,"Likelihood loss\n"+r"$\mathcal{L}_{\rm pred}$",RED,"#FCF1EC",5.7)
    wire([(10.84,3.45),(11.06,3.45),(11.06,.61),(11.94,.61)],color=RED,dashed=True)
    wire([(15.03,2.66),(15.03,.61),(13.76,.61)],color=RED,dashed=True)
    label(3.27,.90,"C: concatenate    Solid: inference    Dashed: training",5.8)
    label(3.27,.42,r"$\kappa_i=0$ without context; otherwise $1$",5.8)
    # Parameters of both branches receive the joint classification gradient.
    label(12.80,1.98,"Joint loss trains\nboth evidence branches",5.8,color=RED)

    fig.savefig(OUT / "method.pdf", dpi=900, metadata={"Title":"SERVE network with a real STU input scan",
                                             "Author":"Anonymous authors"})
    plt.close(fig)
    print(OUT / "method.pdf")


if __name__ == "__main__":
    main()
