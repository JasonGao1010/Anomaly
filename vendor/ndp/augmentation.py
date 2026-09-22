"""NDP Perlin Raise, adapted from 343gltysprk/ndp (Apache-2.0).

Upstream: f11dfbe4181db03a6a036d46b52a308f45d0e255, ood_augmentation.py.
Changes: explicit sequence RNG, no import-time I/O, bounded DBSCAN threads,
and unchanged return for an empty uplift (upstream otherwise raises).
"""

from collections import Counter
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

REVISION = "f11dfbe4181db03a6a036d46b52a308f45d0e255"
SEED = 2025

def _fade(t):  # quintic smoothstep
    return t*t*t*(t*(t*6 - 15) + 10)

def _perlin_grid(H, W, res_h, res_w, rng):
    """
    Tileable 2-D Perlin sampled on an HxW grid; res_h/res_w periods across the grid.
    Output in ~[-1,1] before later min-max normalization.
    """
    theta = rng.random((res_h+1, res_w+1)) * 2*np.pi
    g = np.stack([np.cos(theta), np.sin(theta)], axis=-1)  # (Rh+1, Rw+1, 2)

    ys = np.linspace(0, res_h, H, endpoint=False)
    xs = np.linspace(0, res_w, W, endpoint=False)
    xi = xs.astype(int)
    yi = ys.astype(int)
    xf = xs - xi
    yf = ys - yi

    u = _fade(xf)[None, :]      # (1,W)
    v = _fade(yf)[:, None]      # (H,1)

    def gxy(ix, iy):            # wrap
        return g[iy % (res_h+1), ix % (res_w+1)]

    def dot(ix, iy, dx, dy):
        gr = gxy(ix, iy)
        return gr[...,0]*dx + gr[...,1]*dy

    Xf = xf[None, :]
    Yf = yf[:, None]
    n00 = dot(xi[None,:],   yi[:,None],   Xf,     Yf    )
    n10 = dot(xi[None,:]+1, yi[:,None],   Xf-1.0, Yf    )
    n01 = dot(xi[None,:],   yi[:,None]+1, Xf,     Yf-1.0)
    n11 = dot(xi[None,:]+1, yi[:,None]+1, Xf-1.0, Yf-1.0)

    nx0 = n00*(1-u) + n10*u
    nx1 = n01*(1-u) + n11*u
    n = nx0*(1-v) + nx1*v
    return n.astype(np.float32)

def fbm_perlin_grid(H, W, base_res=(3,3), octaves=3, persistence=0.55, lacunarity=2.0, *, rng):
    total = np.zeros((H, W), dtype=np.float32)
    amp = 1.0; amp_sum = 0.0
    rh, rw = float(base_res[0]), float(base_res[1])
    for _ in range(octaves):
        Rh = max(1, int(np.ceil(rh)))
        Rw = max(1, int(np.ceil(rw)))
        total += amp * _perlin_grid(H, W, Rh, Rw, rng)
        amp_sum += amp
        amp *= persistence
        rh *= lacunarity; rw *= lacunarity
    total /= (amp_sum + 1e-8)
    # Normalize to [0,1]
    tmin, tmax = total.min(), total.max()
    if tmax > tmin:
        total = (total - tmin) / (tmax - tmin)
    return total

# ---------------- Bulge-only augmentation (no normals) ----------------
def perlin_raise(points,
                                     labels,
                                     class_id=40,
                                     target_ratio=0.30,     # fraction of patch points to raise
                                     strength=0.40,         # meters: peak uplift after local norm
                                     patch_radius=1.2,      # meters
                                     grid_res=192,          # Perlin grid resolution
                                     base_res=(3,3),
                                     octaves=3,
                                     persistence=0.55,
                                     lacunarity=2.0,
                                     rng=None,
                                     raise_threshold = 0.01,
                                     RAISED_CLASS=2,
                                     debug=True):
    """
    Steps:
      1) Pick a road-centered patch (radius in meters).
      2) Build a Perlin-fBm grid over the patch bounding box in XY (global).
      3) Map each patch point to the grid cell; take its Perlin value n in [0,1].
      4) Keep the top-quantile points (bulge-only set). Locally min–max normalize n there to [0,1].
      5) Add dz = strength * gain to GLOBAL Z ONLY (no normals).

    Returns (points_out, semantic_labels).
    """

    # 1) select road patch
    if rng is None:
        raise ValueError("pass the shared sequence RNG explicitly")
    road_mask = (labels == class_id)
    if not np.any(road_mask):
        if debug: print("[BULGE0] No points with the given class_id.")
        return points, labels

    road_pts = points[road_mask]
    road_idx = np.where(road_mask)[0]

    center = road_pts[rng.integers(len(road_pts)), :3]
    tree = cKDTree(road_pts[:, :3])
    loc = tree.query_ball_point(center, r=patch_radius)
    if len(loc) < 5:
        if debug: print(f"[BULGE0] Sparse patch ({len(loc)} pts). Increase patch_radius.")
        return points, labels
    gi = road_idx[np.asarray(loc, int)]
    X = points[gi, :3].copy()

    # 2) grid over XY bbox (with padding)
    x0, x1 = X[:,0].min(), X[:,0].max()
    y0, y1 = X[:,1].min(), X[:,1].max()
    pad_x = 0.05 * max(1e-3, x1 - x0)
    pad_y = 0.05 * max(1e-3, y1 - y0)
    x0 -= pad_x; x1 += pad_x; y0 -= pad_y; y1 += pad_y

    perlin = fbm_perlin_grid(grid_res, grid_res,
                             base_res=base_res, octaves=octaves,
                             persistence=persistence, lacunarity=lacunarity, rng=rng)

    # 3) sample per-point Perlin via nearest grid cell (fast and robust); could switch to bilinear if desired
    tx = (X[:,0] - x0) / (x1 - x0 + 1e-12) * (grid_res - 1)
    ty = (X[:,1] - y0) / (y1 - y0 + 1e-12) * (grid_res - 1)
    ix = np.clip(np.round(tx).astype(int), 0, grid_res-1)
    iy = np.clip(np.round(ty).astype(int), 0, grid_res-1)
    nval = perlin[iy, ix]  # in [0,1]

    # 4) bulge-only: threshold by high quantile; local min–max in mask; dz >= 0
    if target_ratio <= 0.0:
        mask = np.zeros_like(nval, dtype=bool)
    elif target_ratio >= 1.0:
        mask = np.ones_like(nval, dtype=bool)
    else:
        th = np.quantile(nval, 1.0 - target_ratio)
        mask = nval >= th
        # ensure non-empty
        if mask.sum() == 0:
            # relax slightly
            th = np.quantile(nval, 1.0 - 0.9*target_ratio)
            mask = nval >= th

    dz = np.zeros_like(nval, dtype=np.float32)
    if mask.any():
        sel = nval[mask]
        smin, smax = sel.min(), sel.max()
        if smax > smin + 1e-8:
            gain = (nval - smin) / (smax - smin + 1e-8)
        else:
            gain = np.zeros_like(nval)
        gain = np.clip(gain, 0.0, 1.0)
        dz[mask] = gain[mask] * strength  # strictly upward

    big = (dz >= raise_threshold)
    dz[~big]=0
    # 5) apply global-Z displacement only
    X[:,2] += dz

    out = points.copy()
    out[gi[big], :3] = X[big]

    # Upstream DBSCAN rejects an empty raised set; retain the unchanged scan.
    if not big.any():
        return points, labels

    clusters = (
    DBSCAN(eps=0.1, min_samples=1, n_jobs=1)
    .fit(out[gi[big], :3])
    .labels_
    )
    most_freq = Counter(clusters).most_common(1)[0][0]
    if most_freq>-1:
        cluster_filter = (clusters==most_freq)
        out = points.copy()
        out[gi[big][cluster_filter], :3] = X[big][cluster_filter]

    if debug:
        frac = mask.mean() if len(mask) else 0.0
        print(f"[BULGE0] patch_pts={len(gi)} affected={mask.sum()} ({frac*100:.1f}%) "
              f"dz_max={dz.max():.3f} m, strength={strength}")

    labels[gi[big][cluster_filter]] = RAISED_CLASS
    return out, labels
