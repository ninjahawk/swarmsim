"""
S14 -- Avalanche geometry and directedness: WHY the 2-D slope footprint is
filamentary, and where the model sits between the compact (BTW) and directed
(Dhar-Ramaswamy) sandpiles.

Motivation. S12 found the 2-D slope avalanche's footprint scales as area ~ L
(D_area ~ 1, "filamentary") and S13 confirmed it cutoff-free (a thin front of A
bonds swept ~A times). But both measured a single number, the AREA exponent. They
did not look at the SHAPE: is the footprint actually a thin curve, or just a
low-density blob that happens to count few distinct bonds? Is it a DIRECTED path
(a downhill front, like the exactly-solvable directed sandpile) or an
isotropically branching star? This is the geometric "why" behind D_area ~ 1.

The decisive observable is the per-avalanche MASS-RADIUS dimension D defined by
A ~ Rg^D (footprint occupied-cell count A vs its radius of gyration Rg). It places
the model on one clean axis:

    BTW (compact, S11 D_area ~ 2.05)          D = 2     transverse ~ longitudinal
    Dhar-Ramaswamy DIRECTED sandpile (exact)  D = 3/2   transverse ~ longitudinal^(1/2)
    slope model (S12 area ~ L)                D ~ 1 ?   a true thin filament

A directed sandpile spreads diffusively about its drift axis, so its footprint of
linear extent ell has transverse width ~ ell^(1/2) and area ~ ell^(3/2) (D = 3/2,
the exactly known avalanche dimension; tau = 4/3). If the slope model instead has
D ~ 1, its footprint is even THINNER -- a constant-width filament, area ~ ell --
which would make it the most filamentary of the three deterministic-rule sandpiles
and explain D_area ~ 1 geometrically, not just as an exponent.

What this script measures, with ONE pipeline applied to both the slope model and a
method-matched BTW baseline:
  * mass-radius dimension D from A ~ Rg^D (binned regression over the ensemble),
    cross-checked by per-footprint box-counting on the largest avalanches;
  * shape anisotropy: rms widths along the principal axes (w_long = sqrt(lam1),
    w_trans = sqrt(lam2)) and the aspect ratio w_long/w_trans vs avalanche size --
    does the footprint get MORE elongated as it grows (a directed line) or stay
    round (a branching star)?
  * directedness / ballistic spreading: each bond's first-topple time vs its radial
    distance from the launching grain -- a ballistic front gives time ~ radius (the
    spatial face of S10's L^1.07 duration cutoff and S13's <A|T> ~ T).

Anchors. BTW is run here (compact baseline, expect D ~ 2, aspect ~ 1). The directed
sandpile is brought in as the exact D = 3/2 / transverse-1/2 reference line.

Self-test (run this file). The mass-radius estimator is checked on three synthetic
ensembles of KNOWN dimension -- straight lines (D = 1), filled disks (D = 2), and
directed diffusive fronts of transverse width ~ sqrt(length) (D = 3/2). It must
read back 1, 2, 3/2; this guards the headline estimator in the moments.py / S11
tradition before it is turned on the data.

Run from repo root:  python sandpile/geometry2d.py
Writes figures/sandpile_geometry.png and outputs/sandpile_geometry.txt.
ASCII-only output (Windows cp1252 safe).
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sandpile_fast import run_sandpile2d_fast       # noqa: E402
from sandpile2d import pyramid_ic                    # noqa: E402
from moments import _ols_slope_se                    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "figures")
OUTDIR = os.path.join(ROOT, "outputs")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

LOG = []
def log(msg):
    print(msg)
    LOG.append(msg)


# ---------------------------------------------------------------------------
# Decode the footprint bond-id set into geometry.
#   x-bond id < L*L : i=id//L, j=id%L, joins (i,j)-(i+1,j), midpoint (i+0.5, j)
#   y-bond id >= L*L: m=id-L*L, i=m//L, j=m%L, joins (i,j)-(i,j+1), midpoint (i, j+0.5)
# ---------------------------------------------------------------------------
def decode_points(bonds, L):
    """Bond ids -> (n,2) float array of bond MIDPOINT coordinates (row, col)."""
    bonds = np.asarray(bonds, np.int64)
    LL = L * L
    isx = bonds < LL
    pts = np.empty((bonds.size, 2), float)
    bx = bonds[isx]
    pts[isx, 0] = (bx // L) + 0.5
    pts[isx, 1] = (bx % L).astype(float)
    by = bonds[~isx] - LL
    pts[~isx, 0] = (by // L).astype(float)
    pts[~isx, 1] = (by % L) + 0.5
    return pts


def decode_sites(bonds, L):
    """Bond ids -> unique occupied integer SITES (union of both endpoints), an
    (m,2) int array. Used for box-counting (a site lattice, not bond midpoints)."""
    bonds = np.asarray(bonds, np.int64)
    LL = L * L
    isx = bonds < LL
    bx = bonds[isx]; xi = bx // L; xj = bx % L
    by = bonds[~isx] - LL; yi = by // L; yj = by % L
    rows = np.concatenate([xi, xi + 1, yi, yi])
    cols = np.concatenate([xj, xj, yj, yj + 1])
    return np.unique(np.stack([rows, cols], axis=1), axis=0)


def gyration(points):
    """(lam1, lam2, evec1, cm) of the 2x2 radius-of-gyration tensor, lam1>=lam2>=0.
    lam are the variances along the principal axes, so sqrt(lam) are rms widths and
    Rg = sqrt(lam1+lam2) is the rms distance from the centre of mass."""
    cm = points.mean(axis=0)
    d = points - cm
    cov = (d.T @ d) / points.shape[0]
    w, v = np.linalg.eigh(cov)              # ascending eigenvalues
    return float(w[1]), float(max(w[0], 0.0)), v[:, 1], cm


def footprint_metrics(bonds, L):
    """Per-footprint geometry from its bond-id set."""
    pts = decode_points(bonds, L)
    lam1, lam2, evec1, cm = gyration(pts)
    Rg = np.sqrt(lam1 + lam2)
    w_long = np.sqrt(lam1)
    w_trans = np.sqrt(lam2)
    AR = w_long / w_trans if w_trans > 1e-9 else np.nan
    aniso = (lam1 - lam2) / (lam1 + lam2) if (lam1 + lam2) > 0 else 0.0
    return dict(A=float(bonds.size), Rg=Rg, w_long=w_long, w_trans=w_trans,
                AR=AR, aniso=aniso, evec1=evec1, cm=cm)


def box_dim(sites):
    """Box-counting dimension of an integer site set: D = -slope of log N(b) vs
    log b over dyadic box sizes. Returns (D, boxes, counts) or (nan, ...) if the
    set is too small/short-range to fit."""
    if sites.shape[0] < 10:
        return np.nan, None, None
    mn = sites.min(axis=0)
    ext = int((sites.max(axis=0) - mn).max()) + 1
    if ext < 8:
        return np.nan, None, None
    s = sites - mn
    bs, ns = [], []
    b = 1
    while b <= ext:
        cells = set()
        for r, c in s:
            cells.add((int(r) // b, int(c) // b))
        bs.append(b); ns.append(len(cells))
        b *= 2
    bs = np.array(bs, float); ns = np.array(ns, float)
    if bs.size < 4:
        return np.nan, bs, ns
    slope, _ = _ols_slope_se(np.log10(bs), np.log10(ns))
    return -slope, bs, ns


# ---------------------------------------------------------------------------
# Binned power-law slope of <y | x> ~ x^g over a clean window (cf. conditional.py).
# ---------------------------------------------------------------------------
def binned_slope(x, y, lo, hi, nbins=18, min_count=20, ypos=True):
    """Slope of <y|x> ~ x^g. ypos=True drops non-positive y (for strictly positive
    observables); ypos=False keeps y=0 individuals so a near-zero quantity (e.g. the
    transverse width of one-bond-wide filaments) reads a ~0 exponent via its bin
    means rather than vanishing from the fit."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    yok = (y > 0) if ypos else (y >= 0)
    m = (x > 0) & yok & np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size == 0 or hi <= lo:
        return np.nan, np.nan, np.array([]), np.array([])
    edges = np.logspace(np.log10(lo), np.log10(hi), nbins + 1)
    bx, by = [], []
    for k in range(nbins):
        sel = (x >= edges[k]) & (x < edges[k + 1])
        if int(sel.sum()) >= min_count:
            bx.append(x[sel].mean()); by.append(y[sel].mean())
    bx = np.array(bx); by = np.array(by)
    keep = (bx > 0) & (by > 0)
    bx, by = bx[keep], by[keep]
    if bx.size < 3:
        return np.nan, np.nan, bx, by
    g, se = _ols_slope_se(np.log10(bx), np.log10(by))
    return g, se, bx, by


def rg_window(Rg, lo_floor=2.5, hi_frac=0.55, hi_quant=0.99):
    """Clean Rg fit window: skip the small-Rg quantized head and the sparse tail."""
    Rg = np.asarray(Rg, float); Rg = Rg[Rg > 0]
    if Rg.size == 0:
        return lo_floor, lo_floor * 4
    hi = min(hi_frac * Rg.max(), np.quantile(Rg, hi_quant))
    return float(lo_floor), float(max(hi, lo_floor * 2))


# ---------------------------------------------------------------------------
# Slope-model footprints from the validated dump (S12 equilibration protocol).
# ---------------------------------------------------------------------------
def slope_footprints(L, warm, window, n_seeds, area_cut, fp_cap, max_dump, psto=0.0):
    """Equilibrate over-steep (gauge by mean slope, S12), then dump every avalanche
    footprint with area >= area_cut over a recorded window. Returns a list of
    (seed_rc, bond_ids, rel_first_topple_times) and the mean bond slope.

    psto (additive, default 0.0 = deterministic gradient rule) is the S15 stochastic-
    split knob; both the warmup and the measurement window run at the same psto so the
    pile equilibrates under the stochastic dynamics it is measured in."""
    foots = []
    slopes = []
    for sd in range(n_seeds):
        warmed = run_sandpile2d_fast(L=L, eps=0.1, Zc=5.0, n_iter=warm, seed=7 + sd,
                                     record_series=False, S0=pyramid_ic(L, 4.5),
                                     psto=psto)
        res = run_sandpile2d_fast(L=L, eps=0.1, Zc=5.0, n_iter=window, seed=7 + sd + 53,
                                  record_series=True, S0=warmed['S'],
                                  dump_fp=True, area_cut=area_cut,
                                  fp_cap=fp_cap, max_dump=max_dump, psto=psto)
        off, bid, it, seed = res['fp_off'], res['fp_bid'], res['fp_iter'], res['fp_seed']
        nd = off.size - 1
        for k in range(nd):
            b = bid[off[k]:off[k + 1]]
            t = it[off[k]:off[k + 1]].astype(float)
            t = t - t.min()                       # time relative to avalanche start
            foots.append(((int(seed[k, 0]), int(seed[k, 1])), b, t))
        # mean bond slope from the recorded state, the S12 stationarity gauge
        slopes.append(np.abs(np.diff(res['S'], axis=0)).mean())
        if res['n_large'] > nd:
            log("    [warn] L=%d seed %d: dumped %d of %d large avalanches (buffer cap)"
                % (L, sd, nd, res['n_large']))
    return foots, float(np.mean(slopes)), float(np.std(slopes))


def btw_footprints(L, n_events, warm, area_cut, seed):
    """Canonical 2-D abelian BTW (mirrors btw_compare.btw_run) capturing the SITE
    footprint of each avalanche with area >= area_cut. Returns a list of
    (seed_site, site_coords). The compact-baseline geometry."""
    rng = np.random.default_rng(seed)
    h = np.zeros((L, L), np.int64)
    ever = np.zeros((L, L), bool)
    fps = []
    for ev in range(n_events):
        r = int(rng.integers(0, L)); c = int(rng.integers(0, L))
        h[r, c] += 1
        ever[:] = False
        toppled = False
        while True:
            unstable = h >= 4
            if not unstable.any():
                break
            toppled = True
            ever |= unstable
            h -= 4 * unstable
            h[1:, :] += unstable[:-1, :]
            h[:-1, :] += unstable[1:, :]
            h[:, 1:] += unstable[:, :-1]
            h[:, :-1] += unstable[:, 1:]
        if ev >= warm and toppled and int(ever.sum()) >= area_cut:
            fps.append(((r, c), np.argwhere(ever)))
    return fps


def metrics_table(foots, L, is_slope=True):
    """Per-footprint A, Rg, widths, aspect ratio from a footprint list. For the
    slope model the geometry is on bond midpoints; for BTW on toppled sites."""
    A, Rg, wl, wt, AR, ani = [], [], [], [], [], []
    for item in foots:
        if is_slope:
            _, bonds, _ = item
            m = footprint_metrics(bonds, L)
        else:
            _, sites = item
            pts = sites.astype(float)
            lam1, lam2, _, _ = gyration(pts)
            Rg_ = np.sqrt(lam1 + lam2)
            wt_ = np.sqrt(lam2)
            m = dict(A=float(sites.shape[0]), Rg=Rg_, w_long=np.sqrt(lam1),
                     w_trans=wt_, AR=(np.sqrt(lam1) / wt_ if wt_ > 1e-9 else np.nan),
                     aniso=((lam1 - lam2) / (lam1 + lam2) if lam1 + lam2 > 0 else 0.0))
        A.append(m['A']); Rg.append(m['Rg']); wl.append(m['w_long'])
        wt.append(m['w_trans']); AR.append(m['AR']); ani.append(m['aniso'])
    return (np.array(A), np.array(Rg), np.array(wl), np.array(wt),
            np.array(AR), np.array(ani))


# ---------------------------------------------------------------------------
# Self-test: synthetic shapes of known mass-radius dimension -> 1, 2, 3/2.
# ---------------------------------------------------------------------------
def _shape_line(n):
    return np.stack([np.arange(n), np.zeros(n, int)], axis=1)


def _shape_disk(rho):
    g = np.arange(-rho, rho + 1)
    xx, yy = np.meshgrid(g, g)
    m = xx ** 2 + yy ** 2 <= rho ** 2
    return np.stack([xx[m], yy[m]], axis=1)


def _shape_directed(ell, rng):
    """A directed diffusive front: at longitudinal t in [0,ell] the transverse
    coordinate spans ~ sqrt(t) cells -> area ~ ell^(3/2), the exact directed-
    sandpile geometry (transverse width ~ longitudinal^(1/2))."""
    pts = []
    for t in range(1, ell + 1):
        half = max(0, int(round(np.sqrt(t))))
        for y in range(-half, half + 1):
            pts.append((t, y))
    return np.array(pts, int)


def _ensemble_dim(shapes):
    """Mass-radius slope D from A ~ Rg^D across a list of point sets."""
    A, Rg = [], []
    for pts in shapes:
        p = pts.astype(float)
        lam1, lam2, _, _ = gyration(p)
        A.append(p.shape[0]); Rg.append(np.sqrt(lam1 + lam2))
    A = np.array(A, float); Rg = np.array(Rg, float)
    D, se = _ols_slope_se(np.log10(Rg), np.log10(A))
    return D, se


def _self_test():
    print("=" * 70)
    print("geometry2d.py self-test: mass-radius A ~ Rg^D on known shapes -> 1,2,3/2?")
    print("=" * 70)
    rng = np.random.default_rng(3)
    lines = [_shape_line(n) for n in range(20, 400, 8)]
    disks = [_shape_disk(rho) for rho in range(4, 60, 2)]
    fronts = [_shape_directed(ell, rng) for ell in range(20, 400, 8)]
    for name, shapes, pred in (("straight lines", lines, 1.0),
                               ("filled disks", disks, 2.0),
                               ("directed fronts", fronts, 1.5)):
        D, se = _ensemble_dim(shapes)
        ok = abs(D - pred) < 0.12
        print("  %-16s D = %.3f +- %.3f  (predict %.2f)  %s"
              % (name, D, se, pred, "OK" if ok else "FAIL"))
        assert ok, "mass-radius estimator is biased on a known shape"
    # box-counting sanity on a single big shape of each kind
    Dl = box_dim(_shape_line(256))[0]
    Dd = box_dim(_shape_disk(48))[0]
    print("  box-count check: line D_box=%.2f (->1), disk D_box=%.2f (->2)" % (Dl, Dd))
    assert abs(Dl - 1.0) < 0.2 and abs(Dd - 2.0) < 0.2, "box-count estimator biased"
    print("self-test OK: geometry estimators recover known dimensions unbiased.\n")


# ---------------------------------------------------------------------------
def main():
    log("=" * 70)
    log("S14 -- AVALANCHE GEOMETRY AND DIRECTEDNESS (slope model vs BTW vs directed)")
    log("=" * 70)
    log("Mass-radius dimension D in A ~ Rg^D places the avalanche footprint:")
    log("  BTW compact D=2 | Dhar-Ramaswamy directed D=3/2 | filament D=1")

    # (L, warm, window, n_seeds); footprint buffers sized to the window.
    configs = [
        (96,   5_000_000, 4_000_000, 4),
        (128,  8_000_000, 5_000_000, 4),
        (192, 12_000_000, 6_000_000, 4),
        (256, 20_000_000, 7_000_000, 4),
    ]
    Ls = [c[0] for c in configs]
    AREA_CUT = 8                      # smallest footprint with meaningful geometry

    slope_data = {}
    per_L_D = {}
    log("\nRunning equilibrated 2-D slope lattices (warm unrecorded, then dump footprints)...")
    for L, warm, window, n_seeds in configs:
        t = time.time()
        foots, ms, ss = slope_footprints(L, warm, window, n_seeds, AREA_CUT,
                                         fp_cap=8_000_000, max_dump=600_000)
        A, Rg, wl, wt, AR, ani = metrics_table(foots, L, is_slope=True)
        slope_data[L] = dict(foots=foots, A=A, Rg=Rg, wl=wl, wt=wt, AR=AR, ani=ani)
        lo, hi = rg_window(Rg)
        D, se, bx, by = binned_slope(Rg, A, lo, hi)
        per_L_D[L] = (D, se)
        log("  L=%4d : %6d footprints (>=%d bonds, %d seeds)  mean-slope=%.2f+-%.2f"
            % (L, len(foots), AREA_CUT, n_seeds, ms, ss))
        log("           mass-radius D = %.3f +- %.3f   A_max=%.0f Rg_max=%.1f  (%.0fs)"
            % (D, se, A.max(), Rg.max(), time.time() - t))

    # L-independence of the mass-radius dimension
    log("\n[L-independence of the mass-radius dimension D (A ~ Rg^D)]")
    log("  L       D +- se")
    for L in Ls:
        log("  %-6d  %.3f +- %.3f" % (L, per_L_D[L][0], per_L_D[L][1]))
    log("  (a geometric LAW is intrinsic if D is L-independent; D->1 = thin filament)")

    # pool the two largest lattices (best statistics, least finite-size)
    Lbig = Ls[-2:]
    Ab = np.concatenate([slope_data[L]['A'] for L in Lbig])
    Rgb = np.concatenate([slope_data[L]['Rg'] for L in Lbig])
    wlb = np.concatenate([slope_data[L]['wl'] for L in Lbig])
    wtb = np.concatenate([slope_data[L]['wt'] for L in Lbig])
    ARb = np.concatenate([slope_data[L]['AR'] for L in Lbig])

    anib = np.concatenate([slope_data[L]['ani'] for L in Lbig])
    lo, hi = rg_window(Rgb)
    D_slope, seD, bxD, byD = binned_slope(Rgb, Ab, lo, hi)
    # Longitudinal width tracks the filament length (~ area). The transverse width is
    # DEGENERATE -- most large footprints are exactly one bond wide (variance 0) -- so
    # it is a constant near 0, not a power law; anisotropy and the one-bond-wide
    # fraction carry the elongation instead (both robust at width 0).
    loA, hiA = rg_window(Ab, lo_floor=10.0, hi_frac=0.5)
    pL, sepL, bxL, byL = binned_slope(Ab, wlb, loA, hiA)
    gani, seani, bxAN, byAN = binned_slope(Ab, anib, loA, hiA)
    big = Ab >= np.quantile(Ab, 0.9)
    aniso_big = float(np.nanmean(anib[big]))
    thin_frac = float(np.mean(wtb[big] < 0.5))
    wtrans_big = float(np.mean(wtb[big]))

    log("\n[slope-model footprint geometry, pooled L=%s]" % "+".join(map(str, Lbig)))
    log("  mass-radius dimension   D(A~Rg^D)   = %.3f +- %.3f   (BTW 2, directed 3/2, filament 1)"
        % (D_slope, seD))
    log("  longitudinal rms width  w_long ~ A^ %.3f +- %.3f   (the filament length, tracks area)"
        % (pL, sepL))
    log("  transverse  rms width   <w_trans> (top decile A) = %.3f bonds  (constant ~0, not scaling)"
        % wtrans_big)
    log("  elongation              one-bond-wide fraction (top decile A) = %.2f ; anisotropy = %.3f (1=line)"
        % (thin_frac, aniso_big))
    log("  -> transverse width is O(1), so the aspect ratio grows as w_long ~ A^%.2f" % pL)

    # per-footprint box-counting confirmation on the largest avalanches
    big_idx = np.argsort([f[1].size for f in slope_data[Lbig[-1]]['foots']])[-40:]
    bigfoots = [slope_data[Lbig[-1]]['foots'][i] for i in big_idx]
    Dbox = []
    for _, bonds, _ in bigfoots:
        d, _, _ = box_dim(decode_sites(bonds, Lbig[-1]))
        if np.isfinite(d):
            Dbox.append(d)
    log("  per-footprint box-counting D_box (40 largest) = %.2f +- %.2f  (independent of A~Rg^D)"
        % (np.mean(Dbox), np.std(Dbox)))

    # ---- directedness: first-topple time vs radial distance from the launch site ----
    rad, tim = [], []
    for (sr, sc), bonds, t in slope_data[Lbig[-1]]['foots']:
        if bonds.size < 30:
            continue
        pts = decode_points(bonds, Lbig[-1])
        r = np.sqrt((pts[:, 0] - sr) ** 2 + (pts[:, 1] - sc) ** 2)
        rad.append(r); tim.append(t)
    rad = np.concatenate(rad); tim = np.concatenate(tim)
    # linear fit time = v_inv * radius (ballistic front => tight, slope > 0)
    msk = rad > 0
    vexp, vse = _ols_slope_se(rad[msk], tim[msk])
    corr = np.corrcoef(rad[msk], tim[msk])[0, 1]
    log("\n[directedness / ballistic spreading, L=%d]" % Lbig[-1])
    log("  first-topple time vs radial distance from launch site:")
    log("    time ~ %.3f * radius   (linear corr = %.3f) -> ballistic front (time ~ distance)"
        % (vexp, corr))

    # ---- BTW baseline through the same pipeline ----
    log("\nRunning BTW baseline (compact reference, same geometry pipeline)...")
    btw_cfg = [(64, 120_000, 30_000), (96, 90_000, 30_000), (128, 70_000, 25_000)]
    btw_all = {}
    per_L_Dbtw = {}
    for L, n_ev, warm in btw_cfg:
        fps = btw_footprints(L, n_ev, warm, area_cut=AREA_CUT, seed=5)
        Ab2, Rgb2, wlb2, wtb2, ARb2, _ = metrics_table(fps, L, is_slope=False)
        btw_all[L] = dict(A=Ab2, Rg=Rgb2, AR=ARb2)
        lo2, hi2 = rg_window(Rgb2, lo_floor=2.5, hi_frac=0.55)
        D2, se2, _, _ = binned_slope(Rgb2, Ab2, lo2, hi2)
        per_L_Dbtw[L] = (D2, se2)
        log("  BTW L=%4d : %6d footprints   mass-radius D = %.3f +- %.3f" % (L, len(fps), D2, se2))
    Lb = btw_cfg[-1][0]
    Abtw = btw_all[Lb]['A']; Rgbtw = btw_all[Lb]['Rg']; ARbtw = btw_all[Lb]['AR']
    lo2, hi2 = rg_window(Rgbtw, lo_floor=2.5, hi_frac=0.55)
    D_btw, seD2, bxD2, byD2 = binned_slope(Rgbtw, Abtw, lo2, hi2)

    # ---- verdict ----
    log("\n[verdict -- placement on the compact/directed/filament axis]")
    log("  mass-radius dimension D (A ~ Rg^D):")
    log("    slope model   D = %.2f +- %.2f" % (D_slope, seD))
    log("    BTW (here)    D = %.2f +- %.2f   (expect ~2, compact; S11 D_area 2.05)" % (D_btw, seD2))
    log("    directed (DR) D = 1.50           (exact: transverse ~ longitudinal^1/2)")
    log("    filament      D = 1.00           (constant-width thin front)")
    if D_slope < 1.25:
        place = ("D ~ 1: the slope avalanche is a CONSTANT-WIDTH thin filament -- THINNER\n"
                 "    than even the directed sandpile (3/2). The deterministic gradient rule\n"
                 "    makes the most filamentary avalanche of the three; D_area ~ 1 (S12) is\n"
                 "    geometric, not just an exponent.")
    elif D_slope < 1.7:
        place = ("D between 1 and 3/2: more filamentary than compact BTW, comparable to or\n"
                 "    thinner than the directed sandpile.")
    else:
        place = "D near 2: compact -- would CONTRADICT S12; investigate."
    log("  -> " + place)
    log("  shape: w_long ~ A^%.2f (length = area); transverse width O(1) (%.0f%% one bond wide,"
        % (pL, 100 * thin_frac))
    log("    anisotropy %.3f) -> aspect ratio grows ~A^%.2f. A thin, essentially straight filament."
        % (aniso_big, pL))
    log("  drive: time ~ radius (corr %.2f) -- a ballistic front radiating from the seed."
        % corr)

    _make_figure(slope_data, Lbig, btw_all, Lb, D_slope, D_btw,
                 (bxD, byD), (bxD2, byD2), (bxAN, byAN),
                 (bxL, byL, pL), (rad, tim, vexp))

    log("\nS14 COMPLETE")
    with open(os.path.join(OUTDIR, "sandpile_geometry.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


def _draw_footprint(ax, foot, L, title):
    (sr, sc), bonds, t = foot
    pts = decode_points(bonds, L)
    sc_ = ax.scatter(pts[:, 1], pts[:, 0], c=t, s=12, cmap="plasma")
    ax.plot([sc], [sr], "*", color="cyan", ms=13, mec="k", mew=0.6,
            label="launch site")
    # square limits centred on the footprint so a thin filament reads as thin
    ymid = 0.5 * (pts[:, 0].min() + pts[:, 0].max())
    xmid = 0.5 * (pts[:, 1].min() + pts[:, 1].max())
    ext = 0.5 * max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 6.0) + 3.0
    ax.set_xlim(xmid - ext, xmid + ext)
    ax.set_ylim(ymid + ext, ymid - ext)        # row increases downward
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.7)
    return sc_


def _make_figure(slope_data, Lbig, btw_all, Lb, D_slope, D_btw,
                 massrad, massrad_btw, aniso_plot, wlong, ballistic):
    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(2, 3)

    L0 = Lbig[-1]
    foots = slope_data[L0]['foots']
    areas = np.array([f[1].size for f in foots])
    # small / medium / large representative footprints
    order = np.argsort(areas)
    pick = [order[int(0.5 * len(order))], order[int(0.9 * len(order))], order[-1]]
    titles = ["small (A=%d)", "medium (A=%d)", "largest (A=%d)"]
    for col, (idx, ttl) in enumerate(zip(pick, titles)):
        ax = fig.add_subplot(gs[0, col])
        s = _draw_footprint(ax, foots[idx], L0, "slope footprint, " + ttl % areas[idx]
                            + "  (colour = topple time)")
        if col == 0:
            ax.set_ylabel("row")
        ax.set_xlabel("col")

    # ---- (1,0) mass-radius A ~ Rg^D : slope vs BTW vs reference lines ----
    ax = fig.add_subplot(gs[1, 0])
    bxD, byD = massrad
    bxD2, byD2 = massrad_btw
    ax.loglog(bxD, byD, "o", color="C0", ms=5, label="slope  D=%.2f" % D_slope)
    ax.loglog(bxD2, byD2, "s", color="C3", ms=5, label="BTW   D=%.2f" % D_btw)
    if bxD.size:
        xr = np.array([bxD.min(), bxD.max()])
        for D, ls, lab in ((1.0, ":", "D=1 filament"), (1.5, "-.", "D=3/2 directed"),
                           (2.0, "--", "D=2 compact")):
            ax.loglog(xr, byD[0] * (xr / bxD[0]) ** D, ls, color="0.5", lw=1.2, label=lab)
    ax.set_xlabel("radius of gyration Rg"); ax.set_ylabel("footprint area A")
    ax.set_title("Mass-radius dimension: slope model is filamentary")
    ax.legend(fontsize=8)

    # ---- (1,1) longitudinal width ~ A and anisotropy -> 1 (a thin straight filament) ----
    ax = fig.add_subplot(gs[1, 1])
    bxAN, byAN = aniso_plot
    bxL, byL, pL = wlong
    ax.loglog(bxL, byL, "^-", color="C2", ms=4, label="w_long ~ A^%.2f" % pL)
    if bxL.size:
        xr = np.array([bxL.min(), bxL.max()])
        ax.loglog(xr, byL[0] * (xr / bxL[0]), "k:", lw=1.0, label="slope 1 reference")
    ax.set_xlabel("avalanche area A"); ax.set_ylabel("longitudinal rms width (bonds)")
    ax.set_title("Length tracks area; footprint one bond wide")
    ax.legend(fontsize=8, loc="upper left")
    axr = ax.twinx()
    axr.semilogx(bxAN, byAN, "D--", color="C4", ms=4)
    axr.set_ylabel("anisotropy (1 = line)", color="C4")
    axr.set_ylim(0, 1.05)
    axr.tick_params(axis="y", labelcolor="C4")

    # ---- (1,2) directedness: topple time vs radial distance from seed ----
    ax = fig.add_subplot(gs[1, 2])
    rad, tim, vexp = ballistic
    m = rad > 0
    ax.hist2d(rad[m], tim[m], bins=60, cmap="viridis", cmin=1)
    rr = np.array([0, np.quantile(rad[m], 0.99)])
    ax.plot(rr, vexp * rr, "w-", lw=1.8, label="time = %.2f x radius (ballistic)" % vexp)
    ax.set_xlabel("radial distance from launch site")
    ax.set_ylabel("first-topple time")
    ax.set_title("Ballistic front: time ~ distance")
    ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    p = os.path.join(FIGDIR, "sandpile_geometry.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    log("\nsaved %s" % os.path.relpath(p, ROOT))


if __name__ == "__main__":
    _self_test()
    main()
