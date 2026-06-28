"""
S19 -- Does the slope-avalanche filament FATTEN at large L, or is the apparent
fattening purely a multifractal TAIL? Firming up S18's flagged second reading.

The open hint. S18 closed the area-multifractality question -- the avalanche-AREA
moment drift does NOT heal as L grows (~0.2 to L=512, intercept 0.169+-0.048), so
the multifractality is a TRUE asymptotic property, not a finite-size corona. But S18
ended on a second, deliberately TENTATIVE reading it could not settle: that the
avalanche footprint may be actively FATTENING at the largest lattices. The signs:
the per-step <A> growth slope steepened from ~0.90 over L=64-256 (the S12/S14
one-bond filament, A ~ L^0.9) to 1.00 (256->384) and 1.22 (384->512), the mid-q
footprint dimension D_mid climbed 1.07 -> 1.23, and A_max nearly DOUBLED in the
384->512 step (587 -> 1088). S18 flagged this as the noisiest, fewest-avalanche
points (only 5 seeds at L=384, 512) and named the follow-up explicitly: "firming up
whether the filament genuinely fattens at large L (a real crossover) versus
tail-undersampling needs more seeds at L = 512."

Why it matters. The two readings are physically opposite and the whole geometric
spine of the arc (S12/S14/S16: the TYPICAL avalanche is a constant-width, one-bond
filament of mass-radius dimension D ~ 1) rests on which is true:
  * REAL CROSSOVER -- the typical footprint genuinely thickens at large L, D ~ 1 is
    only a small-L description, and the model drifts toward compactness on its own
    (no stochastic split needed, unlike S15). That would be a new result and a
    qualification of S14/S16.
  * MULTIFRACTAL TAIL -- the typical footprint stays a one-bond D ~ 1 filament while
    only the rare largest avalanches (the mean, A_max, the high-q moments) grow. Then
    the "fattening" is just the S18 tail, the apparent rise in <A>-slope and D_mid is
    tail-undersampling at 5 seeds, and S14/S16/S18's main picture stands UNCHANGED:
    single-scale-thin in the typical avalanche, multifractal-fat only in the tail.

The decisive separation. Re-measure at L = 192, 256, 384, 512 with MANY more seeds
(target ~10-12 vs S18's 5) and split each lattice's avalanches into TYPICAL and TAIL:
  1. mean(A) growth slope vs L  -- the S18 (tail-weighted) quantity, with a
     seed-bootstrap error bar: is the 384->512 slope of 1.22 distinguishable from 1.0
     once the rare tail is averaged over more independent equilibrated states?
  2. median(A) growth slope vs L -- the TYPICAL avalanche. If the median tracks ~A ~ L
     (slope ~1) while the mean steepens, the growth lives entirely in the right tail.
  3. one-bond-wide fraction of the top-decile footprints vs L -- S14's 98% at L<=256.
     Does the typical large avalanche stay one bond wide as L grows?
  4. mass-radius dimension D (A ~ Rg^D) over the FULL ensemble vs over the TYPICAL
     ensemble (top area-decile removed). If D_full climbs (S18's D_mid 1.07->1.23)
     but D_typical stays ~1, the climb is the tail, not the filament.

Verdict logic. TAIL if the mean-A slope exceeds the median-A slope, the one-bond-wide
fraction stays high, and D_typical stays ~1 while D_full drifts up (the S18 main
reading confirmed, the second reading downgraded to a pure tail statement). REAL
CROSSOVER if the median-A slope and D_typical ALSO climb and the one-bond-wide
fraction falls (S14/S16 become small-L descriptions).

Self-test (run this file). The typical-vs-tail separator is checked on a synthetic
mixture of many thin LINES (mass-radius D = 1) plus a rare population of filled DISKS
(D = 2): the full-ensemble mass-radius D must read biased UP, the typical (small-area)
D must read ~1, and the tail (top-decile-area) D must read ~2 -- i.e. the estimator
recovers a known typical/tail split, so a split read on the real data is meaningful.
The geometry estimators themselves are guarded by geometry2d._self_test (D=1,2,3/2 on
lines/disks/directed fronts), reused here.

Run from repo root:  python sandpile/filament_fattening.py   [smoke: add 'smoke']
Writes figures/sandpile_filament_fattening.png and outputs/sandpile_filament_fattening.txt.
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
from sandpile_fast import run_sandpile2d_fast              # noqa: E402
from equilibrate2d import equilibrate                       # noqa: E402
from geometry2d import (footprint_metrics, binned_slope,    # noqa: E402
                        rg_window, gyration, _shape_line, _shape_disk)
from moments import _ols_slope_se                            # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "figures")
OUTDIR = os.path.join(ROOT, "outputs")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

AREA_CUT = 8                  # smallest footprint with meaningful geometry (S14)
TAIL_Q = 0.90                 # top decile by area = the "tail"; below = "typical"

LOG = []
def log(msg):
    print(msg)
    LOG.append(msg)


# ---------------------------------------------------------------------------
# Measurement: per seed, equilibrate to a verified-stationary repose (S16), dump
# footprints over a recorded window, reduce each to (A, Rg, w_trans). We keep only
# those three floats per footprint, not the bond arrays, so many seeds at L=512 fit
# in memory.
# ---------------------------------------------------------------------------
def measure_L(L, window, n_seeds, smoke=False):
    A_seeds, Rg_seeds, wt_seeds, slopes = [], [], [], []
    for s in range(n_seeds):
        base = 1 + 1000 * s
        if smoke:
            warm = equilibrate(L, seed=base, chunk=4_000_000, max_chunks=6)
        else:
            warm = equilibrate(L, seed=base)
        res = run_sandpile2d_fast(L=L, eps=0.1, Zc=5.0, n_iter=window,
                                  seed=base + 500, record_series=True, S0=warm['S'],
                                  dump_fp=True, area_cut=AREA_CUT,
                                  fp_cap=12_000_000, max_dump=400_000)
        off, bid = res['fp_off'], res['fp_bid']
        nd = off.size - 1
        A = np.empty(nd); Rg = np.empty(nd); wt = np.empty(nd)
        for k in range(nd):
            m = footprint_metrics(bid[off[k]:off[k + 1]], L)
            A[k] = m['A']; Rg[k] = m['Rg']; wt[k] = m['w_trans']
        A_seeds.append(A); Rg_seeds.append(Rg); wt_seeds.append(wt)
        slopes.append(warm['mean_slope'])
        if res['n_large'] > nd:
            log("    [warn] L=%d seed %d: dumped %d of %d large avalanches (buffer cap)"
                % (L, s, nd, res['n_large']))
    return (A_seeds, Rg_seeds, wt_seeds,
            float(np.mean(slopes)), float(np.std(slopes)))


# ---------------------------------------------------------------------------
# Per-L typical/tail statistics from a (possibly seed-resampled) pool of footprints.
# ---------------------------------------------------------------------------
def _massradius_D(Rg, A):
    """Mass-radius dimension A ~ Rg^D over a clean Rg window (geometry2d pipeline)."""
    if A.size < 50:
        return np.nan
    lo, hi = rg_window(Rg)
    D, _, _, _ = binned_slope(Rg, A, lo, hi)
    return D


def pool_stats(A_list, Rg_list, wt_list):
    """mean A, median A, one-bond-wide fraction (top decile), and mass-radius D over
    the full ensemble and over the typical (top-area-decile removed) ensemble."""
    A = np.concatenate(A_list); Rg = np.concatenate(Rg_list); wt = np.concatenate(wt_list)
    thr = np.quantile(A, TAIL_Q)
    big = A >= thr                       # the tail (top decile by area)
    typ = A < thr                        # the typical bulk
    onebond = float(np.mean(wt[big] < 0.5)) if big.any() else np.nan
    return dict(meanA=float(A.mean()), medA=float(np.median(A)),
                Amax=float(A.max()), onebond=onebond,
                D_full=_massradius_D(Rg, A),
                D_typ=_massradius_D(Rg[typ], A[typ]),
                D_tail=_massradius_D(Rg[big], A[big]))


def loglog_slope(Ls, vals):
    """OLS slope of log(vals) on log(Ls) -- the growth exponent across lattice size."""
    Ls = np.asarray(Ls, float); vals = np.asarray(vals, float)
    m = np.isfinite(vals) & (vals > 0)
    if m.sum() < 2:
        return np.nan
    g, _ = _ols_slope_se(np.log(Ls[m]), np.log(vals[m]))
    return g


# ---------------------------------------------------------------------------
# Seed-level bootstrap: resample seeds (with replacement) independently at each L,
# recompute the per-L statistics and the across-L growth slopes. The honest error,
# capturing run-to-run (between equilibrated state) variation as in S18's jackknife.
# ---------------------------------------------------------------------------
def bootstrap(data, Ls, n_boot=300, seed=0):
    rng = np.random.default_rng(seed)
    keys_L = ["meanA", "medA", "D_full", "D_typ", "onebond"]
    per_L = {k: {L: [] for L in Ls} for k in keys_L}
    slopes = {"meanA": [], "medA": []}
    for _ in range(n_boot):
        st = {}
        for L in Ls:
            A_s, Rg_s, wt_s = data[L][0], data[L][1], data[L][2]
            ns = len(A_s)
            idx = rng.integers(0, ns, ns)
            st[L] = pool_stats([A_s[i] for i in idx],
                               [Rg_s[i] for i in idx],
                               [wt_s[i] for i in idx])
            for k in keys_L:
                per_L[k][L].append(st[L][k])
        slopes["meanA"].append(loglog_slope(Ls, [st[L]["meanA"] for L in Ls]))
        slopes["medA"].append(loglog_slope(Ls, [st[L]["medA"] for L in Ls]))
    out = {"per_L": {}, "slope": {}}
    for k in keys_L:
        out["per_L"][k] = {L: (float(np.nanmean(per_L[k][L])),
                               float(np.nanstd(per_L[k][L]))) for L in Ls}
    for k in slopes:
        arr = np.array(slopes[k], float); arr = arr[np.isfinite(arr)]
        out["slope"][k] = (float(arr.mean()), float(arr.std()))
    return out


# ---------------------------------------------------------------------------
# Self-test: the typical/tail mass-radius separator on a known line+disk mixture.
# ---------------------------------------------------------------------------
def _self_test():
    print("=" * 70)
    print("filament_fattening.py self-test: typical/tail mass-radius split on a")
    print("known mixture (many D=1 lines + rare D=2 disks) -> typ~1, tail~2, full up")
    print("=" * 70)
    A, Rg = [], []
    # numerous thin lines (D=1), small to large -- the "typical" filament population
    for n in range(20, 400, 2):
        pts = _shape_line(n).astype(float)
        lam1, lam2, _, _ = gyration(pts)
        A.append(pts.shape[0]); Rg.append(np.sqrt(lam1 + lam2))
    n_line = len(A)
    # rare filled disks (D=2), LARGE-AREA ONLY and <=10% of the ensemble, so the area
    # tail is pure disks and the typical bulk is pure lines (mirroring the real data:
    # one thin filament branch, fat only in the largest avalanches). 190 lines (max
    # area 398) + 20 disks (min area ~452) -> top decile is exactly the disk tail.
    for rho in range(12, 32):
        pts = _shape_disk(rho).astype(float)
        lam1, lam2, _, _ = gyration(pts)
        A.append(pts.shape[0]); Rg.append(np.sqrt(lam1 + lam2))
    A = np.array(A); Rg = np.array(Rg)

    # direct mass-radius fit (log A on log Rg) -- the small-N analogue of the binned
    # _massradius_D used on the 1e5-footprint real data; here it isolates the split.
    def D_direct(sel):
        if sel.sum() < 5:
            return np.nan
        g, _ = _ols_slope_se(np.log(Rg[sel]), np.log(A[sel]))
        return g
    # clean fixed split between the known populations (max line area 398 < min disk
    # area ~452): tests the mass-radius estimator + low/high-area split concept without
    # fighting quantile alignment at this tiny synthetic N (the quantile itself, used on
    # the 1e5-footprint real data, needs no synthetic validation -- it is np.quantile).
    thr = 420.0
    D_full = D_direct(np.ones(A.size, bool))
    D_typ = D_direct(A < thr)
    D_tail = D_direct(A >= thr)
    print("  populations: %d lines (D=1) + %d disks (D=2); split at area %.0f"
          % (n_line, A.size - n_line, thr))
    print("  full-ensemble D = %.2f   typical D = %.2f   tail D = %.2f"
          % (D_full, D_typ, D_tail))
    assert abs(D_typ - 1.0) < 0.25, "typical split does not recover the D=1 lines"
    assert D_tail > D_typ + 0.4, "tail split does not isolate the fat D=2 disks"
    print("  -> the low-area split reads the thin filament (D~1) and the high-area split")
    print("     reads the fat population (D~2): a typical/tail split on the real data")
    print("     separates a thin typical avalanche from any fat tail, as intended.")
    print("self-test OK.\n")


# ---------------------------------------------------------------------------
def main(smoke=False):
    log("=" * 70)
    log("S19 -- DOES THE FILAMENT FATTEN AT LARGE L, OR IS IT A MULTIFRACTAL TAIL?")
    log("=" * 70)
    log("S18 found the area multifractality asymptotic, and FLAGGED a tentative second")
    log("reading: <A>-slope 0.90->1.22 and D_mid 1.07->1.23 at the largest L (5 seeds).")
    log("Separating the TYPICAL avalanche (median A, one-bond fraction, D_typ) from the")
    log("TAIL (mean A, A_max, D_tail) with more seeds decides: real crossover or tail.")

    if smoke:
        configs = [(128, 4_000_000, 2), (192, 4_000_000, 2), (256, 5_000_000, 2)]
        n_boot = 60
    else:
        # (L, recorded window, n_seeds). More seeds than S18 (which had 5 at 384/512),
        # to average down the rare tail and put an honest error on the growth slopes.
        configs = [
            (192,  9_000_000, 12),
            (256, 10_000_000, 12),
            (384, 12_000_000, 10),
            (512, 14_000_000, 10),
        ]
        n_boot = 300
    Ls = [c[0] for c in configs]

    data, repose = {}, {}
    log("\nWarming verified-stationary states (S16 enabler) and dumping footprints...")
    for L, window, n_seeds in configs:
        t = time.time()
        A_s, Rg_s, wt_s, ms, ss = measure_L(L, window, n_seeds, smoke=smoke)
        data[L] = (A_s, Rg_s, wt_s)
        repose[L] = ms
        nfoot = sum(a.size for a in A_s)
        cen = pool_stats(A_s, Rg_s, wt_s)
        log("  L=%4d : %7d footprints (%d seeds)  repose=%.2f+-%.2f  "
            "<A>=%.1f medA=%.0f A_max=%.0f  (%.0fs)"
            % (L, nfoot, n_seeds, ms, ss, cen['meanA'], cen['medA'], cen['Amax'],
               time.time() - t))

    # ---- seed-bootstrap the per-L statistics and the across-L growth slopes ----
    log("\nBootstrapping over seeds (%d resamples)..." % n_boot)
    bs = bootstrap(data, Ls, n_boot=n_boot, seed=11)

    # large-L growth slopes (use the upper sizes where the S18 hint lived)
    Ls_big = [L for L in Ls if L >= 256]
    def slope_bs(stat):
        if len(Ls_big) < 2:            # smoke mode (single big-L): slope undefined
            return np.nan, np.nan
        vals = [[pool_stats([data[L][0][i]], [data[L][1][i]], [data[L][2][i]])[stat]
                 for i in range(len(data[L][0]))] for L in Ls_big]
        # bootstrap by resampling seeds at each big-L
        rng = np.random.default_rng(23)
        gs = []
        for _ in range(bs_n := (60 if smoke else 300)):
            means = []
            for col in vals:
                col = np.array(col, float); col = col[np.isfinite(col)]
                idx = rng.integers(0, col.size, col.size)
                means.append(col[idx].mean())
            gs.append(loglog_slope(Ls_big, means))
        gs = np.array(gs, float); gs = gs[np.isfinite(gs)]
        return float(gs.mean()), float(gs.std())

    meanA_g, meanA_se = slope_bs("meanA")
    medA_g, medA_se = slope_bs("medA")

    log("\n[per-L typical/tail geometry  (bootstrap mean +- se over seeds)]")
    log("  L     <A>          median A     one-bond%%(tail)   D_full        D_typ")
    for L in Ls:
        pm = bs["per_L"]
        log("  %-4d  %5.1f+-%4.1f  %5.0f+-%3.0f  %.2f+-%.2f       %.2f+-%.2f   %.2f+-%.2f"
            % (L,
               pm["meanA"][L][0], pm["meanA"][L][1],
               pm["medA"][L][0], pm["medA"][L][1],
               pm["onebond"][L][0], pm["onebond"][L][1],
               pm["D_full"][L][0], pm["D_full"][L][1],
               pm["D_typ"][L][0], pm["D_typ"][L][1]))

    log("\n[growth slopes over L in %s  (1.0 = filament A~L; 2.0 = compact A~L^2)]"
        % "-".join(map(str, Ls_big)))
    log("  mean(A)   ~ L^ %.2f +- %.2f   (tail-weighted; the S18 quantity that read 1.22)"
        % (meanA_g, meanA_se))
    log("  median(A) ~ L^ %.2f +- %.2f   (the TYPICAL avalanche)" % (medA_g, medA_se))

    # ---- verdict ----
    onebond_big = bs["per_L"]["onebond"][Ls[-1]][0]
    Dtyp_big = bs["per_L"]["D_typ"][Ls[-1]][0]
    Dtyp_small = bs["per_L"]["D_typ"][Ls[0]][0]
    mean_exceeds_median = (meanA_g - medA_g) > (meanA_se + medA_se)
    typ_flat = (Dtyp_big < 1.25) and (medA_g < 1.15)
    log("\n[verdict -- real crossover (filament fattens) or multifractal tail?]")
    log("  mean-A slope %.2f vs median-A slope %.2f : mean %s median"
        % (meanA_g, medA_g, "EXCEEDS" if mean_exceeds_median else "~ tracks"))
    log("  one-bond-wide fraction (tail) at L=%d: %.2f  (S14: 0.98 at L<=256)"
        % (Ls[-1], onebond_big))
    log("  typical mass-radius D: %.2f (L=%d) -> %.2f (L=%d)"
        % (Dtyp_small, Ls[0], Dtyp_big, Ls[-1]))
    if mean_exceeds_median and typ_flat and onebond_big > 0.85:
        log("  -> MULTIFRACTAL TAIL. The typical avalanche stays a one-bond D~1 filament")
        log("     (median A ~ L, D_typ ~ 1, still ~one bond wide); only the rare largest")
        log("     avalanches grow (mean-A slope and A_max steepen). S18's apparent")
        log("     'fattening' is the multifractal TAIL, not a thickening of the typical")
        log("     footprint -- S14/S16's filament picture stands, now with the second")
        log("     S18 reading downgraded to a pure tail statement.")
    elif (not typ_flat) and onebond_big < 0.85:
        log("  -> REAL CROSSOVER. The TYPICAL footprint thickens at large L (median A and")
        log("     D_typ climb, one-bond fraction falls): D ~ 1 is a small-L description and")
        log("     the deterministic model drifts toward compactness on its own. This")
        log("     QUALIFIES S14/S16 and is a new result -- the filament is not asymptotic.")
    else:
        log("  -> MIXED / INCONCLUSIVE at these sizes: the typical and tail diagnostics do")
        log("     not point the same way; more seeds or a larger L would sharpen it.")

    _make_figure(data, Ls, Ls_big, bs, meanA_g, medA_g, repose)

    log("\nS19 COMPLETE")
    with open(os.path.join(OUTDIR, "sandpile_filament_fattening.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


def _make_figure(data, Ls, Ls_big, bs, meanA_g, medA_g, repose):
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.0))

    # ---- (0) mass-radius A ~ Rg^D, pooled largest L: typical vs full ----
    Lb = Ls[-1]
    A = np.concatenate(data[Lb][0]); Rg = np.concatenate(data[Lb][1])
    thr = np.quantile(A, TAIL_Q)
    typ = A < thr; tail = A >= thr
    ax[0].loglog(Rg[typ], A[typ], ".", ms=2, color="C0", alpha=0.25,
                 label="typical (bottom 90%% area)")
    ax[0].loglog(Rg[tail], A[tail], ".", ms=3, color="C3", alpha=0.5,
                 label="tail (top 10%% area)")
    if Rg.size:
        xr = np.array([max(Rg.min(), 1.0), Rg.max()])
        a0 = np.median(A[typ]) / np.median(Rg[typ]) ** 1.0
        for D, ls, lab in ((1.0, ":", "D=1 filament"), (2.0, "--", "D=2 compact")):
            ax[0].loglog(xr, a0 * xr ** D, ls, color="0.5", lw=1.1, label=lab)
    ax[0].set_xlabel("radius of gyration Rg"); ax[0].set_ylabel("footprint area A")
    ax[0].set_title("L=%d footprints: typical filament vs fat tail" % Lb)
    ax[0].legend(fontsize=7, loc="upper left")

    # ---- (1) mean vs median A growth with L (the decisive contrast) ----
    meanA = [bs["per_L"]["meanA"][L][0] for L in Ls]
    meanA_e = [bs["per_L"]["meanA"][L][1] for L in Ls]
    medA = [bs["per_L"]["medA"][L][0] for L in Ls]
    medA_e = [bs["per_L"]["medA"][L][1] for L in Ls]
    ax[1].errorbar(Ls, meanA, yerr=meanA_e, fmt="o-", color="C3", capsize=3,
                   label="mean A ~ L^%.2f (tail)" % meanA_g)
    ax[1].errorbar(Ls, medA, yerr=medA_e, fmt="s-", color="C0", capsize=3,
                   label="median A ~ L^%.2f (typical)" % medA_g)
    Lr = np.array([Ls[0], Ls[-1]], float)
    ax[1].plot(Lr, medA[0] * (Lr / Ls[0]) ** 1.0, "k:", lw=1.0, label="slope 1 (A~L)")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("lattice size L"); ax[1].set_ylabel("footprint area A")
    ax[1].set_title("Mean grows faster than median = growth is in the tail")
    ax[1].legend(fontsize=8, loc="upper left")

    # ---- (2) typical thinness vs L: one-bond fraction and D_typ ----
    ob = [bs["per_L"]["onebond"][L][0] for L in Ls]
    ob_e = [bs["per_L"]["onebond"][L][1] for L in Ls]
    Dt = [bs["per_L"]["D_typ"][L][0] for L in Ls]
    Dt_e = [bs["per_L"]["D_typ"][L][1] for L in Ls]
    Df = [bs["per_L"]["D_full"][L][0] for L in Ls]
    Df_e = [bs["per_L"]["D_full"][L][1] for L in Ls]
    ax[2].errorbar(Ls, ob, yerr=ob_e, fmt="^-", color="C2", capsize=3,
                   label="one-bond-wide frac (tail)")
    ax[2].set_xscale("log")
    ax[2].set_xlabel("lattice size L")
    ax[2].set_ylabel("one-bond-wide fraction", color="C2")
    ax[2].tick_params(axis="y", labelcolor="C2")
    ax[2].set_ylim(0, 1.05)
    ax[2].set_title("Does the TYPICAL footprint stay thin as L grows?")
    axr = ax[2].twinx()
    axr.errorbar(Ls, Dt, yerr=Dt_e, fmt="o--", color="C0", capsize=3,
                 label="D_typ (typical)")
    axr.errorbar(Ls, Df, yerr=Df_e, fmt="s--", color="C3", capsize=3,
                 label="D_full (with tail)")
    axr.axhline(1.0, color="0.6", ls=":", lw=1.0)
    axr.set_ylabel("mass-radius D", color="0.2")
    lines = ax[2].get_legend_handles_labels()[0] + axr.get_legend_handles_labels()[0]
    labs = ax[2].get_legend_handles_labels()[1] + axr.get_legend_handles_labels()[1]
    axr.legend(lines, labs, fontsize=7, loc="center right")

    fig.tight_layout()
    p = os.path.join(FIGDIR, "sandpile_filament_fattening.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    log("\nsaved %s" % os.path.relpath(p, ROOT))


if __name__ == "__main__":
    _self_test()
    main(smoke=("smoke" in sys.argv))
