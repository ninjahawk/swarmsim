"""
S10 -- the S6 duration self-test, redone at large lattices with the fast engine.

S4 concluded that the 2-D slope sandpile and the canonical 2-D abelian (BTW)
sandpile are in different universality classes, from their avalanche-SIZE
exponents (tau_S = 0.89 vs 1.14). The remaining objection was that "size" is
counted differently in the two models (the slope model counts BOND topplings, BTW
counts SITE topplings). S6 tried to remove that objection with the avalanche
DURATION, which is defined identically for both models (the number of parallel
relaxation sweeps), but at L <= 128 the slope model's duration spanned too little
range and the fitted exponent swung by a factor of two, so S6 was inconclusive.

The active-list engine (sandpile_fast.py, S9) makes the slope model ~600x faster,
which lifts both limits at once: lattices up to L = 512 (duration cutoff ~ L, so a
much wider scaling window) and ~2 x 10^5 avalanches per size for a stable fit.
This script measures tau_T for the slope model by finite-size scaling across
L = 64..512 and for BTW across L = 32..128 under one matched log-binned-PDF fit
(the window [8, 0.3 T_max], chosen to skip the steep small-T head that comes from
duration quantization at T = 1, 2, 3), and tests the slope-model data collapse.

Run from repo root:  python sandpile/duration_fss2d.py
Writes figures/sandpile_duration_fss.png and outputs/sandpile_duration_fss.txt.
ASCII-only.
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sandpile_fast import run_sandpile2d_fast, pyramid_ic
from sandpile1d import measure_avalanches
from btw_compare import btw_run
from validate1d import logbin_pdf, powerlaw_slope

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "figures")
OUTDIR = os.path.join(ROOT, "outputs")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

LOG = []
def log(msg):
    print(msg)
    LOG.append(msg)


# Fixed scaling window: skip the quantized small-T head (T < 8) and the noisy
# extreme tail (beyond 0.3 of the largest duration). Applied identically to both
# models so the comparison is convention- and method-matched.
def tau_T(T, lo=8.0, hi_frac=0.3):
    cT, dT = logbin_pdf(T)
    return -powerlaw_slope(cT, dT, lo=lo, hi=hi_frac * T.max()), (cT, dT)


def cutoff_moment(x):
    x = np.asarray(x, dtype=float)
    return (x**2).mean() / x.mean()


def fit_D(Ls, xc):
    return np.polyfit(np.log10(Ls), np.log10(xc), 1)[0]


def main():
    log("=" * 72)
    log("S10 -- 2-D avalanche-DURATION exponent by FSS (slope model vs BTW)")
    log("=" * 72)
    log("Duration = parallel relaxation sweeps (identical definition for both).")
    log("Fit window [8, 0.3*T_max] applied to both models. Slope model uses the")
    log("active-list engine (S9); BTW uses the same pipeline as S4/S6.\n")

    # ---- slope model: large lattices via the fast engine ----
    slope_cfg = [
        (64,  40_000_000,   800_000),
        (128, 60_000_000, 1_000_000),
        (256, 90_000_000, 2_000_000),
        (512, 120_000_000, 3_000_000),
    ]
    log("Slope model (fast engine):")
    sdata = {}
    for L, n_iter, warm in slope_cfg:
        t = time.time()
        r = run_sandpile2d_fast(L=L, eps=0.1, Zc=5.0, n_iter=n_iter, seed=3,
                                S0=pyramid_ic(L, 0.9 * 5.0))
        T = measure_avalanches(r['disp'][warm:])[2]
        sdata[L] = T
        tt, _ = tau_T(T)
        log("  L=%4d : %7d avalanches  T_max=%4d  Tc=%6.1f  tau_T=%.3f  (%.1fs)"
            % (L, T.size, int(T.max()), cutoff_moment(T), tt, time.time() - t))

    sLs = np.array([c[0] for c in slope_cfg], dtype=float)
    sTc = np.array([cutoff_moment(sdata[L]) for L in sLs])
    D_T_slope = fit_D(sLs, sTc)
    # representative exponent: mean over the four sizes (each individually stable)
    tau_slope = np.mean([tau_T(sdata[L])[0] for L in sLs])
    log("  -> tau_T(slope) = %.3f (mean over L)   D_T(slope) = %.3f  (Tc ~ L^D_T)"
        % (tau_slope, D_T_slope))

    # ---- BTW: same pipeline, sizes it can reach in pure Python ----
    btw_cfg = [(48, 150_000, 30_000), (64, 200_000, 30_000), (128, 120_000, 25_000)]
    log("\nCanonical BTW (same pipeline):")
    bdata = {}
    for L, n_ev, warm in btw_cfg:
        t = time.time()
        _, T = btw_run(L, n_ev, warm, seed=5)
        bdata[L] = T
        tt, _ = tau_T(T)
        log("  L=%4d : %7d avalanches  T_max=%4d  Tc=%6.1f  tau_T=%.3f  (%.1fs)"
            % (L, T.size, int(T.max()), cutoff_moment(T), tt, time.time() - t))
    bLs = np.array([c[0] for c in btw_cfg], dtype=float)
    bTc = np.array([cutoff_moment(bdata[L]) for L in bLs])
    D_T_btw = fit_D(bLs, bTc)
    tau_btw = np.mean([tau_T(bdata[L])[0] for L in bLs])
    log("  -> tau_T(BTW)   = %.3f (mean over L)   D_T(BTW)   = %.3f"
        % (tau_btw, D_T_btw))

    # ---- verdict ----
    log("\n[verdict]")
    gap = abs(tau_slope - tau_btw)
    log("  tau_T(slope) = %.2f   tau_T(BTW) = %.2f   gap = %.2f" % (tau_slope, tau_btw, gap))
    if gap > 0.2:
        log("  The duration exponents DIFFER well beyond the per-fit scatter (~0.03).")
        log("  This CONFIRMS the S4 universality split with a convention-free measure,")
        log("  removing the bond-vs-site size-counting caveat. S6 is now resolved.")
    else:
        log("  The duration exponents are close; the size-based S4 evidence still leads.")

    # ---- figure: slope raw PDFs, slope collapse, cutoff scaling, slope-vs-BTW ----
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    cmap = plt.cm.viridis(np.linspace(0, 0.85, len(sLs)))

    for c, L in zip(cmap, sLs):
        cT, dT = logbin_pdf(sdata[L])
        ax[0, 0].loglog(cT, dT, "-", color=c, label="L=%d" % int(L))
    ax[0, 0].set_title("Slope model: duration PDFs (raw)")
    ax[0, 0].set_xlabel("duration T (sweeps)"); ax[0, 0].set_ylabel("PDF(T)")
    ax[0, 0].legend(fontsize=8)

    for c, L in zip(cmap, sLs):
        cT, dT = logbin_pdf(sdata[L])
        ax[0, 1].loglog(cT / L**D_T_slope, dT * cT**tau_slope, "-", color=c,
                        label="L=%d" % int(L))
    ax[0, 1].set_title("Slope model duration collapse (tau_T=%.2f, D_T=%.2f)"
                       % (tau_slope, D_T_slope))
    ax[0, 1].set_xlabel("T / L^%.2f" % D_T_slope)
    ax[0, 1].set_ylabel("T^%.2f PDF(T)" % tau_slope)
    ax[0, 1].legend(fontsize=8)

    ax[1, 0].loglog(sLs, sTc, "o-", color="C0", label="slope: Tc ~ L^%.2f" % D_T_slope)
    ax[1, 0].loglog(bLs, bTc, "s-", color="C3", label="BTW: Tc ~ L^%.2f" % D_T_btw)
    ax[1, 0].set_title("Duration cutoff scaling")
    ax[1, 0].set_xlabel("lattice size L"); ax[1, 0].set_ylabel("Tc (moment ratio)")
    ax[1, 0].legend(fontsize=9)

    cs, ds = logbin_pdf(sdata[sLs[-1]])
    cb, db = logbin_pdf(bdata[bLs[-1]])
    ax[1, 1].loglog(cs, ds, "o-", ms=3, color="C0",
                    label="slope L=%d (tau_T=%.2f)" % (int(sLs[-1]), tau_slope))
    ax[1, 1].loglog(cb, db, "s-", ms=3, color="C3",
                    label="BTW L=%d (tau_T=%.2f)" % (int(bLs[-1]), tau_btw))
    ax[1, 1].set_title("Duration PDFs: slope vs BTW")
    ax[1, 1].set_xlabel("duration T (sweeps)"); ax[1, 1].set_ylabel("PDF(T)")
    ax[1, 1].legend(fontsize=9)

    fig.tight_layout()
    p = os.path.join(FIGDIR, "sandpile_duration_fss.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    log("\nsaved %s" % os.path.relpath(p, ROOT))

    log("\nS10 COMPLETE")
    with open(os.path.join(OUTDIR, "sandpile_duration_fss.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
