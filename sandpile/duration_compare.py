"""
S6 -- universality cross-check via the avalanche-DURATION exponent.

S4 concluded that the 2-D slope sandpile and the canonical 2-D abelian (BTW)
sandpile are in different universality classes, on the basis of their avalanche-
SIZE exponents (tau_S = 0.89 vs 1.14). The one soft spot in that comparison is
that "size" is counted slightly differently in the two models: the slope model
counts BOND topplings, BTW counts SITE topplings. That O(1) difference cannot
change a power-law exponent, but a cleaner discriminator avoids the objection
entirely: avalanche DURATION is defined identically for both models -- the number
of parallel relaxation steps (sweeps) the avalanche takes. If tau_T also differs,
the universality conclusion no longer depends on any counting convention.

This script measures tau_T for both models at matched lattice sizes under the
same log-binned-PDF fit, and reports them side by side.

Run from repo root:  python sandpile/duration_compare.py
Writes figures/sandpile_duration_compare.png and outputs/sandpile_duration_compare.txt.
ASCII-only.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sandpile1d import measure_avalanches
from sandpile2d import run_sandpile2d, pyramid_ic
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


def slope_durations(L, n_iter, warm):
    """Avalanche durations for the 2-D slope model at size L."""
    res = run_sandpile2d(L=L, eps=0.1, Zc=5.0, n_iter=n_iter, seed=3,
                         S0=pyramid_ic(L, 0.90 * 5.0))
    E, P, T = measure_avalanches(res['disp'][warm:])
    return T


def tauT(T):
    cT, dT = logbin_pdf(T)
    return -powerlaw_slope(cT, dT, lo=cT.min() * 2, hi=cT.max() * 0.2), (cT, dT)


def main():
    log("=" * 70)
    log("UNIVERSALITY CROSS-CHECK: avalanche-DURATION exponent (slope vs BTW)")
    log("=" * 70)
    log("Duration = number of parallel relaxation sweeps; identical definition")
    log("for both models, so this comparison is free of any counting convention.\n")

    Ls = [64, 128]
    slope_cfg = {64: (1_200_000, 400_000), 128: (3_000_000, 800_000)}
    btw_cfg = {64: (100_000, 30_000), 128: (60_000, 25_000)}

    results = {}
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for col, (model, getter) in enumerate([("slope", "slope"), ("BTW", "btw")]):
        for L in Ls:
            if model == "slope":
                T = slope_durations(L, *slope_cfg[L])
            else:
                S, T = btw_run(L, *btw_cfg[L], seed=5)
            tt, (cT, dT) = tauT(T)
            results[(model, L)] = tt
            ax[col].loglog(cT, dT, "o-", ms=3, label="L=%d (tau_T=%.2f)" % (L, tt))
            log("  %-5s L=%-4d : %6d avalanches, tau_T = %.3f"
                % (model, L, T.size, tt))
        ax[col].set_title("%s model: duration PDF" % model)
        ax[col].set_xlabel("duration T (sweeps)"); ax[col].set_ylabel("PDF(T)")
        ax[col].legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "sandpile_duration_compare.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)

    log("\n[verdict]")
    sl = np.mean([results[("slope", L)] for L in Ls])
    bt = np.mean([results[("BTW", L)] for L in Ls])
    log("  mean tau_T(slope) = %.2f   mean tau_T(BTW) = %.2f" % (sl, bt))
    log("  duration exponents %s -- %s the size-based conclusion (S4)"
        % ("DIFFER" if abs(sl - bt) > 0.15 else "agree",
           "confirms" if abs(sl - bt) > 0.15 else "complicates"))
    log("\nsaved %s" % os.path.relpath(p, ROOT))
    with open(os.path.join(OUTDIR, "sandpile_duration_compare.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
