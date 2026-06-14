"""
S8 -- Exercise 2: boundary "falloff" avalanches vs bulk toppling avalanches.

The chapter (Exercise 2) asks: track the mass that actually falls off the open
right edge as a time series distinct from the toppled-mass series, compute its
avalanche parameters, ask whether falloff avalanches are scale-invariant, and ask
how falloff energy correlates with the bulk avalanching energy.

A subtlety the exercise glosses over: applying the section-5.4 run-detector
directly to the raw falloff series is not meaningful, because the boundary drains
in many short bursts WITHIN a single bulk avalanche (here ~60 falloff bursts per
boundary-reaching avalanche), so it fragments into thousands of one-iteration
events. The physically meaningful unit is the falloff PER BULK AVALANCHE: for each
avalanche (a run of toppled-mass disp > 0), sum the mass that left the boundary
during it. That is what we use.

Findings to report:
  - what fraction of bulk avalanches actually reach the boundary (the rest are
    interior avalanches that move sand downslope but never evacuate any);
  - whether the per-avalanche falloff energy is itself a power law with an
    N-independent slope (scale invariant);
  - how strongly falloff energy correlates with bulk (toppled) energy.

Run from repo root:  python sandpile/falloff.py
Writes figures/sandpile_falloff.png and outputs/sandpile_falloff.txt.
ASCII-only.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sandpile1d import run_sandpile, triangle_ic
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


def per_avalanche(disp, falloff):
    """For each bulk avalanche (run of disp>0), return (bulk_E, falloff_E)."""
    disp = np.asarray(disp); falloff = np.asarray(falloff)
    active = disp > 0.0
    if not active.any():
        return np.array([]), np.array([])
    a = active.astype(np.int8)
    edges = np.diff(a)
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1) + 1
    if active[0]:
        starts = np.r_[0, starts]
    if active[-1]:
        ends = np.r_[ends, active.size]
    bulkE = np.array([disp[s:e].sum() for s, e in zip(starts, ends)])
    foffE = np.array([falloff[s:e].sum() for s, e in zip(starts, ends)])
    return bulkE, foffE


def main():
    log("=" * 70)
    log("FALLOFF (BOUNDARY) AVALANCHES vs BULK AVALANCHES  (Exercise 2, S8)")
    log("=" * 70)

    runs = [(300, 3_000_000, 600_000), (1000, 5_000_000, 800_000)]
    Zc, eps = 5.0, 0.1
    store = {}
    for N, n_iter, warm in runs:
        res = run_sandpile(N=N, eps=eps, Zc=Zc, n_iter=n_iter, seed=1,
                           S0=triangle_ic(N, 0.95 * Zc))
        bE, fE = per_avalanche(res['disp'][warm:], res['falloff'][warm:])
        reach = fE > 0
        store[N] = (bE, fE, reach)
        log("\n  N=%d : %d bulk avalanches, %d reach boundary (%.0f%%)"
            % (N, bE.size, reach.sum(), 100 * reach.mean()))
        # PDF slopes
        cb, db = logbin_pdf(bE)
        sb = powerlaw_slope(cb, db, lo=cb.min() * 5, hi=cb.max() * 0.2)
        cf, df = logbin_pdf(fE[reach])
        sf = powerlaw_slope(cf, df, lo=cf.min() * 5, hi=cf.max() * 0.2)
        # correlation on boundary-reaching avalanches (log-log)
        lo = np.log10(bE[reach]); lf = np.log10(fE[reach])
        r = np.corrcoef(lo, lf)[0, 1]
        log("    bulk-E PDF slope %.2f ; falloff-E PDF slope %.2f" % (sb, sf))
        log("    log-log corr(bulk E, falloff E) on boundary avalanches = %.2f" % r)
        log("    mean falloff/bulk energy ratio (reaching) = %.3f"
            % (fE[reach] / bE[reach]).mean())

    # figure: falloff PDF scale-invariance (two N) + bulk-vs-falloff scatter
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for N, c in zip([r[0] for r in runs], ["C3", "k"]):
        bE, fE, reach = store[N]
        cf, df = logbin_pdf(fE[reach])
        ax[0].loglog(cf, df, "-", color=c, label="N=%d" % N)
    ax[0].set_xlabel("falloff energy per avalanche")
    ax[0].set_ylabel("PDF")
    ax[0].set_title("Falloff-energy PDF: power law, N-independent slope?")
    ax[0].legend()

    Nbig = runs[-1][0]
    bE, fE, reach = store[Nbig]
    ax[1].loglog(bE[reach], fE[reach], ".", ms=2, alpha=0.3, color="C0")
    ax[1].set_xlabel("bulk (toppled) energy E")
    ax[1].set_ylabel("falloff energy")
    ax[1].set_title("Falloff vs bulk energy, N=%d (boundary-reaching)" % Nbig)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "sandpile_falloff.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    log("\nsaved %s" % os.path.relpath(p, ROOT))
    with open(os.path.join(OUTDIR, "sandpile_falloff.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
