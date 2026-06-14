"""
Phase 2 -- the Grand Challenge: 2-D slope-sandpile critical exponents and the
universality question.

Using the bond-slope 2-D model (sandpile2d.py: every x- and y-bond obeys the
identical 1-D pair rule, open boundaries on all four edges), this script:
  1. confirms the 2-D model self-organizes to a critical state with power-law
     avalanche statistics;
  2. extracts the 2-D exponents by finite-size scaling across L, for two
     observables -- avalanche ENERGY E (displaced mass) and avalanche SIZE S
     (number of bond topplings, the measure used for the canonical BTW sandpile);
  3. compares the 2-D exponents to the 1-D baseline (S3) to answer: do 1-D and
     2-D slope sandpiles share critical exponents?

The size exponent tau_S is the one to compare against the literature value for
the 2-D abelian (BTW) sandpile (tau_S ~ 1.2); that head-to-head comparison, with
BTW measured under this same pipeline, is done in btw_compare.py (Phase 2b).

Run from repo root:  python sandpile/fss2d.py
Writes figures/sandpile_fss2d.png and outputs/sandpile_fss2d.txt.
ASCII-only.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sandpile2d import run_sandpile2d, pyramid_ic
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


def measure_multi(disp, act):
    """Avalanche energy E (sum disp), size S (sum act), duration T from the two
    parallel per-iteration series. Avalanche = maximal run of disp > 0."""
    disp = np.asarray(disp); act = np.asarray(act)
    active = disp > 0.0
    if not active.any():
        z = np.array([])
        return z, z, z
    a = active.astype(np.int8)
    edges = np.diff(a)
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1) + 1
    if active[0]:
        starts = np.r_[0, starts]
    if active[-1]:
        ends = np.r_[ends, active.size]
    E = np.array([disp[s:e].sum() for s, e in zip(starts, ends)])
    S = np.array([act[s:e].sum() for s, e in zip(starts, ends)])
    T = (ends - starts).astype(float)
    return E, S, T


def cutoff_moment(x):
    x = np.asarray(x, dtype=float)
    return (x**2).mean() / x.mean()


def fit_D(Ls, xc):
    A = np.polyfit(np.log10(Ls), np.log10(xc), 1)
    return A[0]


def main():
    log("=" * 70)
    log("2-D SLOPE SANDPILE -- FINITE-SIZE SCALING / UNIVERSALITY")
    log("=" * 70)

    configs = [
        (32,  600_000, 200_000),
        (48,  900_000, 300_000),
        (64, 1_200_000, 400_000),
        (96, 2_000_000, 600_000),
        (128, 3_000_000, 800_000),
    ]

    data = {}
    log("\nRunning 2-D lattices...")
    for L, n_iter, warm in configs:
        res = run_sandpile2d(L=L, eps=0.1, Zc=5.0, n_iter=n_iter, seed=3,
                             S0=pyramid_ic(L, 0.90 * 5.0))
        E, S, T = measure_multi(res['disp'][warm:], res['act'][warm:])
        mslope = np.abs(np.diff(res['S'], axis=0)).mean()
        data[L] = dict(E=E, S=S, T=T)
        log("  L=%4d : %6d avalanches  mean-slope=%.2f  (E_max=%.3g, S_max=%.3g, T_max=%d)"
            % (L, E.size, mslope, E.max() if E.size else -1,
               S.max() if S.size else -1, int(T.max()) if T.size else -1))

    Ls = np.array([c[0] for c in configs], dtype=float)
    Lbig = Ls[-1]

    def exponents(key, label, lo_mult, hi_mult):
        cb, db = logbin_pdf(data[Lbig][key])
        tau = -powerlaw_slope(cb, db, lo=cb.min() * lo_mult, hi=cb.max() * hi_mult)
        xc = np.array([cutoff_moment(data[L][key]) for L in Ls])
        D = fit_D(Ls, xc)
        log("  tau_%s = %.3f    D_%s = %.3f  (cutoff ~ L^D)" % (label, tau, label, D))
        return tau, D

    log("\n[2-D exponents]")
    tauE, DE = exponents('E', 'E', 5, 0.15)
    tauS, DS = exponents('S', 'S', 5, 0.15)
    tauT, DT = exponents('T', 'T', 3, 0.15)

    log("\n[1-D vs 2-D comparison]   (1-D from S3: tau_E~1.03, D_E~2.0, D_T~1.0)")
    log("  observable     tau(1-D)   tau(2-D)")
    log("  energy E         1.03       %.2f" % tauE)
    log("  duration T       0.60       %.2f" % tauT)
    log("  -> if tau differs, 1-D and 2-D are NOT in the same universality class")
    log("  size S (2-D only, for BTW comparison): tau_S = %.2f" % tauS)

    # figure: raw + collapsed energy and size PDFs
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    cmap = plt.cm.plasma(np.linspace(0, 0.85, len(Ls)))

    for c, L in zip(cmap, Ls):
        ce, de = logbin_pdf(data[L]['E'])
        ax[0, 0].loglog(ce, de, "-", color=c, label="L=%d" % int(L))
        ax[0, 1].loglog(ce / L**DE, de * ce**tauE, "-", color=c, label="L=%d" % int(L))
        cs, ds = logbin_pdf(data[L]['S'])
        ax[1, 0].loglog(cs, ds, "-", color=c, label="L=%d" % int(L))
        ax[1, 1].loglog(cs / L**DS, ds * cs**tauS, "-", color=c, label="L=%d" % int(L))
    ax[0, 0].set_title("2-D energy PDFs (raw)"); ax[0, 0].set_xlabel("E"); ax[0, 0].set_ylabel("PDF(E)")
    ax[0, 1].set_title("2-D energy collapse (tau_E=%.2f, D_E=%.2f)" % (tauE, DE))
    ax[0, 1].set_xlabel("E / L^%.2f" % DE); ax[0, 1].set_ylabel("E^%.2f PDF(E)" % tauE)
    ax[1, 0].set_title("2-D size PDFs (raw) -- topplings, BTW-comparable")
    ax[1, 0].set_xlabel("S"); ax[1, 0].set_ylabel("PDF(S)")
    ax[1, 1].set_title("2-D size collapse (tau_S=%.2f, D_S=%.2f)" % (tauS, DS))
    ax[1, 1].set_xlabel("S / L^%.2f" % DS); ax[1, 1].set_ylabel("S^%.2f PDF(S)" % tauS)
    for a in ax.flat:
        a.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "sandpile_fss2d.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    log("\nsaved %s" % os.path.relpath(p, ROOT))

    log("\n2-D FSS COMPLETE")
    with open(os.path.join(OUTDIR, "sandpile_fss2d.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
