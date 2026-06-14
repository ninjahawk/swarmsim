"""
Phase 1 -- rigorous critical-exponent extraction for the 1-D slope sandpile
via finite-size scaling (FSS).

Charbonneau (Ch. 5) shows the avalanche PDFs are power laws whose slope does not
change with lattice size N; the distribution simply "extends farther right" as N
grows. That is the qualitative SOC signature. Here we make it quantitative.

The standard SOC ansatz for an avalanche-size observable x (we use energy E and
duration T) is
        P(x) = x^{-tau_x} * g(x / x_c),    x_c ~ N^{D_x},
i.e. a power law with exponent tau_x truncated by a cutoff x_c that scales as a
power D_x of the system size. Two things follow that we can test directly:

  1. tau_x is read from the slope of the power-law region (size-independent).
  2. D_x is read from how the cutoff x_c grows with N. We estimate the cutoff
     robustly from the moment ratio  x_c ~ <x^2> / <x>  (for 1 < tau < 2 this
     ratio scales linearly with the true cutoff, and unlike a raw tail estimate
     it does not need many rare large events).
  3. With tau_x and D_x in hand, the rescaled curves  x^{tau_x} P(x)  versus
     x / N^{D_x}  must COLLAPSE onto one universal function g -- the decisive
     test that a single (tau, D) pair describes every lattice size.

We also report the scaling-relation check D_x (2 - tau_x): for a conservative,
boundary-dissipative, slowly-driven pile the mean avalanche size is expected to
grow linearly with N (each added grain is eventually carried ~N/2 nodes to the
open edge), which predicts D_E (2 - tau_E) = 1.

Run from the repository root:  python sandpile/fss1d.py
Writes figures/sandpile_fss.png and outputs/sandpile_fss.txt.
ASCII-only output.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sandpile1d import run_sandpile, measure_avalanches, triangle_ic
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


def collect(N, n_iter, warm, eps=0.1, Zc=5.0, seed=7):
    """Run one lattice and return its avalanche (E, P, T) arrays past warmup."""
    S0 = triangle_ic(N, 0.90 * Zc)
    res = run_sandpile(N=N, eps=eps, Zc=Zc, n_iter=n_iter, seed=seed, S0=S0)
    return measure_avalanches(res['disp'][warm:])


def cutoff_moment(x):
    """Robust cutoff estimator x_c ~ <x^2>/<x>."""
    x = np.asarray(x, dtype=float)
    return (x**2).mean() / x.mean()


def fit_D(Ns, xc):
    """Power-law fit x_c ~ N^D; return (D, intercept)."""
    A = np.polyfit(np.log10(Ns), np.log10(xc), 1)
    return A[0], A[1]


def main():
    log("=" * 70)
    log("1-D SANDPILE -- FINITE-SIZE SCALING / CRITICAL EXPONENTS")
    log("=" * 70)

    # Lattice sizes and iteration budgets (more iters for larger N to keep the
    # avalanche count comparable; per-iteration cost grows ~linearly with N).
    configs = [
        (64,   2_000_000,  600_000),
        (128,  3_000_000,  800_000),
        (256,  4_000_000, 1_000_000),
        (512,  6_000_000, 1_500_000),
        (1024, 9_000_000, 2_000_000),
    ]

    data = {}
    log("\nRunning lattices...")
    for N, n_iter, warm in configs:
        E, P, T = collect(N, n_iter, warm)
        data[N] = dict(E=E, P=P, T=T)
        log("  N=%5d : %6d avalanches  (E_max=%.3g, T_max=%d)"
            % (N, E.size, E.max() if E.size else -1, int(T.max()) if T.size else -1))

    Ns = np.array([c[0] for c in configs], dtype=float)

    # ----- energy exponents -----
    log("\n[E] avalanche energy")
    # tau from the largest lattice (widest scaling window).
    cE_big, dE_big = logbin_pdf(data[Ns[-1]]['E'])
    tau_E = -powerlaw_slope(cE_big, dE_big, lo=cE_big.min() * 5, hi=cE_big.max() * 0.1)
    xcE = np.array([cutoff_moment(data[N]['E']) for N in Ns])
    D_E, _ = fit_D(Ns, xcE)
    log("  tau_E = %.3f   (book PDF slope ~ -1.09)" % tau_E)
    log("  D_E   = %.3f   (cutoff E_c ~ N^D_E; geometric expectation ~2)" % D_E)
    log("  scaling check D_E*(2 - tau_E) = %.3f   (conservation predicts ~1)"
        % (D_E * (2 - tau_E)))

    # ----- duration exponents -----
    log("\n[T] avalanche duration")
    cT_big, dT_big = logbin_pdf(data[Ns[-1]]['T'])
    tau_T = -powerlaw_slope(cT_big, dT_big, lo=cT_big.min() * 3, hi=cT_big.max() * 0.1)
    xcT = np.array([cutoff_moment(data[N]['T']) for N in Ns])
    D_T, _ = fit_D(Ns, xcT)
    log("  tau_T = %.3f" % tau_T)
    log("  D_T   = %.3f   (duration cutoff T_c ~ N^D_T; geometric expectation ~1)" % D_T)

    # ----- figure: raw + collapsed PDFs, cutoff scaling -----
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    cmap = plt.cm.viridis(np.linspace(0, 0.85, len(Ns)))

    # (a) raw energy PDFs
    for c, N in zip(cmap, Ns):
        ce, de = logbin_pdf(data[N]['E'])
        ax[0, 0].loglog(ce, de, "-", color=c, label="N=%d" % int(N))
    ax[0, 0].set_xlabel("avalanche energy E"); ax[0, 0].set_ylabel("PDF(E)")
    ax[0, 0].set_title("Energy PDFs (raw): size-independent slope, growing cutoff")
    ax[0, 0].legend(fontsize=8)

    # (b) collapsed energy PDFs: E^tau P(E) vs E/N^D
    for c, N in zip(cmap, Ns):
        ce, de = logbin_pdf(data[N]['E'])
        ax[0, 1].loglog(ce / N**D_E, de * ce**tau_E, "-", color=c, label="N=%d" % int(N))
    ax[0, 1].set_xlabel("E / N^%.2f" % D_E)
    ax[0, 1].set_ylabel("E^%.2f  PDF(E)" % tau_E)
    ax[0, 1].set_title("Energy PDF data collapse (tau_E=%.2f, D_E=%.2f)" % (tau_E, D_E))
    ax[0, 1].legend(fontsize=8)

    # (c) cutoff scaling E_c, T_c vs N
    ax[1, 0].loglog(Ns, xcE, "o-", color="C0", label="E_c ~ N^%.2f" % D_E)
    ax[1, 0].loglog(Ns, xcT, "s-", color="C3", label="T_c ~ N^%.2f" % D_T)
    ax[1, 0].set_xlabel("lattice size N"); ax[1, 0].set_ylabel("cutoff (moment ratio)")
    ax[1, 0].set_title("Cutoff scaling with system size")
    ax[1, 0].legend(fontsize=9)

    # (d) collapsed duration PDFs
    for c, N in zip(cmap, Ns):
        ct, dt = logbin_pdf(data[N]['T'])
        ax[1, 1].loglog(ct / N**D_T, dt * ct**tau_T, "-", color=c, label="N=%d" % int(N))
    ax[1, 1].set_xlabel("T / N^%.2f" % D_T)
    ax[1, 1].set_ylabel("T^%.2f  PDF(T)" % tau_T)
    ax[1, 1].set_title("Duration PDF data collapse (tau_T=%.2f, D_T=%.2f)" % (tau_T, D_T))
    ax[1, 1].legend(fontsize=8)

    fig.tight_layout()
    p = os.path.join(FIGDIR, "sandpile_fss.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    log("\nsaved %s" % os.path.relpath(p, ROOT))

    log("\nFSS COMPLETE")
    with open(os.path.join(OUTDIR, "sandpile_fss.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
