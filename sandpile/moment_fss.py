"""
S11 -- validate the moment-scaling method on canonical BTW (a known result).

Before reading multifractality off OUR slope model (S12), reproduce it where the
answer is already established: the 2-D abelian Bak-Tang-Wiesenfeld sandpile. The
literature result (De Menech, Stella, Tebaldi, PRE 58, R2677 (1998); Tebaldi,
De Menech, Stella, PRL 83, 3952 (1999)) is that BTW does NOT obey simple
finite-size scaling -- its avalanche TOPPLING-NUMBER moments give a NONLINEAR
sigma(q), i.e. a local slope D(q)=d sigma/dq that DRIFTS with q (multifractal /
anomalous scaling) -- while the avalanche AREA (number of distinct toppled sites)
is much closer to a single-fractal, FSS-obeying observable.

If our machinery reports a flat D(q) for BTW toppling number, the method is
broken; it must reproduce the drift. This is the same "check a known result"
discipline as S1, applied to the analysis rather than the model.

Run from repo root:  python sandpile/moment_fss.py
Writes figures/sandpile_moments_btw.png and outputs/sandpile_moments_btw.txt.
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
from btw_compare import btw_run
from moments import avalanche_moments, sigma_of_q, local_slope, bootstrap_sigma

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "figures")
OUTDIR = os.path.join(ROOT, "outputs")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

LOG = []
def log(msg):
    print(msg)
    LOG.append(msg)


def main():
    log("=" * 70)
    log("S11 -- MOMENT-SCALING METHOD VALIDATED ON CANONICAL BTW")
    log("=" * 70)

    # BTW is pure-Python (O(L^2) per sweep); these sizes/counts keep it to a few
    # minutes while giving >~5e4 avalanches per size for stable moments.
    configs = [
        (32, 200_000, 40_000),
        (48, 200_000, 40_000),
        (64, 150_000, 40_000),
        (96, 120_000, 30_000),
        (128, 100_000, 30_000),
    ]
    Ls = [c[0] for c in configs]
    q_grid = np.arange(0.5, 5.01, 0.25)

    S_by_L, A_by_L = {}, {}
    log("\nRunning BTW lattices (size S = topplings, area A = distinct sites)...")
    for L, n_ev, warm in configs:
        t = time.time()
        S, T, A = btw_run(L, n_ev, warm, seed=5, track_area=True)
        S_by_L[L] = S
        A_by_L[L] = A
        log("  L=%4d : %6d avalanches  <S>=%.1f S_max=%.3g  <A>=%.1f A_max=%.3g  (%.1fs)"
            % (L, S.size, S.mean(), S.max(), A.mean(), A.max(), time.time() - t))

    def analyse(by_L, label):
        mom = {L: avalanche_moments(by_L[L], q_grid) for L in Ls}
        sigma, sigma_se = sigma_of_q(mom, Ls, q_grid)
        bs = bootstrap_sigma(by_L, Ls, q_grid, n_boot=200, seed=7)
        Dq = bs["Dq_mean"]
        Dq_sd = bs["Dq_std"]
        # FSS null = a constant D(q). Measure the drift across a resolved q-window
        # (skip the noisy finite-difference edges) and compare to bootstrap noise.
        sel = (q_grid >= 1.0) & (q_grid <= 4.0)
        drift = Dq[sel].max() - Dq[sel].min()
        typ_sd = np.median(Dq_sd[sel])
        log("\n  [%s]  sigma(q) and local slope D(q)=d sigma/dq" % label)
        log("    q     sigma(q)   D(q)+-boot")
        for q, s, d, ds in zip(q_grid, sigma, Dq, Dq_sd):
            if abs((q * 2) % 1) < 1e-9 and (q * 2) % 2 < 1e-9:  # integer q only, terse
                log("    %.1f   %8.3f   %.3f +- %.3f" % (q, s, d, ds))
        log("    D(q) drift over q in [1,4] = %.3f   (bootstrap noise ~ %.3f)"
            % (drift, typ_sd))
        # Honest 3-tier read, not a single threshold (the S6 auto-verdict trap:
        # area's bootstrap noise is tiny, so a hair of finite-size creep trips a
        # naive 4*noise cut even though area is essentially flat). A "strong"
        # verdict needs the drift to be both several times the noise AND a sizable
        # absolute fraction of D itself.
        rel = drift / np.median(Dq[sel])
        if drift > 4 * typ_sd and rel > 0.05:
            verdict = "MULTIFRACTAL (D(q) drifts strongly with q)"
        elif drift > 4 * typ_sd:
            verdict = "near-FSS (a weak, finite-size-like D(q) creep; D ~ %.2f)" % np.median(Dq[sel])
        else:
            verdict = "FLAT D(q) -> simple FSS (D ~ %.2f)" % np.median(Dq[sel])
        log("    -> %s" % verdict)
        return sigma, Dq, Dq_sd, drift, typ_sd

    sigS, DqS, DqS_sd, driftS, sdS = analyse(S_by_L, "toppling number S")
    sigA, DqA, DqA_sd, driftA, sdA = analyse(A_by_L, "area A")

    log("\n[expected, from the literature]")
    log("  toppling number S : multifractal, D(q) drifts upward with q")
    log("  area A            : closer to simple FSS, D(q) much flatter")
    log("  measured drift  S=%.3f  A=%.3f   (S should clearly exceed A)"
        % (driftS, driftA))

    # ---- figure: sigma(q) and D(q) for S vs A ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    ax[0].plot(q_grid, sigS, "o-", color="C3", label="toppling number S")
    ax[0].plot(q_grid, sigA, "s-", color="C0", label="area A")
    ax[0].set_xlabel("moment order q")
    ax[0].set_ylabel("sigma(q)   ( <x^q> ~ L^sigma(q) )")
    ax[0].set_title("BTW moment exponents")
    ax[0].legend()

    ax[1].errorbar(q_grid, DqS, yerr=DqS_sd, fmt="o-", color="C3", capsize=2,
                   label="S: D(q) drift=%.2f" % driftS)
    ax[1].errorbar(q_grid, DqA, yerr=DqA_sd, fmt="s-", color="C0", capsize=2,
                   label="A: D(q) drift=%.2f" % driftA)
    # flat-D reference (FSS): horizontal line at each observable's mid-q value
    ax[1].axhline(np.median(DqS[(q_grid >= 1) & (q_grid <= 4)]), ls="--",
                  color="C3", alpha=0.4)
    ax[1].axhline(np.median(DqA[(q_grid >= 1) & (q_grid <= 4)]), ls="--",
                  color="C0", alpha=0.4)
    ax[1].set_xlabel("moment order q")
    ax[1].set_ylabel("local slope  D(q) = d sigma / d q")
    ax[1].set_title("BTW: D(q) drifts for S (multifractal), flat for A (FSS)")
    ax[1].legend()
    fig.tight_layout()
    p = os.path.join(FIGDIR, "sandpile_moments_btw.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    log("\nsaved %s" % os.path.relpath(p, ROOT))

    # cache the computed D(q) curves so S12 can overlay BTW without re-running it
    np.savez(os.path.join(OUTDIR, "sandpile_moments_btw.npz"),
             q_grid=q_grid, sigmaS=sigS, DqS=DqS, DqS_sd=DqS_sd,
             sigmaA=sigA, DqA=DqA, DqA_sd=DqA_sd, driftS=driftS, driftA=driftA)
    log("cached BTW D(q) curves to outputs/sandpile_moments_btw.npz")

    log("\nS11 COMPLETE")
    with open(os.path.join(OUTDIR, "sandpile_moments_btw.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
