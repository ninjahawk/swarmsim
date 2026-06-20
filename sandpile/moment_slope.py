"""
S12 -- moment-scaling test of the 2-D continuous SLOPE sandpile: simple FSS or
multifractal? With the clean discriminator being avalanche AREA.

Background. S4 separated the slope model from canonical BTW on a single avalanche-
size exponent (tau_S=0.89 vs 1.14), hedged by the bond-vs-site caveat. The
decisive, caveat-free test (validated on BTW in S11) is the moment spectrum: a
model obeying simple finite-size scaling has a CONSTANT local slope D(q)=d sigma/dq
(where <x^q> ~ L^sigma(q)); a multifractal one has a DRIFTING D(q).

Two things this script had to get right, learned the hard way:

  1. The avalanche SIZE and ENERGY have exponent tau<1, so every positive moment
     is CUTOFF-DOMINATED (<S> ~ 0.1*S_max) -- set by the few largest, system-
     spanning avalanches. Their moment scaling is noisy and correction-laden and
     does NOT give a clean verdict. The cure (from S11) is avalanche AREA = number
     of DISTINCT toppled bonds (the footprint), which is bounded and not cutoff-
     dominated -- the clean observable. The engine records it (sandpile_fast.py,
     validated bit-for-bit against a brute-force recount).

  2. The 2-D slope pile equilibrates SLOWLY and only from above: started over-
     steep it begins active (bonds near Zc) and relaxes its mean slope down to the
     repose ~2.5; started near repose at large L it stays dormant and bleeds out.
     So each lattice needs a long warmup, scaled with L and verified by the mean
     slope settling (NOT by the cutoff-dominated <S>, which is too noisy to judge
     stationarity). We warm up with record_series off (no memory cost), confirm
     the mean slope, then measure a recorded window from the equilibrated state.

Run from repo root:  python sandpile/moment_slope.py
Writes figures/sandpile_moments_slope.png and outputs/sandpile_moments_slope.txt.
Requires outputs/sandpile_moments_btw.npz (S11) for the BTW overlay.
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
from sandpile_fast import run_sandpile2d_fast
from sandpile2d import pyramid_ic
from fss2d import measure_multi
from validate1d import logbin_pdf, powerlaw_slope
from moments import avalanche_moments, sigma_of_q, local_slope

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "figures")
OUTDIR = os.path.join(ROOT, "outputs")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

LOG = []
def log(msg):
    print(msg)
    LOG.append(msg)


def equilibrated_run(L, warm, window, seed, psto=0.0):
    """Two-phase run: warm up to the SOC attractor with the series unrecorded (so
    a long, L-scaled warmup costs no memory), then measure a recorded window from
    the equilibrated state. Returns per-avalanche (E, S, A) and the final mean bond
    slope (the stationarity gauge). Area needs track_area on the measurement leg.

    psto (additive, default 0.0) is the S15 stochastic-split knob; warmup and window
    share it so the pile equilibrates under the same dynamics it is measured in."""
    warmed = run_sandpile2d_fast(L=L, eps=0.1, Zc=5.0, n_iter=warm, seed=seed,
                                 record_series=False, S0=pyramid_ic(L, 4.5), psto=psto)
    res = run_sandpile2d_fast(L=L, eps=0.1, Zc=5.0, n_iter=window, seed=seed + 101,
                              record_series=True, S0=warmed['S'], track_area=True,
                              psto=psto)
    E, S, T = measure_multi(res['disp'], res['act'])
    # area per avalanche: group the first-topple series the same way
    A, _, _ = measure_multi(res['area'], res['act'])
    mslope = np.abs(np.diff(res['S'], axis=0)).mean()
    return E, S, A, mslope


def jackknife_Dq(seed_arrays_by_L, Ls, q_grid, n_groups=4):
    """Central D(q) from the pooled sample, plus a seed-group spread as the honest
    error (captures run-to-run variation that within-sample bootstrap cannot)."""
    pooled = {L: np.concatenate(seed_arrays_by_L[L]) for L in Ls}
    mom = {L: avalanche_moments(pooled[L], q_grid) for L in Ls}
    sigma, _ = sigma_of_q(mom, Ls, q_grid)
    Dq = local_slope(q_grid, sigma)
    groups = []
    for g in range(n_groups):
        momg = {L: avalanche_moments(
            np.concatenate(seed_arrays_by_L[L][g::n_groups]), q_grid) for L in Ls}
        sg, _ = sigma_of_q(momg, Ls, q_grid)
        groups.append(local_slope(q_grid, sg))
    return sigma, Dq, np.array(groups).std(0)


def report(sigma, Dq, Dq_sd, q_grid, label):
    sel = (q_grid >= 1.0) & (q_grid <= 4.0)
    drift = Dq[sel].max() - Dq[sel].min()
    typ_sd = np.median(Dq_sd[sel]); Dmid = np.median(Dq[sel])
    log("\n  [%s]   sigma(q), local slope D(q)" % label)
    log("    q     sigma(q)   D(q)+-grp")
    for q, s, d, ds in zip(q_grid, sigma, Dq, Dq_sd):
        if (q * 2) % 2 < 1e-9:
            log("    %.1f   %8.3f   %.3f +- %.3f" % (q, s, d, ds))
    log("    D(q) drift over q in [1,4] = %.3f  (seed-group noise ~%.3f, D_mid ~%.2f)"
        % (drift, typ_sd, Dmid))
    return drift, typ_sd, Dmid


def main():
    log("=" * 70)
    log("S12 -- SLOPE-SANDPILE MOMENT SCALING (equilibrated; area = clean observable)")
    log("=" * 70)

    # (L, warm, window, n_seeds). Warmups are L-scaled and were checked to settle
    # the mean slope (~2.5-2.65); L capped at 256, the largest we can equilibrate
    # and verify. Two-phase running keeps memory to the window only.
    configs = [
        (64,   5_000_000,  4_000_000, 8),
        (96,   8_000_000,  5_000_000, 8),
        (128, 12_000_000,  6_000_000, 8),
        (192, 18_000_000,  8_000_000, 10),
        (256, 30_000_000, 10_000_000, 10),
    ]
    Ls = [c[0] for c in configs]
    q_grid = np.arange(0.5, 5.01, 0.25)

    E_seeds, S_seeds, A_seeds = {}, {}, {}
    log("\nRunning equilibrated 2-D slope lattices (warm unrecorded, then measure)...")
    for L, warm, window, n_seeds in configs:
        t = time.time()
        Es, Ss, As, slopes = [], [], [], []
        for sd in range(n_seeds):
            E, S, A, ms = equilibrated_run(L, warm, window, seed=3 + sd)
            Es.append(E); Ss.append(S); As.append(A); slopes.append(ms)
        E_seeds[L] = Es; S_seeds[L] = Ss; A_seeds[L] = As
        nav = sum(e.size for e in Es)
        log("  L=%4d : %7d avalanches (%d seeds)  mean-slope=%.2f+-%.2f  "
            "<A>=%.1f A_max=%.0f  <S>=%.1f  (%.0fs)"
            % (L, nav, n_seeds, np.mean(slopes), np.std(slopes),
               np.concatenate(As).mean(), np.concatenate(As).max(),
               np.concatenate(Ss).mean(), time.time() - t))

    # scaling-quality: <A> should now be a clean power law (the failure mode that
    # sank size/energy). Print local log-log slopes.
    A_pool = {L: np.concatenate(A_seeds[L]) for L in Ls}
    S_pool = {L: np.concatenate(S_seeds[L]) for L in Ls}
    log("\n[scaling-quality: local log-log slope between consecutive L]")
    log("  L_a->L_b     <A> slope    <S> slope")
    for a, b in zip(Ls[:-1], Ls[1:]):
        eA = np.log(A_pool[b].mean() / A_pool[a].mean()) / np.log(b / a)
        eS = np.log(S_pool[b].mean() / S_pool[a].mean()) / np.log(b / a)
        log("  %3d->%3d      %.2f         %.2f" % (a, b, eA, eS))
    log("  (area should be steady ~constant; size wanders -- cutoff-dominated, tau<1)")

    sigA, DqA, DqA_sd = jackknife_Dq(A_seeds, Ls, q_grid)
    driftA, sdA, DmidA = report(sigA, DqA, DqA_sd, q_grid, "area A (clean observable)")
    sigS, DqS, DqS_sd = jackknife_Dq(S_seeds, Ls, q_grid)
    driftS, sdS, DmidS = report(sigS, DqS, DqS_sd, q_grid, "toppling number S (cutoff-dominated)")
    sigE, DqE, DqE_sd = jackknife_Dq(E_seeds, Ls, q_grid)
    driftE, sdE, DmidE = report(sigE, DqE, DqE_sd, q_grid, "energy E (cutoff-dominated)")

    btw = np.load(os.path.join(OUTDIR, "sandpile_moments_btw.npz"))
    log("\n[verdict on the clean observable -- avalanche AREA]")
    log("  slope-model area : D(q) drift = %.3f   D_area(mid) = %.2f" % (driftA, DmidA))
    log("  BTW area  (S11)  : D(q) drift = %.3f   D_area ~ %.2f  (near-FSS)"
        % (float(btw["driftA"]), np.median(btw["DqA"])))
    log("  BTW topplings    : D(q) drift = %.3f   (multifractal)" % float(btw["driftS"]))
    if driftA < 4 * sdA or driftA / DmidA < 0.03:
        v = ("simple FSS -- the slope model's avalanche AREA is single-fractal, "
             "dimension D_area ~ %.2f" % DmidA)
    else:
        v = ("multifractal -- the area shows a resolved D(q) drift of %.3f" % driftA)
    log("  -> %s" % v)
    log("  Geometry: D_area ~ %.2f (slope) vs ~2.0 (BTW) -- the slope model's"
        % DmidA)
    log("  avalanche footprints are FILAMENTARY, not compact like BTW's.")

    # ---- figure: area PDFs + D(q) overlay ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    cmap = plt.cm.viridis(np.linspace(0, 0.85, len(Ls)))
    for c, L in zip(cmap, Ls):
        ca, da = logbin_pdf(A_pool[L])
        ax[0].loglog(ca, da, "-", color=c, label="L=%d" % L)
    ax[0].set_xlabel("avalanche area A (distinct toppled bonds)")
    ax[0].set_ylabel("PDF(A)")
    ax[0].set_title("Slope-model area PDFs (equilibrated)")
    ax[0].legend(fontsize=8)

    ax[1].errorbar(q_grid, DqA, yerr=DqA_sd, fmt="o-", color="C0", capsize=2,
                   label="slope area: drift=%.2f, D~%.2f" % (driftA, DmidA))
    ax[1].errorbar(q_grid, btw["DqA"], yerr=btw["DqA_sd"], fmt="s-", color="C4",
                   alpha=0.6, capsize=2, label="BTW area: drift=%.2f, D~2.05"
                   % float(btw["driftA"]))
    ax[1].errorbar(q_grid, btw["DqS"], yerr=btw["DqS_sd"], fmt="^-", color="C3",
                   alpha=0.45, capsize=2, label="BTW topplings: drift=%.2f (multifractal)"
                   % float(btw["driftS"]))
    ax[1].axhline(DmidA, ls="--", color="C0", alpha=0.4)
    ax[1].set_xlabel("moment order q")
    ax[1].set_ylabel("local slope  D(q) = d sigma / d q")
    ax[1].set_title("Area D(q): flat = simple FSS (footprint dimension)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "sandpile_moments_slope.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    log("\nsaved %s" % os.path.relpath(p, ROOT))

    log("\nS12 COMPLETE")
    with open(os.path.join(OUTDIR, "sandpile_moments_slope.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
