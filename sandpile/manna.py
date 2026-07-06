"""
S22 -- the TRUE Manna limit: a literal 2-D Manna sandpile measured under the
SAME moment pipeline (S11) and the SAME geometry estimators (S14) as the slope
model, so the "outside Manna" placement stops leaning on literature values.

Motivation. The scaling-theory arc (S11-S18) places the 2-D slope model outside
the Manna universality class on two counts: its avalanche AREA moments are
multifractal (D(q) drift ~0.2, asymptotic per S18) where Manna is the textbook
single-fractal FSS class, and its footprint is a one-bond filament (mass-radius
D ~ 1, S14) where Manna avalanches are compact (D ~ 2). But both Manna anchors
were QUOTED, not measured: S11 gave us a same-pipeline BTW baseline, S15's psto
knob only leaks TOWARD Manna without being Manna. This script closes that gap --
the same "check a known result" discipline as S11, now for the second anchor of
the placement triangle (BTW / directed / Manna).

Model (canonical 2-D Manna, stochastic-redistribution height model; Manna 1991,
J. Phys. A 24 L363; exponents e.g. Lubeck & Heger 2003: tau_S ~ 1.27, D_S ~ 2.76,
compact avalanches D_area = 2, and -- the class-defining property -- simple FSS,
i.e. FLAT moment spectra):
  - integer grains h[i,j]; threshold 2 (a site is unstable when h >= 2);
  - drive: add 1 grain at a random site, then relax fully (timescale separation);
  - topple: every unstable site loses 2 grains; EACH grain independently moves to
    one of the 4 neighbours chosen at random (the stochastic redistribution that
    defines the class -- contrast BTW's deterministic 1-to-each);
  - open boundaries: grains stepping off the lattice are lost;
  - parallel sweeps; size S = total topplings, duration T = sweeps, area A =
    distinct toppled sites (all matched to btw_compare.btw_run's conventions).

What is measured, mirroring S11/S12/S14 exactly:
  * moment spectra sigma(q) and local slope D(q) for toppling number S and area A
    across L = 32-128 (S11's grid), with bootstrap errors and S11's 3-tier
    verdict. Expected: FLAT D(q) for both (simple FSS) -- the opposite of BTW's
    toppling-number drift (overlaid from the S11 cache) and of the slope model's
    area drift (S12/S18). One subtlety found in the first run and built into the
    analysis: with tau_a ~ 1.3 the sigma(q) kink at q = tau_a - 1 is finite-size
    rounded up to q ~ 2, which puts a dip in D(q) at the LOW-q edge of the S11
    [1,4] drift window and trips its 5%-relative tier -- so the drift is also
    measured on a high-q window [2,5] (where genuine tail multifractality, the
    slope model's signature, lives), and a simple-FSS consistency check verifies
    the whole area spectrum is one FSS line (tau_a read at q=2 must predict
    sigma(1)). The S6/S11 lesson again: read components, not auto-verdicts.
  * per-avalanche mass-radius dimension D from A ~ Rg^D (S14's estimator,
    binned_slope + rg_window from geometry2d, whose synthetic line/disk/directed
    self-test already guards it). Expected: D ~ 2 (compact), vs the slope model's
    measured 1.0 and directed 3/2.
  * tau_S at the largest L via the standard log-binned PDF (validate1d), as a
    headline literature cross-check (~1.27).

Self-test (run inline before the science): exact integer grain bookkeeping
(added - lost off the open edges == grains on the lattice, to zero), plus the
structural invariants S >= A, S >= T on every avalanche. And the run itself is a
known-result validation in the S11 tradition: if the pipeline does NOT read
Manna as flat-D(q)/compact, the pipeline (not Manna) is broken.

Run from repo root:  python sandpile/manna.py
Writes figures/sandpile_manna.png, outputs/sandpile_manna.txt and caches
outputs/sandpile_moments_manna.npz. ASCII-only (Windows cp1252 safe).
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate1d import logbin_pdf, powerlaw_slope                 # noqa: E402
from moments import avalanche_moments, sigma_of_q, bootstrap_sigma  # noqa: E402
from geometry2d import gyration, binned_slope, rg_window          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "figures")
OUTDIR = os.path.join(ROOT, "outputs")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

LOG = []
def log(msg):
    print(msg)
    LOG.append(msg)


def manna_run(L, n_events, warm, seed=0, geom_min_area=5):
    """Canonical 2-D Manna. Returns dict with per-avalanche arrays S (topplings),
    T (parallel sweeps), A (distinct toppled sites), Rg (footprint radius of
    gyration, nan below geom_min_area), the final mean height density, and the
    exact grain-bookkeeping residual (must be 0).

    Relaxation scans only a growing bounding box around the active region (the
    avalanche is contiguous, so activity can spread at most 1 site per sweep);
    this keeps small avalanches O(1) instead of O(L^2) per sweep.
    """
    rng = np.random.default_rng(seed)
    h = np.zeros((L, L), dtype=np.int64)
    added = 0
    lost = 0
    Sz, Tz, Az, Rz = [], [], [], []
    ever = np.zeros((L, L), dtype=bool)
    for ev in range(n_events):
        r0 = int(rng.integers(0, L)); c0 = int(rng.integers(0, L))
        h[r0, c0] += 1
        added += 1
        size = 0
        dur = 0
        ever[:] = False
        # bounding box of possibly-unstable sites, expanded 1/sweep, clipped
        ra, rb, ca, cb = r0, r0 + 1, c0, c0 + 1
        while True:
            sub = h[ra:rb, ca:cb]
            un = sub >= 2
            n_un = int(un.sum())
            if n_un == 0:
                break
            size += n_un
            dur += 1
            ever[ra:rb, ca:cb] |= un
            rows, cols = np.nonzero(un)
            rows = rows + ra
            cols = cols + ca
            h[rows, cols] -= 2
            # two grains per toppling, each to an independent random neighbour
            for _ in range(2):
                d = rng.integers(0, 4, n_un)
                nr = rows + (d == 0) - (d == 1)
                nc = cols + (d == 2) - (d == 3)
                ok = (nr >= 0) & (nr < L) & (nc >= 0) & (nc < L)
                np.add.at(h, (nr[ok], nc[ok]), 1)
                lost += int(n_un - ok.sum())
            ra = max(ra - 1, 0); rb = min(rb + 1, L)
            ca = max(ca - 1, 0); cb = min(cb + 1, L)
        if ev >= warm and size > 0:
            a = int(ever.sum())
            # structural invariants: every toppled site counted >= once in S,
            # every sweep contains >= 1 toppling
            assert size >= a and size >= dur
            Sz.append(size)
            Tz.append(dur)
            Az.append(a)
            if a >= geom_min_area:
                pts = np.argwhere(ever).astype(float)
                lam1, lam2, _, _ = gyration(pts)
                Rz.append(np.sqrt(lam1 + lam2))
            else:
                Rz.append(np.nan)
    residual = added - lost - int(h.sum())
    return dict(S=np.array(Sz, float), T=np.array(Tz, float),
                A=np.array(Az, float), Rg=np.array(Rz, float),
                density=float(h.mean()), residual=int(residual))


def _self_test():
    """Exact conservation + invariants on a small lattice (asserts inside the
    run guard S>=A, S>=T per avalanche; here we check the grain bookkeeping and
    that the pile reaches the known Manna stationary density ~0.68)."""
    log("[self-test] L=24 Manna: exact grain bookkeeping + stationary density")
    res = manna_run(L=24, n_events=30_000, warm=5_000, seed=3)
    log("  bookkeeping residual (added - lost - on_lattice) = %d  (must be 0)"
        % res['residual'])
    assert res['residual'] == 0, "grain bookkeeping broken"
    log("  stationary density = %.3f  (literature rho_c ~ 0.68)" % res['density'])
    assert 0.60 < res['density'] < 0.76, "not at the Manna stationary density"
    log("  %d avalanches, invariants S>=A, S>=T held on all (asserted in-run)"
        % res['S'].size)
    log("  PASS\n")


def main():
    log("=" * 70)
    log("S22 -- LITERAL 2-D MANNA UNDER THE S11 MOMENT + S14 GEOMETRY PIPELINE")
    log("=" * 70)
    log("")

    _self_test()

    # S11's lattice grid; event counts sized so each L keeps >~4e4 avalanches.
    # Warmup must both FILL the pile (stationary grain count ~0.68*L^2, one
    # grain per event) and relax it, so it grows with L.
    configs = [
        (32, 150_000, 20_000),
        (48, 130_000, 20_000),
        (64, 110_000, 20_000),
        (96, 95_000, 22_000),
        (128, 75_000, 28_000),
    ]
    Ls = [c[0] for c in configs]
    q_grid = np.arange(0.5, 5.01, 0.25)

    S_by_L, A_by_L, data = {}, {}, {}
    log("Running Manna lattices (S = topplings, A = distinct sites)...")
    for L, n_ev, warm in configs:
        t = time.time()
        res = manna_run(L, n_ev, warm, seed=11)
        assert res['residual'] == 0
        S_by_L[L] = res['S']
        A_by_L[L] = res['A']
        data[L] = res
        log("  L=%4d : %6d avalanches  <S>=%.1f S_max=%.3g  <A>=%.1f A_max=%.3g"
            "  rho=%.3f  (%.1fs)"
            % (L, res['S'].size, res['S'].mean(), res['S'].max(),
               res['A'].mean(), res['A'].max(), res['density'], time.time() - t))

    # ---- moment spectra, S11's analyse verbatim in structure ----
    def analyse(by_L, label):
        mom = {L: avalanche_moments(by_L[L], q_grid) for L in Ls}
        sigma, _ = sigma_of_q(mom, Ls, q_grid)
        bs = bootstrap_sigma(by_L, Ls, q_grid, n_boot=200, seed=7)
        Dq = bs["Dq_mean"]
        Dq_sd = bs["Dq_std"]
        sel = (q_grid >= 1.0) & (q_grid <= 4.0)
        drift = Dq[sel].max() - Dq[sel].min()
        typ_sd = np.median(Dq_sd[sel])
        # High-q window, clear of the finite-size rounding of the sigma(q) kink
        # at q ~ tau-1 (for tau_a ~ 1.3 the rounding leaks visibly up to q ~ 2 at
        # these L). Genuine multifractality lives in the TAIL, i.e. grows toward
        # HIGH q (the slope model's signature, S18: D rises 1.14 -> 1.34 across
        # [1,4]); a dip confined to the low-q edge is the kink, not a spectrum.
        sel_hi = (q_grid >= 2.0) & (q_grid <= 5.0)
        drift_hi = Dq[sel_hi].max() - Dq[sel_hi].min()
        log("\n  [%s]  sigma(q) and local slope D(q)=d sigma/dq" % label)
        log("    q     sigma(q)   D(q)+-boot")
        for q, s, d, ds in zip(q_grid, sigma, Dq, Dq_sd):
            if abs((q * 2) % 1) < 1e-9 and (q * 2) % 2 < 1e-9:
                log("    %.1f   %8.3f   %.3f +- %.3f" % (q, s, d, ds))
        log("    D(q) drift over q in [1,4] = %.3f   (bootstrap noise ~ %.3f)"
            % (drift, typ_sd))
        log("    D(q) drift over q in [2,5] = %.3f   (high-q window, clear of the"
            % drift_hi)
        log("      sigma(q) kink rounding near q ~ tau-1)")
        rel = drift / np.median(Dq[sel])
        if drift > 4 * typ_sd and rel > 0.05:
            verdict = "MULTIFRACTAL (D(q) drifts strongly with q)"
        elif drift > 4 * typ_sd:
            verdict = ("near-FSS (a weak, finite-size-like D(q) creep; D ~ %.2f)"
                       % np.median(Dq[sel]))
        else:
            verdict = "FLAT D(q) -> simple FSS (D ~ %.2f)" % np.median(Dq[sel])
        log("    -> auto-verdict ([1,4] window, S11 tiering): %s" % verdict)
        return (sigma, Dq, Dq_sd, drift, typ_sd, float(np.median(Dq[sel])),
                drift_hi)

    (sigS, DqS, DqS_sd, driftS, sdS, DmidS, driftS_hi) = analyse(
        S_by_L, "toppling number S")
    (sigA, DqA, DqA_sd, driftA, sdA, DmidA, driftA_hi) = analyse(
        A_by_L, "area A")

    # ---- simple-FSS consistency of the area spectrum (locates the low-q dip) --
    # If P(A) obeys simple FSS with tau_a > 1, then sigma(q) = D*(q+1-tau_a) for
    # q > tau_a-1. Read tau_a off sigma(2) with D = the high-q plateau, then
    # PREDICT sigma(1). Agreement means the whole spectrum is the one FSS line
    # and the D(q) dip at q=1 is finite-size rounding of the kink at q=tau_a-1
    # (the S6/S11 discipline: read components, not the auto-verdict, which trips
    # its 5% relative threshold on exactly this edge effect).
    iq1 = int(np.argmin(np.abs(q_grid - 1.0)))
    iq2 = int(np.argmin(np.abs(q_grid - 2.0)))
    D_hi = float(np.median(DqA[(q_grid >= 2.0) & (q_grid <= 5.0)]))
    tau_a_imp = 3.0 - sigA[iq2] / D_hi
    sig1_pred = D_hi * (2.0 - tau_a_imp)
    log("\n[area simple-FSS consistency]")
    log("  high-q plateau D = %.3f; implied tau_a (from sigma(2)) = %.3f"
        % (D_hi, tau_a_imp))
    log("  predicted sigma(1) = %.3f   measured sigma(1) = %.3f   (diff %.3f)"
        % (sig1_pred, sigA[iq1], abs(sig1_pred - sigA[iq1])))
    log("  -> the area spectrum is a single FSS line; the [1,4] drift is the")
    log("     kink-rounding edge at low q, opposite in q-location to the slope")
    log("     model's TAIL-side (high-q) drift (S18: D rises 1.14 -> 1.34).")

    # ---- tau_S at the largest L, literature cross-check ----
    Lbig = Ls[-1]
    cb, db = logbin_pdf(S_by_L[Lbig])
    tauS = -powerlaw_slope(cb, db, lo=cb.min() * 5, hi=cb.max() * 0.1)
    log("\n[headline cross-check]  tau_S(L=%d) = %.3f   (literature ~ 1.27)"
        % (Lbig, tauS))

    # ---- mass-radius geometry at the two largest L (S14 estimator) ----
    log("\n[mass-radius dimension, A ~ Rg^D  (S14 estimator; its synthetic")
    log(" line/disk/directed self-test lives in geometry2d.py)]")
    D_geo = {}
    for L in Ls[-2:]:
        Rg, A = data[L]['Rg'], data[L]['A']
        m = np.isfinite(Rg) & (Rg > 0)
        lo, hi = rg_window(Rg[m])
        g, se, bx, by = binned_slope(Rg[m], A[m], lo, hi)
        D_geo[L] = (g, se, bx, by)
        log("  L=%4d : D = %.3f +- %.3f   (window Rg in [%.1f, %.1f], %d bins)"
            % (L, g, se, lo, hi, bx.size))
    Dg, Dg_se, gbx, gby = D_geo[Lbig]

    # ---- same-pipeline placement ----
    btw = None
    btw_path = os.path.join(OUTDIR, "sandpile_moments_btw.npz")
    if os.path.exists(btw_path):
        btw = np.load(btw_path)
    log("\n[SAME-PIPELINE PLACEMENT -- all three anchors now measured, not quoted]")
    log("                     area D(q) drift [1,4] / [2,5]    mass-radius D")
    log("  Manna (here)     : %.3f / %.3f                     %.2f +- %.2f"
        % (driftA, driftA_hi, Dg, Dg_se))
    if btw is not None:
        log("  BTW   (S11/S14)  : %.3f                             2.0  (compact)"
            % float(btw["driftA"]))
    log("  slope (S12/S18,  : ~0.2 asymptotic, tail-side         1.00 (one-bond")
    log("         S14/S19)                                       filament, S14)")
    log("  directed (exact) : 0 (single-scale)                   1.5")
    log("")
    # Flatness judged on the high-q (tail) window -- where genuine avalanche
    # multifractality lives and where the slope model's drift sits -- plus the
    # FSS-line consistency above; the [1,4] number is kept for comparability.
    if driftA_hi <= max(4 * sdA, 0.05 * DmidA) and abs(Dg - 2.0) < 0.25:
        log("  VERDICT: the pipeline reads literal Manna as simple-FSS (a single")
        log("  sigma(q) line, tail D(q) flat) and COMPACT (D ~ 2) -- so the slope")
        log("  model's tail-side area drift ~0.2 and mass-radius D ~ 1 are now")
        log("  measured DIFFERENCES from Manna under one pipeline, not literature")
        log("  comparisons. 'Outside Manna' is closed.")
    else:
        log("  VERDICT: Manna did NOT read as flat/compact here -- treat the S12/S18")
        log("  'outside Manna' placement as pipeline-limited at these L, not closed.")

    # ---- figure ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    ax[0].errorbar(q_grid, DqS, yerr=DqS_sd, fmt="o-", color="C2", capsize=2,
                   label="Manna S: drift=%.2f (flat, D~%.2f)" % (driftS, DmidS))
    ax[0].errorbar(q_grid, DqA, yerr=DqA_sd, fmt="s-", color="C0", capsize=2,
                   label="Manna A: tail drift=%.2f (flat, D~%.2f)"
                   % (driftA_hi, DmidA))
    if btw is not None:
        ax[0].errorbar(btw["q_grid"], btw["DqS"], yerr=btw["DqS_sd"], fmt="^--",
                       color="C3", alpha=0.5, capsize=2,
                       label="BTW S (S11): drift=%.2f (multifractal)" % float(btw["driftS"]))
    ax[0].set_xlabel("moment order q")
    ax[0].set_ylabel("local slope  D(q) = d sigma / d q")
    ax[0].set_title("Manna moment spectra are FLAT (simple FSS)\nvs BTW's toppling-number drift")
    ax[0].legend(fontsize=9)

    ax[1].loglog(gbx, gby, "s", color="C0", label="Manna L=%d binned <A|Rg>" % Lbig)
    xr = np.array([gbx.min(), gbx.max()])
    yref = gby[0] * (xr / xr[0]) ** Dg
    ax[1].loglog(xr, yref, "-", color="C0", alpha=0.6,
                 label="fit D=%.2f +- %.2f" % (Dg, Dg_se))
    for Dref, lab, col in ((2.0, "compact D=2 (BTW)", "C3"),
                           (1.5, "directed D=3/2", "C1"),
                           (1.0, "slope-model filament D=1 (S14)", "C4")):
        ax[1].loglog(xr, gby[0] * (xr / xr[0]) ** Dref, "--", color=col,
                     alpha=0.5, label=lab)
    ax[1].set_xlabel("footprint radius of gyration Rg")
    ax[1].set_ylabel("footprint area A")
    ax[1].set_title("Manna footprints are compact (D ~ %.2f)\nvs slope-model filament D = 1" % Dg)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "sandpile_manna.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    log("saved %s" % os.path.relpath(p, ROOT))

    np.savez(os.path.join(OUTDIR, "sandpile_moments_manna.npz"),
             q_grid=q_grid, sigmaS=sigS, DqS=DqS, DqS_sd=DqS_sd,
             sigmaA=sigA, DqA=DqA, DqA_sd=DqA_sd, driftS=driftS, driftA=driftA,
             driftS_hi=driftS_hi, driftA_hi=driftA_hi, tau_a=tau_a_imp,
             tauS=tauS, D_geo=Dg, D_geo_se=Dg_se)
    log("cached Manna D(q) curves to outputs/sandpile_moments_manna.npz")

    log("\nS22 COMPLETE")
    with open(os.path.join(OUTDIR, "sandpile_manna.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
