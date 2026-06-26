"""
S18 -- The area-multifractality closure: is the slope model's anomalous avalanche-
AREA scaling a true asymptotic property, or a finite-size corona that heals to
simple FSS as L grows?

The open question. S12 measured the avalanche AREA moment spectrum of the 2-D
continuous slope sandpile and found the local slope D(q) = d sigma(q)/dq (where
<A^q> ~ L^sigma(q)) DRIFTS with q -- a rise of ~0.27 over q in [1,4] -- and read
that as genuine multifractality placing the model in the deterministic (BTW/Zhang)
anomalous-scaling family, NOT the stochastic Manna single-fractal FSS class. But S12
could only equilibrate to L = 256 and quoted that caveat explicitly: "L = 512 is left
to future work." The drift was measured by pooling ALL of L = 64..256 into one
moment regression, so it is an average over the very finite-size range where the
healing (if any) would happen.

Two developments make the question answerable and SHARP now:
  1. S16's equilibration enabler (equilibrate2d.equilibrate) gives verified-stationary
     states at L = 384 and 512, past the S12 ceiling -- a real lever arm in L.
  2. S17 closed the duration thread by proving the SPATIAL sector becomes EXACTLY
     single-scale as L grows: the conditional exponents reach gamma(S|A) -> 2 (size
     scales as area squared) and gamma(A|T) -> 1, with the total activity space-filling
     (S, E ~ L^2). Those two facts have a logical consequence for the UNCONDITIONAL
     area moments: if S ~ A^2 exactly and S ~ L^2 (space-filling), then A ~ L^1 EXACTLY
     -- a mono-fractal filament, D_area -> 1 with NO q-drift. So S17's conditional
     result predicts the S12 area multifractality should HEAL. S18 tests that
     prediction with an independent observable (unconditional moments vs S17's
     conditional exponents): a cross-check of the chapter's central claim.

Method. Warm verified-stationary states at L = 64, 96, 128, 192, 256, 384, 512 via
equilibrate2d.equilibrate (S16), run a recorded window from each (several seeds),
and gather the per-avalanche AREA (distinct toppled bonds -- S12's clean, non-cutoff-
dominated observable, validated bit-for-bit in the engine). The multifractal signature
is LOCALIZED in L by a SLIDING WINDOW: for each set of three consecutive lattice sizes
{L_a, L_b, L_c} regress log<A^q> on log L to get sigma(q) and D(q) = d sigma/dq from
just that window, and read the drift of D(q) over q in [1,4]. As the window slides to
larger L, a finite-size corona makes the windowed drift SHRINK toward zero (the area
becomes simple-FSS / mono-fractal); a true asymptotic multifractal keeps it nonzero.
Drift-vs-1/L_window is extrapolated to L -> infinity (the S17 protocol). The drift per
window uses a quadratic fit of sigma(q) over [1,4] (a smoothed D(q) curvature, robust
to the noisy central-difference endpoints), with a seed-group jackknife error; the raw
max-min range of D(q) is reported alongside for transparency (the S6/S11/S17 discipline:
read the directly-measured components, not one derived auto-verdict).

Self-test (run this file). The sliding-window estimator is checked on two synthetic
area sources at the SAME seven lattice sizes:
  * NULL -- an exact simple-FSS source (one power law x^{-tau} with a single cutoff
    x_c = A*L^D): the windowed drift must read ~0 at EVERY window and extrapolate to
    ~0, i.e. the method neither manufactures multifractality nor a spurious L-trend.
    This is the critical guard that makes a nonzero intercept on the real data
    believable.
  * SENSITIVITY -- a bifractal mixture (a numerous D1=1.0 population plus a rare
    D2=1.7 one) whose moment spectrum genuinely curves: the windowed drift must be
    clearly nonzero (the method detects a real D(q) drift when one is present). The
    test asserts detection, not persistence -- whether the real data's drift persists
    or heals is exactly the experimental result, trusted because the NULL is clean.

Run from repo root:  python sandpile/area_multifractality.py   [smoke: add 'smoke']
Writes figures/sandpile_area_multifractal.png and outputs/sandpile_area_multifractal.txt.
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
from sandpile_fast import run_sandpile2d_fast            # noqa: E402
from fss2d import measure_multi                           # noqa: E402
from equilibrate2d import equilibrate                     # noqa: E402
from moments import (avalanche_moments, sigma_of_q,       # noqa: E402
                     local_slope, _ols_slope_se, _sample_fss)
from validate1d import logbin_pdf                          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "figures")
OUTDIR = os.path.join(ROOT, "outputs")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

Q_LO, Q_HI = 1.0, 4.0          # the resolved range (S12: high-q area moments noisy)

LOG = []
def log(msg):
    print(msg)
    LOG.append(msg)


# ---------------------------------------------------------------------------
# The multifractality metric, localized to a window of lattice sizes.
# ---------------------------------------------------------------------------
def _drift_from_pool(pool, Ls, q_grid):
    """Area D(q) and its q-drift over [Q_LO, Q_HI] from one pooled sample set
    {L: area array}. Returns sigma(q), D(q), the smoothed drift (quadratic fit of
    sigma over the window, robust to endpoint noise) and the raw max-min range."""
    mom = {L: avalanche_moments(pool[L], q_grid) for L in Ls}
    sigma, _ = sigma_of_q(mom, Ls, q_grid)
    Dq = local_slope(q_grid, sigma)
    sel = (q_grid >= Q_LO) & (q_grid <= Q_HI)
    # smoothed drift: sigma(q) ~ c0 + c1 q + c2 q^2 over [Q_LO,Q_HI] -> D = c1 + 2c2 q,
    # so D(Q_HI) - D(Q_LO) = 2 c2 (Q_HI - Q_LO). Curvature c2>0 is the multifractal sign.
    c2, c1, _ = np.polyfit(q_grid[sel], sigma[sel], 2)
    drift_smooth = 2.0 * c2 * (Q_HI - Q_LO)
    d_lo = c1 + 2.0 * c2 * Q_LO
    d_hi = c1 + 2.0 * c2 * Q_HI
    drift_range = Dq[sel].max() - Dq[sel].min()
    dmid = float(np.median(Dq[sel]))
    return dict(sigma=sigma, Dq=Dq, drift=drift_smooth, drift_range=drift_range,
                d_lo=d_lo, d_hi=d_hi, dmid=dmid)


def window_drift(area_seeds_by_L, Ls_win, q_grid, n_groups=4):
    """Central windowed area-D(q) drift over the lattice sizes Ls_win, with a
    seed-group jackknife error that captures run-to-run (between equilibrated state)
    variation -- the honest uncertainty, larger than a within-sample bootstrap."""
    pooled = {L: np.concatenate(area_seeds_by_L[L]) for L in Ls_win}
    central = _drift_from_pool(pooled, Ls_win, q_grid)
    drifts = []
    for g in range(n_groups):
        parts = {L: area_seeds_by_L[L][g::n_groups] for L in Ls_win}
        if any(len(parts[L]) == 0 for L in Ls_win):
            continue                                   # group empty at some L (few seeds)
        sub = {L: np.concatenate(parts[L]) for L in Ls_win}
        drifts.append(_drift_from_pool(sub, Ls_win, q_grid)['drift'])
    se = float(np.std(drifts)) if len(drifts) >= 2 else np.nan
    central['se'] = se
    return central


# ---------------------------------------------------------------------------
# Measurement: warm a verified-stationary state per seed, gather avalanche area.
# ---------------------------------------------------------------------------
def measure_L(L, window, n_seeds, smoke=False):
    """For each seed: equilibrate to a verified-stationary repose (S16 enabler),
    run a recorded window, gather per-avalanche AREA (S12 clean observable, via
    measure_multi on the validated first-topple 'area' series). Returns the list of
    per-seed area arrays plus the mean/spread of the repose slope (the stationarity
    gauge)."""
    areas, slopes = [], []
    for s in range(n_seeds):
        base = 1 + 1000 * s
        warm = equilibrate(L, seed=base) if not smoke else \
            equilibrate(L, seed=base, chunk=4_000_000, max_chunks=6)
        res = run_sandpile2d_fast(L=L, eps=0.1, Zc=5.0, n_iter=window, seed=base + 500,
                                  record_series=True, S0=warm['S'], track_area=True)
        A, _, _ = measure_multi(res['area'], res['act'])
        areas.append(A)
        slopes.append(warm['mean_slope'])
    return areas, float(np.mean(slopes)), float(np.std(slopes))


# ---------------------------------------------------------------------------
# Self-test: a null (simple-FSS) and a sensitivity (bifractal) synthetic source.
# ---------------------------------------------------------------------------
def _synth_fss(Ls, D=1.05, tau=0.90, n_per=300_000, seed=1):
    """Exact simple-FSS area: one power law x^{-tau} on [1, A*L^D]. D(q) is flat by
    construction (moments.py self-test), so the windowed drift must read ~0 at every
    window -- the null the real-data verdict is judged against."""
    rng = np.random.default_rng(seed)
    return {L: [_sample_fss(n_per, tau, 1.0, L ** D, rng)] for L in Ls}


def _synth_bifractal(Ls, D1=1.0, D2=1.7, frac2=0.05, tau=0.90, n_per=300_000, seed=2):
    """A bifractal mixture: a numerous D1 population plus a rare, larger-D2 one. The
    rare big-cutoff population lifts the high-q moments while the numerous one sets the
    low-q ones, so sigma(q) genuinely CURVES and D(q) drifts across q -- a source with
    real multifractality in the measured range, to confirm the estimator detects a
    drift when one exists. NOTE (cf S17's synthetic): the SIGN of the resulting drift
    is an artefact of this finite-L crossover construction (the moving crossover
    contaminates the low-q regression), NOT meant to match the slope model's positive
    rise; the test asserts only that a real sigma(q) curvature is DETECTED (|drift|
    clearly nonzero), the sign-agnostic sensitivity guard."""
    rng = np.random.default_rng(seed)
    out = {}
    for L in Ls:
        n2 = int(frac2 * n_per)
        a1 = _sample_fss(n_per - n2, tau, 1.0, L ** D1, rng)
        a2 = _sample_fss(n2, tau, 1.0, L ** D2, rng)
        out[L] = [np.concatenate([a1, a2])]
    return out


def _windows_of(Ls):
    """Overlapping triples of consecutive lattice sizes (the sliding L-window)."""
    return [Ls[i:i + 3] for i in range(len(Ls) - 2)]


def _self_test():
    print("=" * 70)
    print("area_multifractality.py self-test: null (FSS) ~0 drift; bifractal detected")
    print("=" * 70)
    Ls = [64, 96, 128, 192, 256, 384, 512]
    q_grid = np.arange(0.5, 4.51, 0.25)
    wins = _windows_of(Ls)

    fss = _synth_fss(Ls)
    drift_fss = [window_drift(fss, w, q_grid, n_groups=1)['drift'] for w in wins]
    print("  NULL  simple-FSS source, windowed drift by window center:")
    for w, d in zip(wins, drift_fss):
        print("    L~%3d (%s): drift = %+.3f" % (int(np.exp(np.mean(np.log(w)))),
              "-".join(str(x) for x in w), d))
    assert max(abs(d) for d in drift_fss) < 0.04, \
        "method fabricates an area drift on simple-FSS data"

    bif = _synth_bifractal(Ls)
    drift_bif = [window_drift(bif, w, q_grid, n_groups=1)['drift'] for w in wins]
    print("  SENSITIVITY  bifractal source, windowed drift by window center (sign is")
    print("               construction-specific; magnitude is what must be detected):")
    for w, d in zip(wins, drift_bif):
        print("    L~%3d (%s): drift = %+.3f" % (int(np.exp(np.mean(np.log(w)))),
              "-".join(str(x) for x in w), d))
    assert max(abs(d) for d in drift_bif) > 0.06, \
        "method blind to a genuine multifractal area drift"
    print("  -> null reads flat (|drift|<0.04), real curvature detected (|drift|>0.06).")
    print("self-test OK: a nonzero windowed drift on the model is meaningful.\n")


# ---------------------------------------------------------------------------
def main(smoke=False):
    log("=" * 70)
    log("S18 -- AREA MULTIFRACTALITY: does the S12 area D(q) drift heal as L grows?")
    log("=" * 70)
    log("Clean observable = avalanche AREA (S12). Drift of D(q) over q in [%.0f,%.0f]"
        % (Q_LO, Q_HI))
    log("localized by a sliding 3-size L-window; extrapolated vs 1/L. S12 pooled-")
    log("drift (L<=256) ~ 0.27. S17 predicts area -> mono-fractal D=1 (heal) since")
    log("S ~ A^2 exactly and S ~ L^2, hence A ~ L^1.")

    if smoke:
        configs = [(64, 4_000_000, 3), (96, 4_000_000, 3),
                   (128, 4_000_000, 3), (192, 5_000_000, 3)]
    else:
        # (L, recorded window, n_seeds). Warmup handled by equilibrate() (S16); large
        # L get bigger windows (fewer, larger avalanches) and >=4 seeds for jackknife.
        configs = [
            (64,   8_000_000, 6),
            (96,  10_000_000, 6),
            (128, 10_000_000, 6),
            (192, 12_000_000, 5),
            (256, 14_000_000, 5),
            (384, 16_000_000, 5),
            (512, 20_000_000, 5),
        ]
    Ls = [c[0] for c in configs]
    q_grid = np.arange(0.5, 4.51, 0.25)

    area_seeds, repose = {}, {}
    log("\nWarming verified-stationary states (S16 enabler) and gathering area per L...")
    for L, window, n_seeds in configs:
        t = time.time()
        areas, ms, ss = measure_L(L, window, n_seeds, smoke=smoke)
        area_seeds[L] = areas
        repose[L] = ms
        nav = sum(a.size for a in areas)
        pooled = np.concatenate(areas)
        log("  L=%4d : %7d avalanches (%d seeds)  repose=%.2f+-%.2f  "
            "<A>=%.1f A_max=%.0f  (%.0fs)"
            % (L, nav, n_seeds, ms, ss, pooled.mean(), pooled.max(), time.time() - t))

    # area should still grow ~linearly with L if the footprint stays filamentary
    log("\n[footprint scaling: local log-log slope of <A> between consecutive L]")
    log("  L_a->L_b    <A> slope   (1.0 = filament A~L; 2.0 = compact A~L^2)")
    for a, b in zip(Ls[:-1], Ls[1:]):
        eA = np.log(np.concatenate(area_seeds[b]).mean()
                    / np.concatenate(area_seeds[a]).mean()) / np.log(b / a)
        log("  %3d->%3d     %.2f" % (a, b, eA))

    # ---- the sliding-window multifractality trend ----
    wins = _windows_of(Ls)
    centers, drifts, ses, ranges, dlos, dhis, dmids, Dq_by_win = \
        [], [], [], [], [], [], [], []
    log("\n[windowed area-D(q) drift over q in [%.0f,%.0f] -- the localized signature]"
        % (Q_LO, Q_HI))
    log("  L-window        center   drift(smooth)+-jk   range   D(q~1)->D(q~4)")
    for w in wins:
        r = window_drift(area_seeds, w, q_grid)
        c = float(np.exp(np.mean(np.log(w))))      # geometric-mean L of the window
        centers.append(c); drifts.append(r['drift']); ses.append(r['se'])
        ranges.append(r['drift_range']); dlos.append(r['d_lo']); dhis.append(r['d_hi'])
        dmids.append(r['dmid']); Dq_by_win.append(r['Dq'])
        log("  %-14s  %6.0f   %+.3f +- %.3f    %.3f   %.2f -> %.2f"
            % ("-".join(str(x) for x in w), c, r['drift'], r['se'],
               r['drift_range'], r['d_lo'], r['d_hi']))

    # ---- extrapolate drift vs 1/center_L -> intercept (S17 protocol) ----
    cen = np.array(centers, float)
    dr = np.array(drifts, float)
    sefit, _ = _ols_slope_se(1.0 / cen, dr)
    intercept = dr.mean() - sefit * (1.0 / cen).mean()
    pred = intercept + sefit * (1.0 / cen)
    icpt_se = float(np.sqrt(np.sum((dr - pred) ** 2) / max(1, len(dr) - 2)))

    log("\n[extrapolation: windowed drift vs 1/L_window]")
    log("  L -> infinity intercept (1/L fit) = %+.3f +- %.3f" % (intercept, icpt_se))
    log("  S12 pooled drift over L<=256 (all-L regression, for reference) ~ 0.27")

    # ---- verdict: read the COMPONENTS (drift trend + D_mid trend), not one number.
    log("\n[verdict -- heal (finite-size corona) or true asymptotic multifractal?]")
    healing = (drifts[0] - drifts[-1])
    log("  drift trend small-L -> large-L window: %+.3f -> %+.3f  (change %+.3f)"
        % (drifts[0], drifts[-1], -healing))
    log("  D(mid-q) trend (the footprint dimension): %.2f -> %.2f  (-> 1 = mono-fractal"
        % (dmids[0], dmids[-1]))
    log("    filament, the S17 prediction A ~ L^1)")
    near_zero = abs(intercept) < max(0.05, 2 * icpt_se)
    if near_zero and healing > 0:
        log("  -> HEALS. The windowed drift decreases with L and extrapolates to ~0:")
        log("     the S12 area multifractality is a FINITE-SIZE CORONA. Asymptotically")
        log("     the avalanche area is SIMPLE-FSS / mono-fractal (D_area -> 1), exactly")
        log("     as S17's conditional result (S ~ A^2, S ~ L^2 => A ~ L^1) predicts.")
        log("     The unconditional moments and the conditional exponents AGREE in the")
        log("     L -> infinity limit -- the chapter's single-scale-in-space picture is")
        log("     self-consistent across two independent methods.")
    elif intercept > 0 and not near_zero:
        log("  -> PERSISTS. The windowed drift extrapolates to a nonzero residual %+.3f:"
            % intercept)
        log("     the area multifractality is a TRUE asymptotic property, not a corona.")
        log("     This is in TENSION with S17's conditional gamma(S|A) -> 2 and demands")
        log("     reconciling unconditional moments with conditional exponents (the")
        log("     moments weight the rare largest avalanches the conditional fit does not).")
    else:
        log("  -> PARTIAL. The drift shrinks but to a small nonzero residual %+.3f: most"
            % intercept)
        log("     of the S12 multifractality is finite-size, with a weak true remainder.")

    _make_figure(Ls, area_seeds, q_grid, wins, centers, drifts, ses, dmids,
                 Dq_by_win, intercept, sefit, icpt_se)

    log("\nS18 COMPLETE")
    with open(os.path.join(OUTDIR, "sandpile_area_multifractal.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


def _make_figure(Ls, area_seeds, q_grid, wins, centers, drifts, ses, dmids,
                 Dq_by_win, intercept, sefit, icpt_se):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.3))
    sel = (q_grid >= Q_LO) & (q_grid <= Q_HI)

    # ---- (0) area D(q) per sliding window: does the curve flatten toward D=1? ----
    cmap = plt.cm.viridis(np.linspace(0, 0.85, len(wins)))
    for c, w, Dq in zip(cmap, wins, Dq_by_win):
        ax[0].plot(q_grid[sel], Dq[sel], "-o", color=c, ms=3,
                   label="L~%d (%s)" % (int(centers[wins.index(w)]),
                                        "-".join(str(x) for x in w)))
    ax[0].axhline(1.0, color="0.5", ls="--", lw=1.0, label="mono-fractal filament D=1")
    ax[0].set_xlabel("moment order q")
    ax[0].set_ylabel("local slope  D(q) = d sigma / d q  (area)")
    ax[0].set_title("Area D(q) per sliding L-window\n(flatter + nearer 1 at large L = healing)")
    ax[0].legend(fontsize=7, loc="upper left")

    # ---- (1) the decisive panel: windowed drift vs 1/L_window -> intercept ----
    cen = np.array(centers, float)
    invc = 1.0 / cen
    ax[1].errorbar(invc, drifts, yerr=ses, fmt="o", color="C3", ms=7, capsize=3,
                   label="2-D area D(q) drift")
    xr = np.array([0.0, invc.max() * 1.05])
    ax[1].plot(xr, intercept + sefit * xr, "k--", lw=1.1,
               label="1/L fit -> %+.2f +- %.2f at L=inf" % (intercept, icpt_se))
    ax[1].plot(0, intercept, "ks", ms=8)
    ax[1].axhline(0.0, color="C0", ls=":", lw=1.0, label="simple FSS (no drift)")
    for c, d in zip(cen, drifts):
        ax[1].annotate("L~%d" % int(c), (1.0 / c, d), fontsize=7,
                       textcoords="offset points", xytext=(4, 5))
    ax[1].set_xlabel("1 / L_window")
    ax[1].set_ylabel("area D(q) drift over q in [%.0f,%.0f]" % (Q_LO, Q_HI))
    ax[1].set_title("Does the area multifractality heal as L grows?")
    ax[1].legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    p = os.path.join(FIGDIR, "sandpile_area_multifractal.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    log("\nsaved %s" % os.path.relpath(p, ROOT))


if __name__ == "__main__":
    _self_test()
    main(smoke=("smoke" in sys.argv))
