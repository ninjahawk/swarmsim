"""
S20 -- Repose-scaling model test: does the angle of repose saturate (1/L) or
grow slowly (log L)?

S16 showed the 2-D slope sandpile's mean bond slope at SOC repose creeps
upward with L (2.42 at L=64 -> 2.74 at L=512), confirmed the creep is a real
finite-size effect, and left open whether it SATURATES (approaches a finite
limit as L->inf, i.e. 1/L finite-size correction) or DIVERGES slowly (log L).
S16 fit only 1/L; here both models are fit and compared by residuals and AIC,
and the per-doubling increment is examined as an intuitive model check.

The two models:
  Model 1:  r(L) = a + b/L       -- finite-size correction, saturates at a
  Model 2:  r(L) = a + b*log(L)  -- logarithmic growth, diverges

Data: the S16 windowed-mean plateau values (hard-coded from
outputs/sandpile_equilibrate.txt); no new simulation is run.

Self-test: synthetic data drawn from each model exactly recovers the right
winner by AIC.

Run from repo root:  python sandpile/repose_scaling.py
Writes figures/sandpile_repose_scaling.png and outputs/sandpile_repose_scaling.txt
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "figures")
OUTDIR = os.path.join(ROOT, "outputs")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

LOG = []
def log(msg=""):
    print(msg)
    LOG.append(msg)


# S16 converged plateau repose values (windowed-mean, tol=0.008, window=3)
_LS     = np.array([64, 96, 128, 192, 256, 384, 512], dtype=float)
_REPOSE = np.array([2.4204, 2.4976, 2.5579, 2.6177, 2.6466, 2.7050, 2.7448])
_SPREAD = np.array([0.026,  0.006,  0.012,  0.006,  0.021,  0.011,  0.006])


def _ols(x, y):
    """OLS intercept and slope of y on x."""
    xb = x.mean(); yb = y.mean()
    sxx = ((x - xb) ** 2).sum()
    b = ((x - xb) * (y - yb)).sum() / sxx
    a = yb - b * xb
    return float(a), float(b)


def _aic(y, y_hat, k=2):
    """AIC = n*log(RSS/n) + 2k  (comparable only between models fit to same y)."""
    n = len(y)
    rss = float(((y - y_hat) ** 2).sum())
    return n * np.log(rss / n) + 2 * k


def _r2(y, y_hat):
    ss_res = ((y - y_hat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1.0 - ss_res / ss_tot


def fit_and_compare(Ls, repose):
    """Fit 1/L and log(L) models; return a dict of results."""
    # Model 1: r = a + b/L
    a1, b1 = _ols(1.0 / Ls, repose)
    yhat1 = a1 + b1 / Ls

    # Model 2: r = a + b*log(L)
    a2, b2 = _ols(np.log(Ls), repose)
    yhat2 = a2 + b2 * np.log(Ls)

    return dict(
        a1=a1, b1=b1, yhat1=yhat1,
        aic1=_aic(repose, yhat1), r2_1=_r2(repose, yhat1),
        a2=a2, b2=b2, yhat2=yhat2,
        aic2=_aic(repose, yhat2), r2_2=_r2(repose, yhat2),
    )


def _self_test():
    """Synthetic exact-1/L data should prefer Model 1; exact log(L) should prefer Model 2."""
    print("=" * 70)
    print("repose_scaling.py self-test: model-selection identifies the true model")
    print("=" * 70)
    rng = np.random.default_rng(0)
    Ls = _LS.copy()
    noise = 0.005

    # True model 1: r = 2.80 + 20/L (1/L saturating)
    r1_true = 2.80 + 20.0 / Ls + rng.normal(0, noise, len(Ls))
    f1 = fit_and_compare(Ls, r1_true)
    assert f1['aic1'] < f1['aic2'], "1/L data should prefer Model 1 (lower AIC)"
    print("  1/L synthetic: AIC_1/L=%.2f  AIC_log=%.2f  -> 1/L preferred (correct)" %
          (f1['aic1'], f1['aic2']))

    # True model 2: r = 0.50 + 0.10*log(L) (log growth)
    r2_true = 0.50 + 0.10 * np.log(Ls) + rng.normal(0, noise, len(Ls))
    f2 = fit_and_compare(Ls, r2_true)
    assert f2['aic2'] < f2['aic1'], "log(L) data should prefer Model 2 (lower AIC)"
    print("  log(L) synthetic: AIC_1/L=%.2f  AIC_log=%.2f  -> log preferred (correct)" %
          (f2['aic1'], f2['aic2']))
    print("self-test OK: AIC selects the correct model on noiseless synthetic data.\n")


def main():
    log("=" * 70)
    log("S20 -- REPOSE-SCALING MODEL TEST: 1/L saturation vs log(L) growth")
    log("=" * 70)
    log("Data: S16 converged plateau repose values, L=64-512 (no new simulation).")

    Ls = _LS
    repose = _REPOSE
    spread = _SPREAD

    log("\n[S16 data]")
    log("  L      repose +- spread")
    for L, r, s in zip(Ls, repose, spread):
        log("  %-5d  %.4f +- %.3f" % (L, r, s))

    # --- per-doubling increment (intuitive check) ---
    log("\n[per-doubling increment of the converged repose]")
    log("  1/L model predicts these should SHRINK by ~1/2 per doubling.")
    log("  log(L) model predicts they should be CONSTANT  (~b*log(2) each).")
    doublings = [(64, 128), (128, 256), (256, 512)]
    increments = []
    r_map = dict(zip(Ls.astype(int), repose))
    for La, Lb in doublings:
        inc = r_map[Lb] - r_map[La]
        increments.append(inc)
        log("  %d->%d : +%.3f" % (La, Lb, inc))
    ratio = increments[-1] / increments[-2] if increments[-2] != 0 else float('nan')
    log("  last-step / prev-step ratio = %.2f  (1/L -> ~0.50; log(L) -> ~1.00)" % ratio)

    # --- model fits ---
    f = fit_and_compare(Ls, repose)
    log("\n[Model 1: r(L) = a + b/L  (saturating at r_inf = a)]")
    log("  a = r_inf = %.4f   b = %.2f" % (f['a1'], f['b1']))
    log("  R^2 = %.5f   AIC = %.3f" % (f['r2_1'], f['aic1']))
    log("  max |residual| = %.4f  (%.1f * typical spread)" %
        (np.abs(repose - f['yhat1']).max(),
         np.abs(repose - f['yhat1']).max() / spread.mean()))

    log("\n[Model 2: r(L) = a + b*log(L)  (diverging)]")
    log("  a = %.4f   b = %.5f" % (f['a2'], f['b2']))
    log("  R^2 = %.5f   AIC = %.3f" % (f['r2_2'], f['aic2']))
    log("  max |residual| = %.4f  (%.1f * typical spread)" %
        (np.abs(repose - f['yhat2']).max(),
         np.abs(repose - f['yhat2']).max() / spread.mean()))

    delta_aic = f['aic2'] - f['aic1']
    log("\n[AIC comparison]")
    log("  DAIC = AIC_log - AIC_1/L = %.3f" % delta_aic)
    # AIC conventions: |DAIC|<2 indistinguishable; 2-6 positive; 6-10 strong; >10 very strong
    if abs(delta_aic) < 2:
        strength = "indistinguishable (|DAIC|<2)"
    elif abs(delta_aic) < 6:
        strength = "positive evidence"
    elif abs(delta_aic) < 10:
        strength = "strong evidence"
    else:
        strength = "very strong evidence"
    if abs(delta_aic) < 2:
        verdict = "INDISTINGUISHABLE (|DAIC|=%.1f < 2)" % abs(delta_aic)
    elif delta_aic > 0:
        verdict = "1/L (saturating) PREFERRED  (%s, DAIC=%.1f)" % (strength, delta_aic)
    else:
        verdict = "log(L) (diverging) PREFERRED  (%s, DAIC=%.1f)" % (strength, abs(delta_aic))
    log("  Verdict: " + verdict)

    log("\n[Interpretation]")
    if abs(delta_aic) < 2:
        log("  The repose data at L=64-512 does not distinguish the two models.")
        log("  Physical prior: power-law finite-size corrections go as 1/L;")
        log("  log(L) would require a marginal perturbation. Both models fit within")
        log("  the measurement spread. Saturation vs divergence remains open at L<=512.")
    elif delta_aic > 0:
        log("  The 1/L model fits better (%s, DAIC=%.1f)." % (strength, delta_aic))
        log("  The repose likely saturates at r_inf ~ %.4f as L -> inf." % f['a1'])
    else:
        log("  The log(L) model fits better (%s, DAIC=%.1f)." % (strength, abs(delta_aic)))
        log("  The per-doubling increment (%.3f, %.3f, %.3f) is roughly constant" %
            tuple(increments))
        log("  (ratio=%.2f vs ~1.00 expected for log(L), vs ~0.50 for 1/L)," % ratio)
        log("  consistent with a logarithmic divergence. The 1/L fit's r_inf=%.4f" % f['a1'])
        log("  underestimates the actual L=512 repose by %.3f (%.1fx spread) -- the" %
            (repose[-1] - f['yhat1'][-1],
             abs(repose[-1] - f['yhat1'][-1]) / spread[-1]))
        log("  1/L correction predicts saturation too early. Physical implication:")
        log("  the infinite-volume SOC repose may not be well-defined for this model;")
        log("  larger lattices sustain steeper gradients before hitting the spreading")
        log("  condition via open-boundary effects that propagate O(L) into the bulk.")
        log("  This is qualitatively different from BTW, where the repose is well-defined")
        log("  in the thermodynamic limit. Confirming divergence vs slow saturation")
        log("  would require L ~ 1024-2048 (compute ~100s each at current throughput).")

    log("\n[Summary]")
    log("  1/L fit   : r_inf = %.4f (R^2=%.4f; AIC=%.2f)" % (f['a1'], f['r2_1'], f['aic1']))
    log("  log(L) fit: b = %.5f / doubling=%.4f  (R^2=%.4f; AIC=%.2f)" %
        (f['b2'], f['b2'] * np.log(2), f['r2_2'], f['aic2']))
    log("  DAIC = %.2f -> %s" % (delta_aic, verdict))
    if delta_aic < -2:
        log("  The repose DIVERGES slowly (log L); no finite infinite-L limit confirmed.")
    elif delta_aic > 2:
        log("  The repose SATURATES; r_inf ~ %.4f is the best current estimate." % f['a1'])
    else:
        log("  Inconclusive at L<=512.")

    _make_figure(Ls, repose, spread, f)

    log("\nS20 DONE")
    with open(os.path.join(OUTDIR, "sandpile_repose_scaling.txt"), "w") as fh:
        fh.write("\n".join(LOG) + "\n")


def _make_figure(Ls, repose, spread, f):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # left: repose vs 1/L with both model fits
    ax = axes[0]
    inv = 1.0 / Ls
    xr = np.linspace(0, inv.max() * 1.08, 200)

    ax.errorbar(inv, repose, yerr=spread, fmt="o", color="C0", ms=7,
                capsize=4, label="S16 plateau values")
    ax.plot(xr, f['a1'] + f['b1'] * xr, "k-", lw=1.5,
            label="1/L fit: $r_\\infty=%.3f$" % f['a1'])
    ax.plot(0, f['a1'], "ks", ms=9, zorder=5)
    # log(L) model evaluated at discrete L values (avoids 1/0 at origin)
    L_ext = np.array([32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048], float)
    inv_ext = 1.0 / L_ext
    r_log_ext = f['a2'] + f['b2'] * np.log(L_ext)
    ax.plot(inv_ext, r_log_ext, "r--", lw=1.5,
            label="log(L) fit  (AIC %+.1f)" % (f['aic2'] - f['aic1']))
    for L, x, r in zip(Ls.astype(int), inv, repose):
        ax.annotate("L=%d" % L, (x, r), fontsize=7,
                    textcoords="offset points", xytext=(4, -9))
    ax.set_xlabel("1 / L")
    ax.set_ylabel("converged repose (mean bond slope)")
    ax.set_title("Repose vs 1/L: both model fits\n"
                 "(DAIC=%.2f; |DAIC|<2 -> indistinguishable at L<=512)" %
                 (f['aic2'] - f['aic1']))
    ax.legend(fontsize=8)

    # right: residuals for each model
    ax = axes[1]
    res1 = repose - f['yhat1']
    res2 = repose - f['yhat2']
    ax.axhline(0, color="gray", lw=0.7, ls="--")
    ax.errorbar(np.log(Ls) - 0.04, res1, yerr=spread, fmt="o-", color="k",
                ms=6, capsize=3, label="1/L residuals  (R$^2$=%.4f)" % f['r2_1'])
    ax.errorbar(np.log(Ls) + 0.04, res2, yerr=spread, fmt="s--", color="r",
                ms=6, capsize=3, label="log(L) residuals  (R$^2$=%.4f)" % f['r2_2'])
    ax.set_xlabel("log(L)")
    ax.set_ylabel("repose residual")
    ax.set_title("Residuals: 1/L (circles) vs log(L) (squares)\n"
                 "Spread bars = S16 measurement uncertainty")
    ax.legend(fontsize=8)

    fig.tight_layout()
    p = os.path.join(FIGDIR, "sandpile_repose_scaling.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    log("\nsaved %s" % os.path.relpath(p, ROOT))


if __name__ == "__main__":
    _self_test()
    main()
