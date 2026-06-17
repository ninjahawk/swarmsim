"""
Moment-scaling machinery for the sandpile universality thread (S11-S12).

Single avalanche exponents (tau) are a blunt instrument: S4 separated the 2-D
slope model from canonical BTW on tau_S = 0.89 vs 1.14, but had to hedge it
(bond-vs-site counting, modest L). The field's decisive discriminator is instead
how the WHOLE family of moments <x^q> scales with system size L. For a
distribution obeying simple finite-size scaling (gap scaling),

    P(x) = x^{-tau} G(x / x_c),   x_c ~ L^D,

the moment exponent sigma(q) defined by  <x^q> ~ L^{sigma(q)}  is

    sigma(q) = D * (q + 1 - tau)     for q > tau - 1,

i.e. LINEAR in q with a single slope D. A model that obeys simple FSS therefore
has a CONSTANT local slope  D(q) = d sigma/dq = D. A model whose D(q) DRIFTS with
q is multifractal / anomalous (a single tau is then ill-defined). This is exactly
the BTW-vs-Manna distinction (De Menech-Stella-Tebaldi 1998; Tebaldi et al 1999),
and the tool this module provides.

Nothing here touches the physics; it operates on per-avalanche observable arrays
(energy E, toppling-number S, ...) produced by the validated engines via
fss2d.measure_multi. ASCII-only output.

Self-test (run this file): synthesize avalanches from an EXACT simple-FSS
distribution (power law x^{-tau} with a hard cutoff x_c = A*L^D) at several L and
confirm the machinery (a) recovers the analytic sigma(q) = D(q+1-tau) line and
(b) reports a FLAT D(q). That guards against the method inventing multifractality
where there is none -- the precondition for trusting a "multifractal" verdict on
the slope model in S12.
"""

import numpy as np


def avalanche_moments(x, q_grid):
    """Raw moments <x^q> of a per-avalanche observable array x, over a grid of q.

    x<=0 entries are dropped (a non-toppling iteration carries no avalanche).
    float64 is sufficient here: the largest avalanches (S ~ 1e5, E ~ 1e6) raised
    to the highest q used (~6) stay far below the float64 overflow ceiling, and
    numpy longdouble is no wider than float64 on Windows anyway.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[x > 0.0]
    q_grid = np.asarray(q_grid, dtype=np.float64)
    if x.size == 0:
        return np.full(q_grid.shape, np.nan)
    logx = np.log(x)
    # <x^q> = mean(exp(q*logx)); computing via logs keeps the powering vectorized
    # and avoids forming a huge (n_aval x n_q) array when n_aval is large.
    out = np.empty(q_grid.shape, dtype=np.float64)
    for i, q in enumerate(q_grid):
        out[i] = np.exp(q * logx).mean()
    return out


def _ols_slope_se(xv, yv):
    """Least-squares slope of y on x, plus the standard error of that slope."""
    xv = np.asarray(xv, dtype=np.float64)
    yv = np.asarray(yv, dtype=np.float64)
    n = xv.size
    xbar = xv.mean()
    sxx = ((xv - xbar) ** 2).sum()
    slope = ((xv - xbar) * (yv - yv.mean())).sum() / sxx
    intercept = yv.mean() - slope * xbar
    resid = yv - (intercept + slope * xv)
    if n > 2:
        s2 = (resid ** 2).sum() / (n - 2)
        se = np.sqrt(s2 / sxx)
    else:
        se = np.nan
    return slope, se


def sigma_of_q(moments_by_L, Ls, q_grid):
    """For each q, regress log10<x^q> on log10 L across lattice sizes Ls to get
    sigma(q) (the moment-scaling exponent) and its regression standard error.

    moments_by_L : dict L -> array of <x^q> aligned with q_grid (from
                   avalanche_moments).
    Returns (sigma, sigma_se), both arrays aligned with q_grid.
    """
    Ls = np.asarray(Ls, dtype=np.float64)
    logL = np.log10(Ls)
    nq = len(q_grid)
    sigma = np.empty(nq)
    sigma_se = np.empty(nq)
    for i in range(nq):
        y = np.log10(np.array([moments_by_L[L][i] for L in Ls]))
        sigma[i], sigma_se[i] = _ols_slope_se(logL, y)
    return sigma, sigma_se


def local_slope(q_grid, sigma):
    """D(q) = d sigma / d q by central finite difference. Constant -> simple FSS;
    drifting with q -> multifractal. This is the headline diagnostic."""
    q_grid = np.asarray(q_grid, dtype=np.float64)
    return np.gradient(np.asarray(sigma, dtype=np.float64), q_grid)


def step_slope(q_grid, sigma):
    """Unit-step moment-difference sigma(q+dq)-sigma(q) over the grid spacing,
    the form De Menech et al. read multifractality from. Returns (q_mid, slope)."""
    q = np.asarray(q_grid, dtype=np.float64)
    s = np.asarray(sigma, dtype=np.float64)
    dq = np.diff(q)
    return 0.5 * (q[:-1] + q[1:]), np.diff(s) / dq


def bootstrap_sigma(samples_by_L, Ls, q_grid, n_boot=200, seed=0):
    """Resample avalanches (with replacement) within each lattice size to put
    honest error bars on sigma(q) and D(q). High-q moments are dominated by a few
    largest avalanches, so the across-bootstrap spread is the meaningful
    uncertainty, not the per-fit regression SE.

    samples_by_L : dict L -> per-avalanche observable array.
    Returns dict with sigma_mean, sigma_std, Dq_mean, Dq_std (all aligned with
    q_grid), computed over n_boot resamples.
    """
    rng = np.random.default_rng(seed)
    Ls = list(Ls)
    q_grid = np.asarray(q_grid, dtype=np.float64)
    sig_boot = np.empty((n_boot, len(q_grid)))
    Dq_boot = np.empty((n_boot, len(q_grid)))
    for b in range(n_boot):
        mom = {}
        for L in Ls:
            x = samples_by_L[L]
            idx = rng.integers(0, x.size, x.size)
            mom[L] = avalanche_moments(x[idx], q_grid)
        sig, _ = sigma_of_q(mom, Ls, q_grid)
        sig_boot[b] = sig
        Dq_boot[b] = local_slope(q_grid, sig)
    return dict(
        sigma_mean=sig_boot.mean(0), sigma_std=sig_boot.std(0),
        Dq_mean=Dq_boot.mean(0), Dq_std=Dq_boot.std(0),
    )


# ---------------------------------------------------------------------------
# Self-test: an exact simple-FSS source must read out as flat D(q).
# ---------------------------------------------------------------------------
def _sample_fss(n, tau, x_min, x_c, rng):
    """Draw n samples from P(x) ~ x^{-tau} on [x_min, x_c] (hard upper cutoff).
    Valid for any tau != 1, including tau < 1 (our model's regime). Inverse-CDF:
        x = [ x_min^{1-tau} + u*(x_c^{1-tau} - x_min^{1-tau}) ]^{1/(1-tau)}.
    This is the simple-FSS form with G a step function and x_c the cutoff."""
    a = 1.0 - tau
    u = rng.random(n)
    lo = x_min ** a
    hi = x_c ** a
    return (lo + u * (hi - lo)) ** (1.0 / a)


def _self_test():
    print("=" * 68)
    print("moments.py self-test: exact simple-FSS source -> flat D(q)?")
    print("=" * 68)
    rng = np.random.default_rng(1)
    q_grid = np.arange(0.5, 5.01, 0.25)

    # Two regimes: a BTW-like tau>1 and a slope-model-like tau<1. Both should
    # give sigma(q) = D(q+1-tau) (linear) and D(q) = D (flat), since simple FSS
    # holds by construction regardless of tau.
    for tau, D in ((1.20, 2.50), (0.90, 2.20)):
        Ls = [64, 128, 256, 512]
        n_per = 400_000
        x_min = 1.0
        A = 1.0  # x_c = A * L^D
        samples = {L: _sample_fss(n_per, tau, x_min, A * L ** D, rng) for L in Ls}
        mom = {L: avalanche_moments(samples[L], q_grid) for L in Ls}
        sigma, _ = sigma_of_q(mom, Ls, q_grid)
        Dq = local_slope(q_grid, sigma)
        # The simple-FSS moment exponent depends on which cutoff carries the
        # normalization. For 1<tau<2 the lower cutoff x_min dominates and (for
        # q>tau-1) sigma(q)=D(q+1-tau). For tau<1 (our slope model's regime) the
        # UPPER cutoff x_c dominates the normalization, giving sigma(q)=D*q
        # (consistent with sigma(0)=0, since <x^0>=1 for every L). In BOTH cases
        # the local slope D(q)=d sigma/dq is the constant D -- that flatness, not
        # the intercept, is the FSS signature the analysis relies on.
        if tau > 1.0:
            sigma_analytic = D * (q_grid + 1.0 - tau)
        else:
            sigma_analytic = D * q_grid

        max_sigma_err = np.abs(sigma - sigma_analytic).max()
        Dq_mid = Dq[2:-2]  # ignore finite-difference edge effects
        Dq_spread = Dq_mid.max() - Dq_mid.min()
        print("\n  tau=%.2f  D=%.2f" % (tau, D))
        print("    max |sigma_fit - D(q+1-tau)| = %.3f   (expect small)" % max_sigma_err)
        print("    D(q) over q in [%.2f,%.2f]: min=%.3f max=%.3f spread=%.3f  (expect ~0)"
              % (q_grid[2], q_grid[-3], Dq_mid.min(), Dq_mid.max(), Dq_spread))
        # tolerances: sampling noise at n=4e5 keeps both well under these
        assert max_sigma_err < 0.15, "FSS sigma(q) not recovered"
        assert Dq_spread < 0.15, "FSS D(q) not flat -- method invents multifractality"
        print("    PASS: simple-FSS source reads out as flat D(q) ~ %.2f" % Dq_mid.mean())

    print("\nself-test OK: the machinery does not manufacture multifractality.")


if __name__ == "__main__":
    _self_test()
