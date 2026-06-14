"""
Validation of the 1-D slope-based sandpile against the predictable outcomes in
Charbonneau (2017), Chapter 5. This is the "check limiting cases / known
results" step the professor asks for, done before any new exploration.

Checks, in order:
  1. Global mass balance (eq 5.6 + 5.8): total grains added by forcing minus
     total sand drained at the open boundary equals the final pile mass, to
     machine precision. Confirms the redistribution is conservative and the
     boundary is the only sink.
  2. Mass time series from an empty N=100 pile (reproduces figure 5.4A): linear
     transient growth, then a statistically stationary plateau; plus the
     intermittent displaced-mass series (figure 5.4B).
  3. Angle of repose: the stationary mean slope sits a few percent BELOW Zc
     (the book reports ~7% for eps=0.1), because stochastic forcing tips some
     pairs over before the whole pile reaches Zc.
  4. E-T avalanche correlation (reproduces figure 5.6): all avalanches fall in a
     wedge bounded by slope +1 (line avalanches, E ~ T) and +2 (wedge
     avalanches, E ~ T^2) in a log-log plot.
  5. Avalanche-energy PDF for several lattice sizes (reproduces figure 5.7A):
     a power law whose logarithmic slope is independent of N (book: ~ -1.09).
  6. Initial-condition independence (Exercise 3): an empty pile, a triangular
     pile at repose, and a uniformly-loaded pile converge to the same stationary
     slope and the same energy-PDF slope -- the SOC state is an attractor.

Run from the repository root:  python sandpile/validate1d.py
Writes figures into figures/ and a text summary to outputs/sandpile_validate.txt.

ASCII-only output (Windows cp1252 safe).
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sandpile1d import (run_sandpile, measure_avalanches, triangle_ic,
                        angle_of_repose)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "figures")
OUTDIR = os.path.join(ROOT, "outputs")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

LOG = []
def log(msg):
    print(msg)
    LOG.append(msg)


def stationary_mask(n_iter, frac=0.5):
    """Boolean index selecting the back fraction of a run (the stationary part)."""
    m = np.zeros(n_iter, dtype=bool)
    m[int(n_iter * (1 - frac)):] = True
    return m


def logbin_pdf(x, nbins=40, xmin=None, xmax=None):
    """Logarithmically-binned probability density of positive data x.
    Returns (centers, density) over non-empty bins."""
    x = np.asarray(x)
    x = x[x > 0]
    if x.size == 0:
        return np.array([]), np.array([])
    xmin = x.min() if xmin is None else xmin
    xmax = x.max() if xmax is None else xmax
    edges = np.logspace(np.log10(xmin), np.log10(xmax * 1.0001), nbins + 1)
    counts, _ = np.histogram(x, bins=edges)
    widths = np.diff(edges)
    centers = np.sqrt(edges[:-1] * edges[1:])
    density = counts / (widths * x.size)
    nz = counts > 0
    return centers[nz], density[nz]


def powerlaw_slope(centers, density, lo=None, hi=None):
    """Least-squares slope of log10(density) vs log10(centers) over [lo,hi]."""
    c, d = np.asarray(centers), np.asarray(density)
    if lo is not None:
        keep = c >= lo
        c, d = c[keep], d[keep]
    if hi is not None:
        keep = c <= hi
        c, d = c[keep], d[keep]
    if c.size < 2:
        return np.nan
    A = np.polyfit(np.log10(c), np.log10(d), 1)
    return A[0]


# ---------------------------------------------------------------------------
# Check 1: global mass balance (instrumented run).
# ---------------------------------------------------------------------------
def check_mass_balance():
    log("\n[1] Global mass balance (eq 5.6 + 5.8)")
    N, eps, Zc, n_iter, seed = 60, 0.1, 5.0, 60000, 3
    rng = np.random.default_rng(seed)
    S = np.zeros(N)
    added = 0.0
    drained = 0.0
    for n in range(n_iter):
        d = S[1:] - S[:-1]
        z = np.abs(d)
        u = z >= Zc
        if u.any():
            contrib = np.where(u, d * 0.25, 0.0)
            move = np.zeros(N)
            move[:-1] += contrib
            move[1:] -= contrib
            S += move
        else:
            r = rng.integers(0, N)
            g = rng.uniform(0.0, eps)
            S[r] += g
            added += g
        drained += S[N - 1]      # this much is about to be removed by eq 5.8
        S[N - 1] = 0.0
    residual = added - drained - S.sum()
    log("  grains added   = %.6f" % added)
    log("  boundary drain = %.6f" % drained)
    log("  final mass     = %.6f" % S.sum())
    log("  residual (added - drained - final) = %.3e" % residual)
    assert abs(residual) < 1e-9, "mass not conserved"
    log("  PASS: sand is conserved up to boundary drainage")


# ---------------------------------------------------------------------------
# Check 2 + 3: mass series, intermittency, angle of repose (empty N=100 pile).
# ---------------------------------------------------------------------------
def check_mass_series_and_repose():
    log("\n[2,3] Mass series, intermittency, angle of repose (N=100 from empty)")
    N, eps, Zc = 100, 0.1, 5.0
    n_iter = 2_000_000
    res = run_sandpile(N=N, eps=eps, Zc=Zc, n_iter=n_iter, seed=0)
    mass, disp = res['mass'], res['disp']

    # Angle of repose. The final-snapshot mean slope is one noisy sample; a more
    # robust estimate time-averages the mass over the stationary back-half and
    # converts it to a slope assuming the (observed) triangular profile,
    # mass = slope * N(N-1)/2  =>  slope = 2*mass/(N(N-1)). We report both.
    snap_slope, snap_def = angle_of_repose(res['S'], Zc)
    back = stationary_mask(n_iter, frac=0.5)
    avg_slope = 2.0 * mass[back].mean() / (N * (N - 1))
    avg_def = (Zc - avg_slope) / Zc
    log("  stationary slope (final snapshot)      = %.4f  (Zc = %.1f)" % (snap_slope, Zc))
    log("  stationary slope (time-avg of mass)    = %.4f" % avg_slope)
    log("  fractional deficit (Zc - slope)/Zc     = %.1f%% (snapshot) / %.1f%% (time-avg)"
        % (100 * snap_def, 100 * avg_def))
    log("  NOTE: the book states ~7%% for eps=0.1, but its own fig-5.4A mass (~2.1e4)")
    log("        back-calculates to slope ~4.24 (~15%%). Our deficit is ~15-16%%; the")
    log("        eps-dependence below tests the mechanism (slope -> Zc as eps -> 0).")

    # Figure: mass series (top) + a zoomed displaced-mass burst series (bottom).
    fig, ax = plt.subplots(2, 1, figsize=(8, 7))
    ax[0].plot(np.arange(n_iter) / 1e5, mass / 1e4, lw=0.7, color="k")
    ax[0].set_xlabel("iteration  [10^5]")
    ax[0].set_ylabel("sandpile mass M  [10^4]")
    ax[0].set_title("1-D sandpile mass: transient growth -> stationary plateau "
                    "(cf. Charbonneau fig 5.4A)")
    # Zoom on a stationary window for the intermittent displaced mass.
    z0, z1 = int(0.80 * n_iter), int(0.82 * n_iter)
    ax[1].plot(np.arange(z0, z1) / 1e5, disp[z0:z1], lw=0.6, color="C3")
    ax[1].set_xlabel("iteration  [10^5]")
    ax[1].set_ylabel("displaced mass  dM")
    ax[1].set_title("Intermittent avalanche discharge in the stationary state "
                    "(cf. fig 5.4B)")
    fig.tight_layout()
    p = os.path.join(FIGDIR, "sandpile_validate_mass.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    log("  saved %s" % os.path.relpath(p, ROOT))
    return res


# ---------------------------------------------------------------------------
# Check 4: E-T wedge (N=1000, triangle IC to skip transient).
# ---------------------------------------------------------------------------
def check_ET_wedge():
    log("\n[4] Avalanche E-T correlation wedge (N=1000)")
    N, eps, Zc = 1000, 0.1, 5.0
    n_iter = 5_000_000
    S0 = triangle_ic(N, 0.95 * Zc)
    res = run_sandpile(N=N, eps=eps, Zc=Zc, n_iter=n_iter, seed=1, S0=S0)
    # discard a warmup to settle onto the attractor
    warm = 500_000
    E, P, T = measure_avalanches(res['disp'][warm:])
    log("  avalanches measured = %d" % E.size)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.loglog(T, E, ".", ms=1.5, alpha=0.3, color="k")
    tt = np.array([1, T.max()]) if T.size else np.array([1, 10])
    # bounding lines slope +1 and +2, anchored at the smallest quantum dM0.
    dm0 = E[T == 1].min() if (T == 1).any() else E.min()
    ax.loglog(tt, dm0 * tt, "--", color="C0", label="slope +1  (E ~ T)")
    ax.loglog(tt, dm0 * tt**2, "-.", color="C3", label="slope +2  (E ~ T^2)")
    ax.set_xlabel("avalanche duration  T")
    ax.set_ylabel("avalanche energy  E")
    ax.set_title("E-T wedge, N=1000 (cf. Charbonneau fig 5.6)")
    ax.legend()
    fig.tight_layout()
    p = os.path.join(FIGDIR, "sandpile_validate_ET.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    log("  saved %s" % os.path.relpath(p, ROOT))


# ---------------------------------------------------------------------------
# Check 5: energy/peak PDFs across lattice size (N-independent power-law slope).
# ---------------------------------------------------------------------------
def check_pdfs():
    log("\n[5] Avalanche-energy PDF across lattice size (N-independent slope)")
    eps, Zc = 0.1, 5.0
    runs = [(100, 2_000_000, 300_000), (300, 3_000_000, 500_000),
            (1000, 5_000_000, 700_000)]
    colors = {100: "C3", 300: "C2", 1000: "k"}

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    slopes_E, slopes_P = {}, {}
    for N, n_iter, warm in runs:
        S0 = triangle_ic(N, 0.95 * Zc)
        res = run_sandpile(N=N, eps=eps, Zc=Zc, n_iter=n_iter, seed=2, S0=S0)
        E, P, T = measure_avalanches(res['disp'][warm:])
        cE, dE = logbin_pdf(E)
        cP, dP = logbin_pdf(P)
        # fit the scaling region: above the small-E flattening, below the cutoff.
        sE = powerlaw_slope(cE, dE, lo=cE.min() * 5, hi=cE.max() * 0.2)
        sP = powerlaw_slope(cP, dP, lo=cP.min() * 3, hi=cP.max() * 0.3)
        slopes_E[N], slopes_P[N] = sE, sP
        ax[0].loglog(cE, dE, "-", color=colors[N], label="N=%d (slope %.2f)" % (N, sE))
        ax[1].loglog(cP, dP, "-", color=colors[N], label="N=%d (slope %.2f)" % (N, sP))
        log("  N=%4d : %6d avalanches, PDF(E) slope %.2f, PDF(P) slope %.2f"
            % (N, E.size, sE, sP))

    ax[0].set_xlabel("avalanche energy  E"); ax[0].set_ylabel("PDF(E)")
    ax[0].set_title("Energy PDF (cf. fig 5.7A; book slope ~ -1.09)"); ax[0].legend()
    ax[1].set_xlabel("avalanche peak  P"); ax[1].set_ylabel("PDF(P)")
    ax[1].set_title("Peak PDF (cf. fig 5.7B; book slope ~ -1.12)"); ax[1].legend()
    fig.tight_layout()
    p = os.path.join(FIGDIR, "sandpile_validate_pdf.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    log("  saved %s" % os.path.relpath(p, ROOT))
    log("  PDF(E) slopes by N: %s" % {k: round(v, 2) for k, v in slopes_E.items()})
    log("  -> slope is approximately N-independent if these agree")


# ---------------------------------------------------------------------------
# Check 6: initial-condition independence (Exercise 3).
# ---------------------------------------------------------------------------
def check_eps_sweep():
    """Exercise 4: does the angle-of-repose deficit close as eps -> 0? The book
    claims the stationary slope approaches Zc only in the limit eps -> 0 (small
    grains overshoot the threshold by less). Confirming the trend validates the
    mechanism behind the deficit, independent of the exact percentage."""
    log("\n[4b] Angle-of-repose vs forcing amplitude eps (Exercise 4), N=100")
    N, Zc = 100, 5.0
    n_iter, warm = 2_000_000, 700_000
    log("  eps     slope    deficit%   PDF(E) slope")
    for eps in (0.01, 0.05, 0.1, 0.5, 1.0):
        S0 = triangle_ic(N, 0.90 * Zc)
        res = run_sandpile(N=N, eps=eps, Zc=Zc, n_iter=n_iter, seed=5, S0=S0)
        back = stationary_mask(n_iter, frac=0.5)
        avg_slope = 2.0 * res['mass'][back].mean() / (N * (N - 1))
        deficit = (Zc - avg_slope) / Zc
        E, P, T = measure_avalanches(res['disp'][warm:])
        cE, dE = logbin_pdf(E)
        sE = powerlaw_slope(cE, dE, lo=cE.min() * 5, hi=cE.max() * 0.2)
        log("  %-6.2f  %.4f   %5.1f      %.2f" % (eps, avg_slope, 100 * deficit, sE))
    log("  -> deficit should shrink monotonically toward 0 as eps -> 0")


def check_ic_independence():
    log("\n[6] Initial-condition independence (Exercise 3), N=100")
    # N=100 so an empty pile fully fills well within the run (transient ~Zc*N^2/eps
    # ~ 0.5M iters); at N=300 it would need ~4.5M and stay in transient.
    N, eps, Zc = 100, 0.1, 5.0
    n_iter, warm = 3_000_000, 1_500_000
    ics = {
        "empty": np.zeros(N),
        "triangle@repose": triangle_ic(N, 0.90 * Zc),
        "uniform-load": np.full(N, 2.0),
    }
    for name, S0 in ics.items():
        res = run_sandpile(N=N, eps=eps, Zc=Zc, n_iter=n_iter, seed=4, S0=S0.copy())
        back = stationary_mask(n_iter, frac=0.5)
        avg_slope = 2.0 * res['mass'][back].mean() / (N * (N - 1))
        deficit = (Zc - avg_slope) / Zc
        E, P, T = measure_avalanches(res['disp'][warm:])
        cE, dE = logbin_pdf(E)
        sE = powerlaw_slope(cE, dE, lo=cE.min() * 5, hi=cE.max() * 0.2)
        log("  IC=%-16s slope=%.3f (deficit %.1f%%), PDF(E) slope=%.2f, n_av=%d"
            % (name, avg_slope, 100 * deficit, sE, E.size))
    log("  -> time-avg stationary slope and PDF slope should agree across ICs")


def main():
    log("=" * 70)
    log("1-D SANDPILE VALIDATION  (Charbonneau Ch. 5)")
    log("=" * 70)
    check_mass_balance()
    check_mass_series_and_repose()
    check_eps_sweep()
    check_ET_wedge()
    check_pdfs()
    check_ic_independence()
    log("\nVALIDATION COMPLETE")
    with open(os.path.join(OUTDIR, "sandpile_validate.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
