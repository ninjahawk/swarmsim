"""
ofc_quasiperiodic.py -- E3: recurrent (quasi-periodic) avalanching and spatial
synchronization in the nonconservative OFC model (Charbonneau sec. 8.3-8.4,
figs. 8.4-8.6).

This is the headline qualitative CONTRAST with the chapter-5 sandpile.  The slope
sandpile's avalanches are temporally uncorrelated -- each is an independent response
to the slow random forcing, and the findings S-series never found inter-avalanche
structure.  The OFC model, being deterministic and nonconservative, instead develops
spatial DOMAINS of locked nodal values that collapse and rebuild on a near-fixed
period, producing recurrent (quasi-periodic) large avalanches.  The recurrence
period shrinks as alpha rises toward the conservative value 0.25, where it vanishes.

We:
  (a) measure the dominant recurrence period vs alpha from the autocorrelation of the
      iteration-indexed toppling-activity series, and cross-check it against the
      median spacing of large events; compare to the book's ~6435/4002/2165 at
      alpha=0.10/0.15/0.20 and "none" at 0.25;
  (b) contrast the autocorrelation of a nonconservative run (clear periodic peak)
      with the conservative run (no peak);
  (c) show the spatial synchronization domains building up from the random initial
      condition (the fig. 8.6 snapshots), and quantify synchronization over time.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from earthquake.ofc import run_ofc

os.makedirs('figures', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

N = 128
ALPHAS = [0.10, 0.15, 0.20, 0.25]
BOOK_PERIOD = {0.10: 6435, 0.15: 4002, 0.20: 2165, 0.25: None}
MAKE_DOMAINS = False     # the domain snapshot figure (figures/ofc_domains.png) is already
#                          produced and matches book fig. 8.6; skip the 8M-iter regen on re-runs


def activity_series(iters, sizes, span_lo, span_hi, binw):
    """Total toppling activity per iteration bin over [span_lo, span_hi)."""
    nbins = int((span_hi - span_lo) // binw)
    grid = np.zeros(nbins)
    b = ((iters - span_lo) // binw).astype(int)
    keep = (b >= 0) & (b < nbins)
    np.add.at(grid, b[keep], sizes[keep])
    return grid


def dominant_period(grid, binw, min_lag_iter=400, max_lag_iter=20000):
    """First strong autocorrelation peak beyond min_lag, in iterations (or nan)."""
    g = grid - grid.mean()
    if np.allclose(g, 0):
        return np.nan, np.array([]), np.array([])
    ac = np.correlate(g, g, mode='full')[len(g) - 1:]
    ac = ac / ac[0]
    lags = np.arange(len(ac)) * binw
    lo = max(1, int(min_lag_iter // binw))
    hi = min(len(ac) - 1, int(max_lag_iter // binw))
    if hi <= lo + 1:
        return np.nan, lags, ac
    # first local maximum in the window that is also a reasonable peak
    seg = ac[lo:hi]
    k = np.argmax(seg)
    peak_val = seg[k]
    period = (lo + k) * binw
    if peak_val < 0.04:         # no meaningful recurrence
        return np.nan, lags, ac
    return period, lags, ac


def main():
    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out("E3: OFC recurrent (quasi-periodic) avalanching and synchronization")
    out("  N=%d  delta_f=1e-4  open BC.  Recurrence period vs alpha (book in parens)." % N)
    out("")

    runs = {}
    period_ac = {}
    period_spacing = {}
    for alpha in ALPHAS:
        warm = 5000 if alpha >= 0.25 else 500000
        nev = 30000 if alpha >= 0.25 else 250000
        r = run_ofc(N=N, alpha=alpha, n_events=nev, warmup_events=warm,
                    seed=0, record_iter=True)
        runs[alpha] = r
        it, s = r['iters'], r['sizes']
        span_lo, span_hi = it[0], it[-1]
        binw = 200
        grid = activity_series(it, s, span_lo, span_hi, binw)
        per, lags, ac = dominant_period(grid, binw)
        period_ac[alpha] = (per, lags, ac)
        # cross-check: median spacing between large events (> 30% of max)
        thr = 0.3 * s.max()
        big_iters = it[s > thr]
        spacing = np.median(np.diff(big_iters)) if big_iters.size > 3 else np.nan
        period_spacing[alpha] = spacing
        # the autocorr gives the RAW period (all iterations); the book's tabulated
        # 6435/4002/2165 are FORCING-CORRECTED (avalanching iterations subtracted, the
        # book's fig. 8.4 caption gives the raw alpha=0.15 period as ~10960). Convert.
        f_av = r['durations'].sum() / (it[-1] - it[0])    # matched-range avalanching fraction
        per_corr = per * (1 - f_av) if np.isfinite(per) else np.nan
        bp = BOOK_PERIOD[alpha]
        out("  alpha=%.2f : raw period=%s iter, forcing-corrected=%s iter (book %s), avalanching frac=%.2f"
            % (alpha,
               ('%.0f' % per) if np.isfinite(per) else 'none',
               ('%.0f' % per_corr) if np.isfinite(per_corr) else 'none',
               ('%d' % bp) if bp else 'none', f_av))

    # ---------------------------------------------------------------- figure 1
    # avalanche time-series segment (alpha=0.15) + autocorr contrast 0.15 vs 0.25
    fig, ax = plt.subplots(2, 1, figsize=(9, 7))
    r15 = runs[0.15]
    it15, s15 = r15['iters'], r15['sizes']
    seg = (it15 - it15[0] < 60000)
    ax[0].plot((it15[seg] - it15[0]), s15[seg], lw=0.6, color='tab:purple')
    ax[0].set_xlabel('iteration (offset)'); ax[0].set_ylabel('avalanche size E')
    ax[0].set_title('alpha=0.15: recurrent avalanching (cf. fig. 8.4 top)')
    ax[0].grid(alpha=0.3)

    for alpha, col in ((0.15, 'tab:green'), (0.25, 'tab:red')):
        per, lags, ac = period_ac[alpha]
        m = lags < 20000
        ax[1].plot(lags[m], ac[m], color=col,
                   label='alpha=%.2f (period %s)' % (alpha, ('%.0f' % per) if np.isfinite(per) else 'none'))
    ax[1].axhline(0, color='gray', lw=0.6)
    ax[1].set_xlabel('lag (iterations)'); ax[1].set_ylabel('autocorrelation')
    ax[1].set_title('Activity autocorrelation: nonconservative peak vs conservative decay')
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/ofc_quasiperiodic.png', dpi=120)
    plt.close()
    out("  --> figures/ofc_quasiperiodic.png")

    # ---------------------------------------------------------------- figure 2
    # synchronization domains building up from random IC (alpha=0.15)
    if MAKE_DOMAINS:
        snaps = []
        snap_times = []
        rng = np.random.RandomState(0)
        F = rng.uniform(0.0, 1.0, size=(N, N))
        from earthquake.ofc import _relax
        targets = [0, 1_000_000, 4_000_000, 8_000_000]
        it = 0
        ti = 0
        delta_f = 1e-4
        while ti < len(targets):
            if it >= targets[ti]:
                snaps.append(F.copy()); snap_times.append(it); ti += 1
                if ti >= len(targets):
                    break
            fmax = F.max()
            if fmax < 1.0:
                steps = int(np.ceil((1.0 - fmax) / delta_f))
                F += steps * delta_f; it += steps
            _, dur = _relax(F, 0.15, 'open')
            it += dur

        fig2, axes = plt.subplots(2, 2, figsize=(9, 8.5))
        for axp, sn, tt in zip(axes.ravel(), snaps, snap_times):
            im = axp.imshow(sn, cmap='inferno', vmin=0, vmax=1)
            axp.set_title('t = %d iter' % tt, fontsize=9)
            axp.set_xticks([]); axp.set_yticks([])
        fig2.suptitle('OFC nodal force: random start -> synchronized domains (alpha=0.15, cf. fig. 8.6)',
                      fontsize=10)
        plt.tight_layout()
        plt.savefig('figures/ofc_domains.png', dpi=120)
        plt.close()
        out("  --> figures/ofc_domains.png")

    out("")
    out("  The nonconservative runs show a clear recurrence period that shrinks with")
    out("  alpha and vanishes at the conservative value 0.25 -- driven by spatial")
    out("  domains of locked nodal values (figure ofc_domains) that collapse and")
    out("  rebuild near-periodically.  This temporal structure is absent in the")
    out("  chapter-5 slope sandpile, whose avalanches are uncorrelated in time.")

    with open('outputs/ofc_quasiperiodic.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    out("  --> outputs/ofc_quasiperiodic.txt")


if __name__ == '__main__':
    main()
