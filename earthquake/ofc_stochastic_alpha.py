"""
ofc_stochastic_alpha.py -- E6 (self-test): does a mildly stochastic conservation
parameter break the OFC quasi-periodicity? (Charbonneau Exercise 4.)

The recurrent avalanching of the nonconservative OFC model (E3) rests on spatial
domains of EXACTLY equal nodal values that stay synchronized because the deterministic
forcing and the fixed-alpha redistribution preserve equality.  Exercise 4 asks whether
drawing a fresh alpha uniformly in [0.14, 0.16] at every toppling node -- so that
redistribution no longer maps equal neighbours to equal neighbours -- is enough to
destroy that synchronization and wash out the periodicity.

Prediction (pre-registered, in the self-test tradition of the flocking F47/F81 and the
sandpile S6 checks): synchronization requires equality to be an exact fixed point of
the update, so even a small per-topple jitter in alpha should erode the domains and
suppress the recurrence peak in the avalanche-activity autocorrelation.  We compare
fixed alpha=0.15 against alpha drawn in [0.14, 0.16] and report whether the
autocorrelation periodic peak survives.
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
BINW = 200


def activity_autocorr(iters, sizes, binw=BINW):
    lo, hi = iters[0], iters[-1]
    nb = int((hi - lo) // binw)
    grid = np.zeros(nb)
    b = ((iters - lo) // binw).astype(int)
    keep = (b >= 0) & (b < nb)
    np.add.at(grid, b[keep], sizes[keep])
    g = grid - grid.mean()
    ac = np.correlate(g, g, mode='full')[len(g) - 1:]
    ac = ac / ac[0]
    return np.arange(len(ac)) * binw, ac


def peak_strength(lags, ac, lo_iter=800, hi_iter=20000):
    m = (lags >= lo_iter) & (lags <= hi_iter)
    if m.sum() < 2:
        return np.nan, np.nan
    k = np.argmax(ac[m])
    return lags[m][k], ac[m][k]


def main():
    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out("E6 (self-test): does stochastic alpha in [0.14,0.16] break OFC quasi-periodicity?")
    out("  N=%d  delta_f=1e-4.  Compare fixed alpha=0.15 to per-topple random alpha." % N)
    out("  Prediction: synchronization needs exact equality, so jitter should suppress the peak.")
    out("")

    fig, ax = plt.subplots(figsize=(8, 5))
    results = {}
    for label, kw, col in (("fixed alpha=0.15", dict(alpha=0.15), 'tab:green'),
                           ("alpha~U[0.14,0.16]", dict(alpha=0.15, alpha_noise=(0.14, 0.16)), 'tab:orange')):
        r = run_ofc(N=N, n_events=200000, warmup_events=300000, seed=0,
                    record_iter=True, **kw)
        lags, ac = activity_autocorr(r['iters'], r['sizes'])
        per, strength = peak_strength(lags, ac)
        results[label] = (per, strength)
        out("  %-20s : recurrence peak at %.0f iter, autocorr strength=%.3f, maxE=%d"
            % (label, per, strength, r['sizes'].max()))
        m = lags < 20000
        ax.plot(lags[m], ac[m], color=col, label='%s (peak %.2f)' % (label, strength))

    ax.axhline(0, color='gray', lw=0.6)
    ax.set_xlabel('lag (iterations)'); ax.set_ylabel('activity autocorrelation')
    ax.set_title('Does stochastic alpha break OFC quasi-periodicity? (N=%d)' % N)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/ofc_stochastic_alpha.png', dpi=120)
    plt.close()

    s_fixed = results["fixed alpha=0.15"][1]
    s_noisy = results["alpha~U[0.14,0.16]"][1]
    out("")
    if np.isfinite(s_fixed) and np.isfinite(s_noisy):
        ratio = s_noisy / s_fixed if s_fixed else np.nan
        if s_noisy < 0.5 * s_fixed:
            out("  -> CONFIRMED: the recurrence peak collapses under stochastic alpha")
            out("     (strength %.3f -> %.3f, %.0f%% of the fixed-alpha peak). Synchronization"
                % (s_fixed, s_noisy, 100 * ratio))
            out("     requires exact equality of neighbouring nodal values, which per-topple alpha")
            out("     jitter destroys; the avalanching loses its periodic component.")
        else:
            out("  -> NOT CONFIRMED: the peak survives the alpha jitter (strength %.3f -> %.3f)."
                % (s_fixed, s_noisy))
            out("     Synchronization is more robust to redistribution noise than predicted;")
            out("     recorded honestly as a falsified self-test prediction.")
    out("  --> figures/ofc_stochastic_alpha.png")

    with open('outputs/ofc_stochastic_alpha.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    out("  --> outputs/ofc_stochastic_alpha.txt")


if __name__ == '__main__':
    main()
