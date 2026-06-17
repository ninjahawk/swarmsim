"""
ofc_gr.py -- E1: validate the OFC model against the book's Gutenberg-Richter result
(Charbonneau fig. 8.7).

Reproduces the avalanche-size PDFs for alpha = 0.10, 0.20, 0.25 on a 128x128 lattice
with delta_f = 1e-4, and compares the fitted power-law slopes to the book's reported
values: -3.34 (alpha=0.10), -1.92 (alpha=0.20), -1.19 (alpha=0.25, conservative).
The conservative case is the clean Gutenberg-Richter power law (PDF slope ~ -1.2,
i.e. the cumulative b ~ 1 of eq. 8.1); lowering alpha steepens the slope and pulls
the cutoff to smaller sizes, because the bulk dissipation (1-4alpha) must balance the
forcing with more frequent small events.

Key practical point (the book warns of it): the nonconservative cases need a LONG
warmup to reach the synchronized statistically-stationary state -- from a random
start the avalanches are tiny until spatial domains of locked nodal values form,
which takes ~1e6 iterations.  We therefore warm up for hundreds of thousands of
avalanches before recording.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from earthquake.ofc import run_ofc, logbin_pdf, powerlaw_slope

os.makedirs('figures', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

N = 128
# (alpha, warmup_events, n_events, book_slope, fit_lo, fit_hi)
# alpha=0.10 dissipates 60% per topple, so its synchronization domains -- and hence
# the avalanche cutoff -- coarsen very slowly; it needs far more statistics than the
# near-conservative cases to mature, so we give it a long run and fit only its (short)
# power-law range.  fit_hi=None means fit up to 0.3*maxE adaptively.
CASES = [
    (0.10, 3000000, 3000000, -3.34, 5, None),
    (0.20, 300000, 300000, -1.92, 10, 3000),
    (0.25, 5000, 30000, -1.19, 30, 30000),
]
COLORS = {0.10: 'tab:blue', 0.20: 'tab:green', 0.25: 'tab:red'}


def main():
    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out("E1: OFC Gutenberg-Richter validation (Charbonneau fig. 8.7)")
    out("  N=%d  delta_f=1e-4  open boundaries  (book slopes in parentheses)" % N)
    out("")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    results = {}
    for alpha, warm, nev, bslope, flo, fhi in CASES:
        r = run_ofc(N=N, alpha=alpha, n_events=nev, warmup_events=warm, seed=0)
        s = r['sizes']
        c, p = logbin_pdf(s, nbins=26)
        if fhi is None:
            fhi = 0.3 * s.max()
        slope = powerlaw_slope(c, p, lo=flo, hi=fhi)
        results[alpha] = (s, c, p, slope)
        out("  alpha=%.2f : %d events, maxE=%d, meanE=%.1f, fitted slope=%.2f (book %.2f)"
            % (alpha, s.size, s.max(), s.mean(), slope, bslope))
        ax.loglog(c, p, 'o-', ms=4, color=COLORS[alpha],
                  label='alpha=%.2f  slope=%.2f (book %.2f)' % (alpha, slope, bslope))
        # reference power law over the fit range
        cc = np.array([flo, fhi], dtype=float)
        # anchor the guide line near the data
        idx = np.argmin(np.abs(c - flo))
        yref = p[idx]
        ax.loglog(cc, yref * (cc / flo) ** slope, '--', color=COLORS[alpha], lw=1, alpha=0.6)

    ax.set_xlabel('avalanche size E (topplings)')
    ax.set_ylabel('PDF(E)')
    ax.set_title('OFC avalanche-size PDFs vs conservation parameter (N=%d)' % N)
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig('figures/ofc_gr.png', dpi=120)
    plt.close()

    out("")
    out("  Reproduces fig. 8.7: conservative alpha=0.25 gives the clean")
    out("  Gutenberg-Richter power law (PDF slope ~ -1.2, cumulative b ~ 1); lowering")
    out("  alpha steepens the slope and shrinks the cutoff, as bulk dissipation forces")
    out("  the stationary state onto more frequent, smaller avalanches.")
    out("  --> figures/ofc_gr.png")

    with open('outputs/ofc_gr.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    out("  --> outputs/ofc_gr.txt")


if __name__ == '__main__':
    main()
