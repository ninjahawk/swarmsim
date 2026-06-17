"""
ofc_et.py -- E4: avalanche size-duration relationship in the OFC model
(Charbonneau Exercise 5).

For alpha = 0.15, 0.20, 0.25 we record both the avalanche size E (total topplings)
and duration T (number of synchronous relaxation sweeps), and ask what statistical
relation E(T) holds.  In a compact spreading avalanche on a 2-D lattice one expects
E to grow faster than linearly in T -- each sweep the active front is roughly a ring
whose radius grows with T, so the cumulative toppled area scales like a power of T.
We bin E by T and fit a power law E ~ T^gamma.

This parallels the sandpile duration analysis (findings S3/S10), where the
size-duration scaling relation FAILED in 1-D (quantized line/wedge avalanches) and a
clean duration exponent was only recovered in 2-D with the fast engine.  Here both E
and T are defined the same way for OFC across all alpha, so the comparison across the
conservation parameter is clean.
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
ALPHAS = [0.15, 0.20, 0.25]
COLORS = {0.15: 'tab:purple', 0.20: 'tab:green', 0.25: 'tab:red'}


def binned_ET(T, E, nbins=18):
    """Mean E in log-spaced T bins (only bins with >= 20 events)."""
    T = T.astype(float); E = E.astype(float)
    tmax = T.max()
    edges = np.unique(np.floor(np.logspace(0, np.log10(tmax + 1), nbins + 1)).astype(int)).astype(float)
    centers, meanE = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (T >= a) & (T < b)
        if m.sum() >= 20:
            centers.append(np.sqrt(a * b)); meanE.append(E[m].mean())
    return np.array(centers), np.array(meanE)


def main():
    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out("E4: OFC avalanche size-duration relation E(T) (Exercise 5)")
    out("  E = total topplings, T = relaxation sweeps; fit E ~ T^gamma.")
    out("")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for alpha in ALPHAS:
        warm = 5000 if alpha >= 0.25 else 300000
        nev = 30000 if alpha >= 0.25 else 250000
        r = run_ofc(N=N, alpha=alpha, n_events=nev, warmup_events=warm, seed=0)
        E, T = r['sizes'], r['durations']
        c, m = binned_ET(T, E)
        # fit over the well-populated middle of the range
        fitm = (c >= 3)
        gamma = np.polyfit(np.log10(c[fitm]), np.log10(m[fitm]), 1)[0] if fitm.sum() >= 2 else np.nan
        out("  alpha=%.2f : maxT=%d  maxE=%d  E ~ T^%.2f"
            % (alpha, T.max(), E.max(), gamma))
        ax.loglog(c, m, 'o-', color=COLORS[alpha], ms=4,
                  label='alpha=%.2f  E~T^%.2f' % (alpha, gamma))
    ax.set_xlabel('duration T (sweeps)'); ax.set_ylabel('mean size <E | T>')
    ax.set_title('OFC avalanche size vs duration (N=%d)' % N)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig('figures/ofc_et.png', dpi=120)
    plt.close()
    out("")
    out("  A power-law E ~ T^gamma with gamma > 1 reflects a compact spreading front:")
    out("  the toppled area accumulates faster than the avalanche's lifetime.")
    out("  --> figures/ofc_et.png")

    with open('outputs/ofc_et.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    out("  --> outputs/ofc_et.txt")


if __name__ == '__main__':
    main()
