"""
ofc_fss.py -- E2: finite-size scaling of OFC avalanches, and the conservation test
(Charbonneau Exercise 2; the quantitative payoff of sandpile findings S5/S7).

S5 found that bulk conservation is NECESSARY for true self-organized criticality in
the slope sandpile: with conservation the avalanche cutoff grows as a power of system
size (no characteristic scale), but any bulk dissipation truncates the distribution
at a fixed, dissipation-set size independent of L.  OFC is the canonical model in
which conservation is a continuous knob (alpha), so it lets us test that claim
directly and quantitatively.

For each alpha we sweep the lattice side L and measure a cutoff proxy for the
avalanche-size distribution -- the moment ratio <E^2>/<E>, which scales as the upper
cutoff for a power law with exponent < 2 -- and ask whether it GROWS with L (a true
critical system, no intrinsic scale) or SATURATES (a subcritical system with a
finite correlation length below L).  The prediction from S5: the conservative case
alpha=0.25 should scale as a power of L (cutoff ~ L^D with D ~ 2, system-spanning),
while strongly dissipative alpha should saturate to an L-independent characteristic
size.  Whatever the intermediate alpha does is the genuinely open question (OFC's
criticality for alpha < 0.25 is "perennially debated").
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from earthquake.ofc import run_ofc

os.makedirs('figures', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

L_VALUES = [16, 24, 32, 48, 64, 96]
ALPHAS = [0.10, 0.20, 0.25]
COLORS = {0.10: 'tab:blue', 0.20: 'tab:green', 0.25: 'tab:red'}


def cutoff_proxy(sizes):
    """<E^2>/<E>, the standard moment-ratio estimate of the distribution cutoff."""
    s = sizes.astype(float)
    return (s ** 2).mean() / s.mean()


def main():
    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out("E2: OFC finite-size scaling and the conservation test (Exercise 2; cf. S5/S7)")
    out("  cutoff proxy = <E^2>/<E>; does it grow with L (critical) or saturate (subcritical)?")
    out("")

    data = {}            # alpha -> (Ls, cutoffs, means, maxes)
    for alpha in ALPHAS:
        Ls, cuts, means, maxes = [], [], [], []
        for L in L_VALUES:
            if alpha >= 0.25:
                warm, nev = 5000, 15000
            else:
                warm, nev = max(20000, 25 * L * L), 150000
            r = run_ofc(N=L, alpha=alpha, n_events=nev, warmup_events=warm, seed=0)
            s = r['sizes']
            co = cutoff_proxy(s)
            Ls.append(L); cuts.append(co); means.append(s.mean()); maxes.append(s.max())
            out("  alpha=%.2f L=%3d : <E>=%8.1f  cutoff<E2>/<E>=%9.1f  maxE=%8d"
                % (alpha, L, s.mean(), co, s.max()))
        data[alpha] = (np.array(Ls, float), np.array(cuts), np.array(means), np.array(maxes))
        out("")

    # fit cutoff ~ L^D for each alpha (log-log slope)
    out("  cutoff scaling exponent D (cutoff ~ L^D):")
    fits = {}
    for alpha in ALPHAS:
        Ls, cuts, _, _ = data[alpha]
        D = np.polyfit(np.log10(Ls), np.log10(cuts), 1)[0]
        fits[alpha] = D
        verdict = ("CRITICAL-like (grows ~L^2)" if D > 1.5 else
                   "intermediate" if D > 0.7 else "SATURATING (subcritical)")
        out("    alpha=%.2f : D=%.2f  -> %s" % (alpha, D, verdict))

    # ---------------------------------------------------------------- figure
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for alpha in ALPHAS:
        Ls, cuts, means, _ = data[alpha]
        ax[0].loglog(Ls, cuts, 'o-', color=COLORS[alpha],
                     label='alpha=%.2f  D=%.2f' % (alpha, fits[alpha]))
    # L^2 guide
    Lg = np.array([L_VALUES[0], L_VALUES[-1]], float)
    c0 = data[0.25][1][0]
    ax[0].loglog(Lg, c0 * (Lg / Lg[0]) ** 2, 'k--', lw=1, alpha=0.5, label='slope 2 (L^2)')
    ax[0].set_xlabel('lattice side L'); ax[0].set_ylabel('cutoff  <E^2>/<E>')
    ax[0].set_title('Avalanche cutoff vs system size')
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which='both')

    for alpha in ALPHAS:
        Ls, _, means, _ = data[alpha]
        ax[1].loglog(Ls, means, 's-', color=COLORS[alpha], label='alpha=%.2f' % alpha)
    ax[1].set_xlabel('lattice side L'); ax[1].set_ylabel('mean avalanche size <E>')
    ax[1].set_title('Mean avalanche size vs system size')
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, which='both')
    plt.tight_layout()
    plt.savefig('figures/ofc_fss.png', dpi=120)
    plt.close()
    out("")
    out("  --> figures/ofc_fss.png")

    out("")
    out("  Interpretation: a cutoff that grows as ~L^2 means avalanches are limited only")
    out("  by the system (true criticality); a saturating cutoff means a dissipation-set")
    out("  characteristic size below L (subcritical), exactly the S5 signature transferred")
    out("  to the canonical earthquake model.")

    with open('outputs/ofc_fss.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    out("  --> outputs/ofc_fss.txt")


if __name__ == '__main__':
    main()
