"""
escape_energy.py -- co-adaptation thread, F92 candidate. Robustness of the F87
brake under a DIFFERENT fitness model (energy budget), the student's chosen test.

F87-F91 used capture/removal selection, under which a high escape weight is nearly
free once established (escape avoids the predator and costs nothing else). The
energy-budget model bakes in the metabolic cost that F70 says CREATES the valley:
fitness = survival benefit MINUS a cost proportional to the escape weight. Here
that is an always-on per-step death hazard metab_cost * w (escaping is expensive
even when safe), on top of the predator-capture hazard; dead agents are replaced by
mutated clones of survivors exactly as before. So a high weight is no longer free --
it pays a constant metabolic price -- which should (a) reinforce the brake (the
valley is now strictly costlier to climb) and (b) replace the "high w is free"
plateau of F87 with an INTERIOR optimal weight balancing escape benefit against
metabolic cost.

Exp1 (fixed cost, sweep initial weight): does the brake still trap low starts, and
are high starts pulled DOWN toward an interior optimum rather than staying at 2?
Exp2 (seeded escape, sweep cost): how does the evolved steady weight (the ESS) fall
as the metabolic cost rises, and is there a cost above which escape stops paying?
Both reuse escape_evolution.run_evolution (the F87 loop) with metab_cost > 0.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import evolution.escape_evolution as E

os.makedirs('figures', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

N = E.BASE['N']
W0_VALUES = [0.0, 0.5, 1.0, 2.0]
COSTS = [0.0, 0.25, 0.5, 1.0, 2.0]
FIXED_COST = 0.5
N_SEEDS = 2
ESC_THRESH = 0.75
EVOLVE = 15000          # 150 tu


def seeded_w(f):
    w = np.zeros(N)
    w[:int(round(f * N))] = 2.0
    return w


def steady(a, frac=3):
    return float(np.mean(a[-len(a)//frac:]))


def main():
    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('Energy-budget fitness model: does the F87 brake survive an explicit metabolic cost of escape?')
    out('  N=%d  %d seeds  %d tu  death hazard = predator-capture + metab_cost * w per tu\n'
        % (N, N_SEEDS, EVOLVE * E.BASE['dt']))

    saved = E.N_EVOLVE
    E.N_EVOLVE = EVOLVE
    try:
        out('Exp1 -- fixed metabolic cost c=%.2f, sweep initial weight w0:' % FIXED_COST)
        out('  (does the brake still trap low starts; are high starts pulled DOWN to an interior optimum?)')
        exp1 = {}
        for w0 in W0_VALUES:
            runs = [E.run_evolution(w0, s, metab_cost=FIXED_COST, esc_thresh=ESC_THRESH)
                    for s in range(N_SEEDS)]
            exp1[w0] = runs
            wS = np.mean([steady(r['w_mean']) for r in runs])
            fE = np.mean([steady(r['frac_esc']) for r in runs])
            phi = np.mean([steady(r['phi']) for r in runs])
            out('    w0=%.2f -> steady mean w=%.2f (escaper frac %.2f)  Phi=%.2f' % (w0, wS, fE, phi))

        out('')
        out('Exp2 -- seeded f=0.5 escapers, sweep metabolic cost c (the ESS vs cost):')
        exp2 = {}
        for c in COSTS:
            runs = [E.run_evolution(0.0, s, w_init=seeded_w(0.5), metab_cost=c, esc_thresh=ESC_THRESH)
                    for s in range(N_SEEDS)]
            exp2[c] = runs
            wS = np.mean([steady(r['w_mean']) for r in runs])
            fE = np.mean([steady(r['frac_esc']) for r in runs])
            phi = np.mean([steady(r['phi']) for r in runs])
            out('    cost c=%.2f -> ESS mean w=%.2f (escaper frac %.2f)  Phi=%.2f' % (c, wS, fE, phi))
    finally:
        E.N_EVOLVE = saved

    # interpretation
    out('')
    w_low = np.mean([steady(r['w_mean']) for r in exp1[0.0]])
    w_high = np.mean([steady(r['w_mean']) for r in exp1[2.0]])
    brake = w_low < 0.6
    pulled_down = w_high < 1.7
    if brake:
        out('  -> The brake SURVIVES the energy-budget model: from no escape the weight stays low (w~%.2f),'
            % w_low)
        out('     so escape still cannot originate de novo -- the F87 brake is not an artifact of capture/removal.')
    else:
        out('  -> From no escape the weight reached w~%.2f under this model -- the brake does NOT clearly hold.'
            % w_low)
    if w_high < 0.5:
        out('  -> The F87 "high w is free / stable once present" result does NOT survive: even a w0=2 start')
        out('     COLLAPSES to w~%.2f -- escape is abandoned entirely, not balanced at an interior optimum.'
            % w_high)
    elif w_high < 1.7:
        out('  -> A w0=2 start is pulled DOWN to an interior optimum w~%.2f (escape benefit balances cost).'
            % w_high)
    else:
        out('  -> A high start stays high (w~%.2f); the metabolic cost c=%.2f is too weak to matter.'
            % (w_high, FIXED_COST))
    ess = [np.mean([steady(r['w_mean']) for r in exp2[c]]) for c in COSTS]
    out('  -> ESS vs cost (seeded): ' + ', '.join('c=%.2f:w=%.2f' % (c, e) for c, e in zip(COSTS, ess)))
    if ess[0] > 1.0 and ess[1] < 0.5:
        out('     SHARP threshold: escape persists only at near-zero cost (c=0 -> w=%.2f) and collapses at the'
            % ess[0])
        out('     first appreciable cost (c=%.2f -> w=%.2f). Nearly all-or-nothing, NOT a graded interior optimum.'
            % (COSTS[1], ess[1]))
        out('     Mechanism: the metabolic cost is paid by every escaper continuously, while predation threatens')
        out('     only the few agents near a predator at any instant, so a modest per-capita cost outweighs the')
        out('     diffuse benefit. Under an energy budget, collective escape is viable only when essentially free.')

    # ---------------------------------------------------------------- figure
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(W0_VALUES)))
    for w0, cc in zip(W0_VALUES, colors):
        for r in exp1[w0]:
            ax[0].plot(r['t'], r['w_mean'], color=cc, alpha=0.85,
                       label='w0=%.2f' % w0 if r['seed'] == 0 else None)
    ax[0].axhspan(0.0, 0.5, color='red', alpha=0.07)
    ax[0].axhline(1.0, ls=':', color='green', alpha=0.6, label='w=alpha')
    ax[0].set_xlabel('time (tu)'); ax[0].set_ylabel('population mean escape weight w')
    ax[0].set_title('Exp1: energy budget (cost c=%.2f), sweep w0\n(brake holds; high start pulled down)'
                    % FIXED_COST)
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    ax[1].plot(COSTS, ess, 'o-', lw=2, color='darkorange')
    ax[1].axhline(1.0, ls=':', color='green', alpha=0.6, label='w=alpha (escape works)')
    ax[1].set_xlabel('metabolic cost c'); ax[1].set_ylabel('ESS mean escape weight (seeded start)')
    ax[1].set_title('Exp2: evolved steady weight falls with metabolic cost')
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3); ax[1].set_ylim(0, 2.1)

    fig.suptitle('Energy-budget fitness: robustness of the escape-evolution brake (N=%d)' % N, fontsize=10)
    plt.tight_layout()
    plt.savefig('figures/escape_energy_1.png', dpi=120)
    plt.close()
    out('\n  --> figures/escape_energy_1.png')
    with open('outputs/escape_energy.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    out('  --> outputs/escape_energy.txt')
    out('\nEnergy-budget robustness experiment complete.')


if __name__ == '__main__':
    main()
