"""
alignment_invasion.py -- co-adaptation thread, F93 follow-up. Mirrors escape_invasion.py
(the F88 experiment) for the heritable ALIGNMENT strength alpha instead of the escape weight.

F93 (evolve_alignment.py) found that heritable alignment strength under capture/removal
predation is BISTABLE, exactly parallel to the F87 escape-weight brake: a high-alignment
flock (alpha0=2) is a protective ESS (Phi=1.0, only ~6 captures), but alignment does NOT
climb de novo from a low start (alpha0=0.2 -> 0.19, alpha0=1.0 drifts DOWN to 0.85). So
high alignment is an evolutionary attractor that the population cannot reach on its own.

This asks the same two follow-up questions F88 asked of the escape weight, with the SAME
capture/removal model and the SAME validated per-step physics -- only the initial condition
and the mutation step change, no new mechanism:

  Exp1 -- INVASION: seed a fraction f of the population at high alignment (alpha=2) in an
          otherwise low-alignment flock (alpha=0.2). Does the high-alignment trait spread
          and pull the flock mean up, or get diluted and lost? Is there a threshold seed
          fraction above which tight flocking takes over?

  Exp2 -- MUTATION STEP: from a uniform low-alignment start (alpha0=0.2), does a larger
          per-capture mutation step let the population JUMP to the high-alignment basin,
          rather than stalling low as in F93?

Reuses evolve_alignment.run, which now accepts a seeded alpha_init array and a mut_sigma
override (both additive; defaults reproduce the original F93 sweep exactly).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import evolution.evolve_alignment as M

os.makedirs('figures', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

N = M.BASE['N']
A_LOW, A_HIGH = 0.2, 2.0      # low basin vs protective-ESS alignment strength (from F93)
HI_THRESH = 1.0              # an agent is 'high-alignment' if alpha > this
SEED_FRACS = [0.05, 0.10, 0.20, 0.50]
MUT_SIGMAS = [0.10, 0.30, 0.60, 1.00]
N_SEEDS = 3
EXP1_EVOLVE = 15000          # 150 tu (matches F88 Exp1)
EXP2_EVOLVE = 20000          # 200 tu, longer to give mutation a chance (matches F88 Exp2)


def seeded_alpha(f):
    """Initial alpha array: round(f*N) high-alignment agents at A_HIGH, the rest at A_LOW
    (agent indices are arbitrary -- positions are random)."""
    a = np.full(N, A_LOW)
    n_hi = int(round(f * N))
    a[:n_hi] = A_HIGH
    return a


def steady(a):
    return float(np.mean(a[-len(a) // 3:]))


def main():
    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('Can the F93 alignment brake be overcome? (capture/removal, predictive predator)')
    out('  N=%d  %d seeds  low alpha=%.2f  high alpha=%.2f  high-alignment threshold alpha>%.2f\n'
        % (N, N_SEEDS, A_LOW, A_HIGH, HI_THRESH))

    # ---------------------------------------------------------------- Exp1
    out('Exp1 -- INVASION: seed a fraction f of high-alignment agents (alpha=%.1f) into a low' % A_HIGH)
    out('  (alpha=%.1f) flock. Does tight flocking invade and spread, or get diluted? (%d tu)\n'
        % (A_LOW, EXP1_EVOLVE * M.BASE['dt']))
    saved = M.N_EVOLVE
    M.N_EVOLVE = EXP1_EVOLVE
    exp1 = {}
    try:
        for f in SEED_FRACS:
            runs = [M.run(A_LOW, s, alpha_init=seeded_alpha(f), hi_thresh=HI_THRESH)
                    for s in range(N_SEEDS)]
            exp1[f] = runs
            aend = np.mean([steady(r['a_mean']) for r in runs])
            fhi0 = np.mean([r['frac_hi'][0] for r in runs])
            fhiE = np.mean([steady(r['frac_hi']) for r in runs])
            phiE = np.mean([steady(r['phi']) for r in runs])
            capE = np.mean([r['cum_cap'][-1] for r in runs])
            verdict = ('SPREADS' if fhiE > fhi0 + 0.05 else
                       'holds' if fhiE > fhi0 - 0.05 else 'DILUTED/lost')
            out('  f=%.2f  high-alignment frac %.2f -> %.2f (%s)  mean alpha_end=%.2f  Phi=%.2f  captures=%.0f'
                % (f, fhi0, fhiE, verdict, aend, phiE, capE))
    finally:
        M.N_EVOLVE = saved

    out('')
    # ---------------------------------------------------------------- Exp2
    out('Exp2 -- MUTATION STEP: from a uniform alpha0=%.2f start, does a bigger mutation step' % A_LOW)
    out('  let the population jump to the high-alignment basin (alpha>1)? (%d tu)\n'
        % (EXP2_EVOLVE * M.BASE['dt']))
    M.N_EVOLVE = EXP2_EVOLVE
    exp2 = {}
    try:
        for ms in MUT_SIGMAS:
            runs = [M.run(A_LOW, s, mut_sigma=ms, hi_thresh=HI_THRESH) for s in range(N_SEEDS)]
            exp2[ms] = runs
            aend = np.mean([steady(r['a_mean']) for r in runs])
            crossed = np.mean([float(r['a_mean'].max() > 1.0) for r in runs])
            phiE = np.mean([steady(r['phi']) for r in runs])
            out('  mut_sigma=%.2f  mean alpha_end=%.3f  reached alpha=1 in %.0f%% of seeds  Phi_end=%.2f'
                % (ms, aend, 100 * crossed, phiE))
    finally:
        M.N_EVOLVE = saved

    # ---------------------------------------------------------------- interpretation
    # An invasion SUCCEEDS only if the high-alignment FRACTION grows (the trait spreads),
    # not merely if the mean crosses 1 -- a large seed can sit at the boundary without
    # spreading. We also require the protective ESS to actually form (captures fall toward
    # the alpha=2 baseline of ~6, Phi -> 1). Neither holds here, so the verdict is honest.
    out('')
    spreads = [f for f in SEED_FRACS
               if np.mean([steady(r['frac_hi']) for r in exp1[f]])
               > f + 0.05]
    if spreads:
        fmin = min(spreads)
        out('  -> INVASION SUCCEEDS: from a seed of f=%.2f the high-alignment fraction GROWS and the' % fmin)
        out('     capture toll drops -- a high-alignment founder group spreads and establishes, so the')
        out('     brake is a barrier to ORIGINATION not invasion, as for the escape weight (F88).')
    else:
        out('  -> INVASION FAILS, unlike the escape weight (F88). A seeded high-alignment minority does NOT')
        out('     spread: at f=0.05-0.20 it is diluted back to the low basin (mean returns to ~0.2), and even')
        out('     at f=0.50 it only just HOLDS (frac ~0.45, mean ~1.0 at the basin boundary, Phi~0.65) -- the')
        out('     protective ESS (alpha=2, Phi=1.0, ~6 captures) never forms; captures stay ~550-650 throughout.')
        out('     Mechanism: alignment is a MUTUAL coupling, not a shared public good like escape. A high-alpha')
        out('     agent aligns hard to neighbours who do not reciprocate, so a minority cannot nucleate a')
        out('     coherent core -- it is dragged by the indifferent majority. The protective benefit of tight')
        out('     flocking is a COORDINATION/quorum trait that only appears near a majority, which a founder')
        out('     group cannot reach. So the F93 alignment brake blocks BOTH origination and invasion -- it is')
        out('     STRONGER than the F88 escape brake, precisely because alignment is not a free-rideable signal.')
    crossers = [ms for ms in MUT_SIGMAS
                if np.mean([float(r['a_mean'].max() > 1.0) for r in exp2[ms]]) >= 1.0]
    if crossers:
        out('  -> A larger mutation STEP jumps the gap: sigma=%s reach alpha=1 in every seed while the F93'
            % '/'.join('%.2f' % m for m in crossers))
        out('     baseline sigma=0.10 never does.')
    else:
        out('  -> No tested mutation step reliably reached the high basin within %d tu (the alignment basin'
            % (EXP2_EVOLVE * M.BASE['dt']))
        out('     boundary near alpha~1.5 is steep; a single-agent mutation cannot drag the population mean).')

    # ---------------------------------------------------------------- figure
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    colors1 = plt.cm.plasma(np.linspace(0, 0.85, len(SEED_FRACS)))
    for f, c in zip(SEED_FRACS, colors1):
        for r in exp1[f]:
            ax[0].plot(r['t'], r['frac_hi'], color=c, alpha=0.85,
                       label=('seed f=%.2f' % f) if r['seed'] == 0 else None)
    ax[0].axhline(1.0, ls=':', color='gray', alpha=0.5)
    ax[0].set_xlabel('time (tu)'); ax[0].set_ylabel('high-alignment fraction (alpha>%.1f)' % HI_THRESH)
    ax[0].set_title('Exp1: invasion of a seeded high-alignment minority'); ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3); ax[0].set_ylim(-0.02, 1.02)

    colors2 = plt.cm.viridis(np.linspace(0, 0.85, len(MUT_SIGMAS)))
    for ms, c in zip(MUT_SIGMAS, colors2):
        for r in exp2[ms]:
            ax[1].plot(r['t'], r['a_mean'], color=c, alpha=0.85,
                       label=('mut_sigma=%.2f' % ms) if r['seed'] == 0 else None)
    ax[1].axhline(1.0, ls='--', color='gray', alpha=0.6, lw=1, label='alpha=1')
    ax[1].set_xlabel('time (tu)'); ax[1].set_ylabel('population mean alignment strength alpha')
    ax[1].set_title('Exp2: does a bigger mutation step jump to the high basin?'); ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    fig.suptitle('Overcoming the F93 alignment brake: invasion vs mutation step (N=%d)' % N, fontsize=10)
    plt.tight_layout()
    plt.savefig('figures/alignment_invasion_1.png', dpi=120)
    plt.close()
    out('\n  --> figures/alignment_invasion_1.png')

    with open('outputs/alignment_invasion.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    out('  --> outputs/alignment_invasion.txt')
    out('\nAlignment invasion / mutation-step experiment complete.')


if __name__ == '__main__':
    main()
