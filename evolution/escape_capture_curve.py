"""
escape_capture_curve.py -- co-adaptation thread, F91 candidate. Diagnostic for F90.

F90 found (by EVOLUTION) that the capture-selected predator lead converges to ~3 tu,
not the F66 most-disruptive value of ~2 tu, and inferred that capture-maximisation
and coherence-disruption are distinct objectives with distinct optima. This pins
that down by DIRECT MEASUREMENT rather than inference: against frozen no-escape prey,
sweep a FIXED predator lead and measure, on the same axis, (a) the capture rate and
(b) the order parameter (coherence). If the capture rate peaks near lead ~3 while
coherence is lowest (most disrupted) near lead ~2, the F90 inference is confirmed and
the geometry explained -- the lead that most fragments the flock is not the lead that
catches the most prey, because fragmentation scatters the prey out of reach.

No evolution here: predators are held at a fixed common lead (evolve_pred=False) and
the prey escape weight is frozen at zero (evolve_prey=False); only the steady capture
rate and coherence are read off. Pure characterisation of the same capture/removal
model used in F87-F90.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import evolution.escape_coevolution as C

os.makedirs('figures', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

LEADS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
N_SEEDS = 3
MEASURE_STEPS = 6000        # 60 tu of measurement at each fixed lead


def main():
    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('Capture rate and coherence vs FIXED predator lead (frozen no-escape prey) -- F90 diagnostic')
    out('  N=%d  n_pred=%d  %d seeds  %d tu measurement per lead'
        % (C.BASE['N'], C.N_PRED, N_SEEDS, MEASURE_STEPS * C.BASE['dt']))
    out('  F66 found lead~2 the most DISRUPTIVE (lowest Phi) by hand; F90 evolved lead~3 by CAPTURE selection.')
    out('  Question: does the CAPTURE rate peak at a different lead than the coherence minimum?\n')

    saved = C.N_CO
    C.N_CO = MEASURE_STEPS
    res = {}
    try:
        for L in LEADS:
            runs = [C.run_coevolution(0.0, s, evolve_prey=False, evolve_pred=False, fixed_lead=L)
                    for s in range(N_SEEDS)]
            # steady capture rate (captures per tu) and mean coherence over the run
            rate = np.mean([r['cum_cap'][-1] / (MEASURE_STEPS * C.BASE['dt']) for r in runs])
            rate_sd = np.std([r['cum_cap'][-1] / (MEASURE_STEPS * C.BASE['dt']) for r in runs])
            phi = np.mean([r['phi'][len(r['phi'])//4:].mean() for r in runs])
            res[L] = (rate, rate_sd, phi)
            out('  lead=%.1f tu   capture rate=%6.2f /tu (+/-%.2f)   mean Phi=%.3f'
                % (L, rate, rate_sd, phi))

    finally:
        C.N_CO = saved

    rates = np.array([res[L][0] for L in LEADS])
    phis = np.array([res[L][2] for L in LEADS])
    lead_cap_opt = LEADS[int(np.argmax(rates))]
    lead_phi_min = LEADS[int(np.argmin(phis))]
    out('')
    out('  -> Capture rate PEAKS at lead=%.1f tu; coherence is LOWEST (most disrupted) at lead=%.1f tu.'
        % (lead_cap_opt, lead_phi_min))
    if lead_cap_opt > lead_phi_min:
        out('     The capture optimum lies at a LONGER lead than the disruption optimum -- confirming F90 by')
        out('     direct measurement: maximally fragmenting the flock (short lead) scatters prey out of reach,')
        out('     while a longer lead places predators on the flock\'s path and catches more. Capture-maximisation')
        out('     and coherence-disruption are genuinely distinct objectives with distinct optima.')
    else:
        out('     The two optima coincide or invert here -- F90\'s capture/disruption distinction is not')
        out('     reproduced by this direct sweep; revisit the evolved-lead interpretation.')

    # ---------------------------------------------------------------- figure
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ll = np.array(LEADS)
    errs = np.array([res[L][1] for L in LEADS])
    ax1.errorbar(ll, rates, yerr=errs, marker='o', lw=2, color='crimson', capsize=4,
                 label='capture rate (/tu)')
    ax1.set_xlabel('fixed predator lead time (tu)')
    ax1.set_ylabel('capture rate (/tu)', color='crimson')
    ax1.tick_params(axis='y', labelcolor='crimson')
    ax1.axvline(lead_cap_opt, ls=':', color='crimson', alpha=0.5)
    ax2 = ax1.twinx()
    ax2.plot(ll, phis, marker='s', lw=2, color='navy', label='order parameter Phi')
    ax2.set_ylabel('order parameter Phi (coherence)', color='navy')
    ax2.tick_params(axis='y', labelcolor='navy')
    ax2.axvline(lead_phi_min, ls=':', color='navy', alpha=0.5)
    ax1.axvline(2.0, ls='--', color='gray', alpha=0.5)
    ax1.set_title('Capture rate vs coherence vs fixed lead\n(capture optimum ~%.0f, disruption optimum ~%.0f)'
                  % (lead_cap_opt, lead_phi_min), fontsize=10)
    fig.tight_layout()
    plt.savefig('figures/escape_capture_curve_1.png', dpi=120)
    plt.close()
    out('\n  --> figures/escape_capture_curve_1.png')
    with open('outputs/escape_capture_curve.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    out('  --> outputs/escape_capture_curve.txt')
    out('\nCapture-curve diagnostic complete.')


if __name__ == '__main__':
    main()
