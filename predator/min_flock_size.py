# min_flock_size.py -- Below what N does collective evasion fail?
#
# Finding 7 (predator.py exp 4) showed dilution: larger flocks expose a smaller
# fraction to a single predator.  But it stopped at N=10 as the smallest size.
# Open question: at what N does the flock fail to form/maintain coherence at
# all under predator pressure?  Two related thresholds:
#
#   (a) Coherence threshold: minimum N at which Phi(no-predator) > 0.9
#   (b) Evasion threshold:   minimum N at which Phi(with-predator) > 0.9
#       AND the predator does not eliminate (interpenetrate) the group.
#
# Method
# ------
# Sweep N in [3, 5, 8, 12, 18, 25, 40, 60, 100] under three conditions:
#   no-predator, naive predator (1), encircling-pair (2 predators, opposite).
# Measure steady-state Phi, time-fraction with predator within prey r0 of any prey
# (capture proxy), and min prey-pred distance.

import os
import numpy as np
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from model import Flock, Predator
from flocking import order_parameter

os.makedirs('figures', exist_ok=True)

N_SEEDS = 8
N_WARMUP = 1000
N_ITER   = 3000
RECORD_EVERY = 25
N_VALS   = [3, 5, 8, 12, 18, 25, 40, 60, 100]


def run(N, predators_factory, seed):
    np.random.seed(seed)
    # Slow-prey regime so the v0=0.05 predators can actually pursue
    # (matches encirclement.py / Findings 5-16).
    flock = Flock(seed=seed, N=N, v0=0.02, ramp=0.1)
    preds = predators_factory()

    phi_ts = []; mind_ts = []; capture_ts = []
    for step in range(N_ITER):
        flock.evolve(predators=preds)
        if step >= N_WARMUP and step % RECORD_EVERY == 0:
            phi_ts.append(flock.phi)
            if preds:
                dists = [p.dist_to_flock(flock) for p in preds]
                mind = min(dists)
                mind_ts.append(mind)
                # capture proxy: predator within prey r0 of any prey
                capture_ts.append(int(mind <= 2 * flock.r0))
    return dict(
        phi=np.mean(phi_ts),
        mind=np.mean(mind_ts) if mind_ts else float('nan'),
        capture_frac=np.mean(capture_ts) if capture_ts else 0.0,
    )


conds = {
    'none':     lambda: [],
    'naive':    lambda: [Predator(strategy='naive')],
    'encircle2':lambda: [Predator(strategy='encircle', angle=0,   enc_radius=0.15),
                        Predator(strategy='encircle', angle=180, enc_radius=0.15)],
}

results = {c: {} for c in conds}

for N in N_VALS:
    print('N=%d' % N)
    for c, factory in conds.items():
        phi_runs = []; mind_runs = []; cap_runs = []
        for s in range(N_SEEDS):
            r = run(N, factory, s)
            phi_runs.append(r['phi'])
            if not np.isnan(r['mind']):
                mind_runs.append(r['mind'])
            cap_runs.append(r['capture_frac'])
        results[c][N] = dict(
            phi=np.mean(phi_runs), phi_std=np.std(phi_runs),
            mind=np.mean(mind_runs) if mind_runs else float('nan'),
            cap=np.mean(cap_runs),
        )
        print('  %-10s Phi=%.3f +/- %.3f  mind=%.4f  cap_frac=%.2f' %
              (c, results[c][N]['phi'], results[c][N]['phi_std'],
               results[c][N]['mind'], results[c][N]['cap']))


# Plot
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Minimum viable flock size (%d seeds each)' % N_SEEDS, fontsize=11)

ax = axes[0]
colors = {'none': 'steelblue', 'naive': 'darkorange', 'encircle2': 'crimson'}
for c in conds:
    phis = [results[c][N]['phi']     for N in N_VALS]
    stds = [results[c][N]['phi_std'] for N in N_VALS]
    ax.errorbar(N_VALS, phis, yerr=stds, fmt='o-', color=colors[c],
                lw=2, capsize=4, label=c)
ax.set_xscale('log'); ax.set_xlabel('N (log)'); ax.set_ylabel('Steady-state Phi')
ax.set_ylim(0, 1.05); ax.legend(fontsize=9)
ax.set_title('Coherence vs flock size')
ax.axhline(0.9, ls=':', color='k', alpha=0.4)
ax.grid(alpha=0.3)

ax = axes[1]
for c in ('naive', 'encircle2'):
    minds = [results[c][N]['mind'] for N in N_VALS]
    ax.plot(N_VALS, minds, 'o-', color=colors[c], lw=2, label=c)
ax.set_xscale('log'); ax.set_xlabel('N (log)'); ax.set_ylabel('Mean min pred-prey distance')
ax.legend(fontsize=9)
ax.set_title('Evasion distance')
ax.grid(alpha=0.3)

ax = axes[2]
for c in ('naive', 'encircle2'):
    caps = [results[c][N]['cap'] for N in N_VALS]
    ax.plot(N_VALS, caps, 'o-', color=colors[c], lw=2, label=c)
ax.set_xscale('log'); ax.set_xlabel('N (log)'); ax.set_ylabel('Time fraction predator within 2*r0 of a prey')
ax.legend(fontsize=9)
ax.set_title('Capture frequency')
ax.set_ylim(-0.05, 1.05)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/min_size_1_summary.png', dpi=120)
plt.close()
print('  --> figures/min_size_1_summary.png')

print('\nMin-flock-size analysis complete.')
