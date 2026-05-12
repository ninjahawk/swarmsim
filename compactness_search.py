"""
compactness_search.py -- Search for a true phase transition at intermediate compactness.

Background (Finding 12)
-----------------------
The repulsion-only model was tested at C=0.78 (caged, each agent oscillates like an
independent harmonic oscillator) and C=0.10 (dilute, agents barely interact, behave
like independent random walkers).  Both regimes gave N-independent KE/N and a
monotonically rising susceptibility chi -- no critical point.

Professor's question
--------------------
Is there an intermediate compactness where agents can form a solid AND cooperatively
rearrange, producing a true diverging susceptibility?

This script
-----------
Scans C = 0.15, 0.20, 0.30, 0.40, 0.50, 0.60 with finite-size scaling at
N = 25, 50, 100, 200.  At each (C, N) we sweep noise amplitude eta and compute:
  - KE/N: should collapse across N if no phase transition
  - chi = N * Var(KE/N): should peak at finite eta and diverge with N if critical

A genuine phase transition signature: chi(eta) has a peak that grows with N and
shifts toward a finite eta_c as N -> inf.

Uses model.py (Flock class with alpha=0, v0=0 for repulsion-only dynamics).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from model import Flock

os.makedirs('figures', exist_ok=True)

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------
N_SEEDS     = 8
N_VALS      = [25, 50, 100, 200]
ETA_VALS    = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
C_VALS      = [0.15, 0.20, 0.30, 0.40, 0.50, 0.60]
N_ITER      = 3000
WARMUP_FRAC = 0.8   # discard first 80% as transient

colors = ['steelblue', 'firebrick', 'seagreen', 'darkorange']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ke_stats(ke_tail, N):
    """Return mean KE/N and variance of KE/N from tail samples."""
    ke_n = np.array(ke_tail) / N
    return float(ke_n.mean()), float(ke_n.var())


def run_one(N, eta, C, seed):
    """Single seed: repulsion-only flock, return KE timeseries from steady state."""
    r0 = np.sqrt(C / (np.pi * N))
    # alpha=0: no flocking force; v0=0, mu>0: gives damping back to zero speed
    flock = Flock(N=N, r0=r0, eps=25.0, rf=0.1,
                  alpha=0.0, v0=0.0, mu=10.0, ramp=eta, dt=0.01, seed=seed)
    ke_series = []
    warmup = int(N_ITER * WARMUP_FRAC)
    for i in range(N_ITER):
        flock.evolve()
        if i >= warmup:
            ke_series.append(flock.kinetic_energy)
    return ke_series


def sweep_C(C):
    """Run the full (N, eta) grid for one compactness value C."""
    print(f'\n=== C = {C:.2f} ===')
    results = {}
    for N in N_VALS:
        r0 = np.sqrt(C / (np.pi * N))
        print(f'  N={N:3d}  r0={r0:.5f}', flush=True)
        ke_means = []
        ke_vars  = []
        for eta in ETA_VALS:
            km_all = []
            kv_all = []
            for s in range(N_SEEDS):
                ke_tail = run_one(N, eta, C, seed=s)
                km, kv = ke_stats(ke_tail, N)
                km_all.append(km)
                kv_all.append(kv)
            ke_means.append(float(np.mean(km_all)))
            ke_vars.append(float(np.mean(kv_all)))
            print(f'    eta={eta:5.1f}  KE/N={ke_means[-1]:.4f}  chi={N*ke_vars[-1]:.4f}',
                  flush=True)
        results[N] = (np.array(ETA_VALS), np.array(ke_means), np.array(ke_vars))
    return results


# ---------------------------------------------------------------------------
# Run all compactness values
# ---------------------------------------------------------------------------
all_results = {}
for C in C_VALS:
    all_results[C] = sweep_C(C)


# ---------------------------------------------------------------------------
# Plotting: one figure per C value, KE/N and chi side-by-side
# ---------------------------------------------------------------------------

fig_ke, axes_ke = plt.subplots(2, 3, figsize=(14, 8))
fig_chi, axes_chi = plt.subplots(2, 3, figsize=(14, 8))
axes_ke  = axes_ke.flatten()
axes_chi = axes_chi.flatten()

for ci, C in enumerate(C_VALS):
    ax_ke  = axes_ke[ci]
    ax_chi = axes_chi[ci]
    results = all_results[C]

    for ni, N in enumerate(N_VALS):
        eta_arr, ke_arr, kv_arr = results[N]
        chi_arr = N * kv_arr
        ax_ke.plot(eta_arr, ke_arr, 'o-', color=colors[ni], label=f'N={N}')
        ax_chi.plot(eta_arr, chi_arr, 'o-', color=colors[ni], label=f'N={N}')

    ax_ke.set_title(f'C = {C:.2f}')
    ax_ke.set_xlabel('eta')
    ax_ke.set_ylabel('KE/N')
    ax_ke.legend(fontsize=7)

    ax_chi.set_title(f'C = {C:.2f}')
    ax_chi.set_xlabel('eta')
    ax_chi.set_ylabel('chi = N * Var(KE/N)')
    ax_chi.legend(fontsize=7)

fig_ke.suptitle('KE/N vs noise: repulsion-only model at intermediate compactness',
                fontsize=11)
fig_chi.suptitle('Susceptibility chi vs noise: looking for a diverging peak',
                 fontsize=11)
fig_ke.tight_layout()
fig_chi.tight_layout()
fig_ke.savefig('figures/compactness_search_KE.png', dpi=120)
fig_chi.savefig('figures/compactness_search_chi.png', dpi=120)
plt.close('all')


# ---------------------------------------------------------------------------
# Summary: print chi peak locations
# ---------------------------------------------------------------------------
print('\n=== Chi-peak summary ===')
print(f'{"C":>5}  {"N":>5}  {"eta_peak":>10}  {"chi_peak":>10}')
print('-' * 38)
for C in C_VALS:
    results = all_results[C]
    for N in N_VALS:
        eta_arr, _, kv_arr = results[N]
        chi_arr = N * kv_arr
        i_peak  = int(np.argmax(chi_arr))
        print(f'{C:5.2f}  {N:5d}  {eta_arr[i_peak]:10.1f}  {chi_arr[i_peak]:10.4f}')

print('\nFigures saved: compactness_search_KE.png, compactness_search_chi.png')
