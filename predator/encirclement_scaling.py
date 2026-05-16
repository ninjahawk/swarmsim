# encirclement_scaling.py -- Does the encirclement threshold scale with flock size?
#
# Finding 14 showed Phi drops to 0.77 at n_pred=6 for N=350.
# Question: is 6 a fixed number, or does the threshold scale with N?
# If it scales as n_pred/N = const, that implies a predator-to-prey ratio law.
#
# Experiments:
#   1. Fixed n_pred=6, vary N (50, 100, 200, 350, 500) -- does Phi recover for larger N?
#   2. Fixed ratio n_pred/N ~ 1/58 (the N=350 threshold), vary N
#   3. Full threshold sweep: for N=100 and N=350, find Phi vs n_pred

import os
import numpy as np
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from flocking import params, buffer, force, order_parameter
from predator import PREY_DEFAULT, PRED_DEFAULT
from multi_predator import geom_multi, mean_min_pred_dist
from encirclement import run_encirclement

os.makedirs('figures', exist_ok=True)
N_SEEDS = 8
R_ENC = 0.15
n_it = 4000; n_fr = 200


# =============================================================================
# EXP 1: FIXED n_pred=6, VARY N
# =============================================================================
print('Exp 1: Fixed n_pred=6, vary N  (%d seeds)' % N_SEEDS)
N_vals = [50, 100, 200, 350, 500]
n_pred_fixed = 6

res1 = {}
for N in N_vals:
    phi_all = []; dist_all = []
    pp = PREY_DEFAULT.copy(); pp['n_iter'] = n_it; pp['N'] = N
    for s in range(N_SEEDS):
        f = run_encirclement(n_pred=n_pred_fixed, R_enc=R_ENC,
                             prey_overrides=pp, n_frames=n_fr, seed=s)
        _, _, phi = geom_multi(f)
        phi_all.append(phi[-40:].mean())
        dist_all.append(mean_min_pred_dist(f)[-40:].mean())
    res1[N] = dict(phi=np.mean(phi_all), phi_std=np.std(phi_all),
                   dist=np.mean(dist_all))
    print('  N=%3d  Phi=%.3f +/- %.3f  dist=%.3f' % (
        N, res1[N]['phi'], res1[N]['phi_std'], res1[N]['dist']))


# =============================================================================
# EXP 2: FIXED RATIO n_pred/N ~ 6/350, VARY N
# =============================================================================
print('\nExp 2: Fixed ratio n_pred/N=6/350, vary N  (%d seeds)' % N_SEEDS)
ratio = 6. / 350.

res2 = {}
for N in N_vals:
    n_pred_r = max(1, round(ratio * N))
    phi_all = []; dist_all = []
    pp = PREY_DEFAULT.copy(); pp['n_iter'] = n_it; pp['N'] = N
    for s in range(N_SEEDS):
        f = run_encirclement(n_pred=n_pred_r, R_enc=R_ENC,
                             prey_overrides=pp, n_frames=n_fr, seed=s)
        _, _, phi = geom_multi(f)
        phi_all.append(phi[-40:].mean())
        dist_all.append(mean_min_pred_dist(f)[-40:].mean())
    res2[N] = dict(phi=np.mean(phi_all), phi_std=np.std(phi_all),
                   dist=np.mean(dist_all), n_pred=n_pred_r)
    print('  N=%3d  n_pred=%d  Phi=%.3f +/- %.3f  dist=%.3f' % (
        N, n_pred_r, res2[N]['phi'], res2[N]['phi_std'], res2[N]['dist']))


# =============================================================================
# EXP 3: FULL PHI vs n_pred SWEEP FOR N=100 AND N=350
# =============================================================================
print('\nExp 3: Full Phi vs n_pred sweep  (N=100 and N=350, %d seeds)' % N_SEEDS)
n_pred_sweep = [1, 2, 3, 4, 6, 8, 10, 12]
compare_N = [100, 350]

res3 = {}
for N in compare_N:
    res3[N] = {}
    pp = PREY_DEFAULT.copy(); pp['n_iter'] = n_it; pp['N'] = N
    for n_pred in n_pred_sweep:
        phi_all = []
        for s in range(N_SEEDS):
            f = run_encirclement(n_pred=n_pred, R_enc=R_ENC,
                                 prey_overrides=pp, n_frames=n_fr, seed=s)
            _, _, phi = geom_multi(f)
            phi_all.append(phi[-40:].mean())
        res3[N][n_pred] = dict(phi=np.mean(phi_all), phi_std=np.std(phi_all))
        print('  N=%3d  n_pred=%2d  Phi=%.3f +/- %.3f' % (
            N, n_pred, res3[N][n_pred]['phi'], res3[N][n_pred]['phi_std']))


# =============================================================================
# FIGURES
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Encirclement threshold scaling with flock size  (%d seeds)' % N_SEEDS, fontsize=11)

# Exp 1: fixed n_pred=6, vary N
phi_m = [res1[N]['phi'] for N in N_vals]
phi_s = [res1[N]['phi_std'] for N in N_vals]
axes[0].errorbar(N_vals, phi_m, yerr=phi_s, fmt='o-', color='steelblue', lw=2, capsize=5)
axes[0].axhline(0.77, color='gray', ls='--', lw=0.8, label='N=350 baseline')
axes[0].set_xlabel('Flock size N'); axes[0].set_ylabel('Phi')
axes[0].set_title('Fixed n_pred=6, vary N\n(does larger flock resist?)')
axes[0].set_ylim(0, 1); axes[0].legend(fontsize=8)

# Exp 2: fixed ratio
phi_m2 = [res2[N]['phi'] for N in N_vals]
phi_s2 = [res2[N]['phi_std'] for N in N_vals]
axes[1].errorbar(N_vals, phi_m2, yerr=phi_s2, fmt='s-', color='darkorange', lw=2, capsize=5)
axes[1].axhline(0.77, color='gray', ls='--', lw=0.8, label='N=350 baseline')
for N in N_vals:
    axes[1].annotate('n_p=%d' % res2[N]['n_pred'], (N, res2[N]['phi']),
                     textcoords='offset points', xytext=(4, 4), fontsize=7)
axes[1].set_xlabel('Flock size N'); axes[1].set_ylabel('Phi')
axes[1].set_title('Fixed ratio n_pred/N=6/350, vary N\n(ratio law test)')
axes[1].set_ylim(0, 1); axes[1].legend(fontsize=8)

# Exp 3: full sweep for N=100 vs N=350
colors3 = {'steelblue': 100, 'crimson': 350}
for color, N in [(c, n) for c, n in [('steelblue', 100), ('crimson', 350)]]:
    phi_m3 = [res3[N][n]['phi'] for n in n_pred_sweep]
    phi_s3 = [res3[N][n]['phi_std'] for n in n_pred_sweep]
    axes[2].errorbar(n_pred_sweep, phi_m3, yerr=phi_s3, fmt='o-', color=color,
                     lw=2, capsize=4, label='N=%d' % N)
axes[2].axhline(0.5, color='gray', ls='--', lw=0.8, label='Phi=0.5')
axes[2].set_xlabel('Number of encircling predators'); axes[2].set_ylabel('Phi')
axes[2].set_title('Phi vs n_pred for N=100 vs N=350')
axes[2].set_ylim(0, 1); axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig('figures/encircle_scaling.png', dpi=120)
plt.close()
print('\n  --> figures/encircle_scaling.png')
print('Encirclement scaling analysis complete.')
