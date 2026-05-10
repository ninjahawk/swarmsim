# compactness_phase.py -- Does a true phase transition appear at lower compactness?
#
# The previous phase_transition.py used r0=0.05 with fixed N values.
# Because compactness C = pi*N*r0^2 grows with N (not fixed), the earlier
# finite-size scaling mixed two effects: changing N AND changing density.
#
# Here we fix compactness by scaling r0 = sqrt(C_target / (pi*N)) so that
# each system size has the same density. We test two compactness values:
#   C_high = 0.78 (default regime -- agents highly caged)
#   C_low  = 0.10 (dilute regime -- agents have room to move)
#
# Prediction: at low compactness, agents are not caged, so a cooperative
# phase transition (diverging susceptibility) might emerge.

import os
import numpy as np
import matplotlib.pyplot as plt
from flocking import run, params, kinetic_energy

os.makedirs('figures', exist_ok=True)
N_SEEDS = 8
N_VALS  = [25, 50, 100, 200]
ETA_VALS = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
colors  = ['steelblue', 'firebrick', 'seagreen', 'darkorange']


def ke_stats(frames, N):
    tail = frames[int(0.8*len(frames)):]
    ke_n = [kinetic_energy(vx, vy)/N for _, _, vx, vy in tail]
    return float(np.mean(ke_n)), float(np.var(ke_n))


def run_sweep(N_vals, eta_vals, C_target, n_seeds, label):
    print(f'\n--- Compactness C={C_target:.2f} ({label}) ---')
    results = {}
    for N in N_vals:
        r0 = np.sqrt(C_target / (np.pi * N))
        print(f'N={N}  r0={r0:.4f}  C={np.pi*N*r0**2:.3f}')
        ke_means = []; ke_vars = []
        for eta in eta_vals:
            p = params(dict(N=N, n_iter=3000, alpha=0., v0=0., mu=10.,
                            eps=25., r0=float(r0), rf=0.1, ramp=float(eta)))
            km_all = []; kv_all = []
            for s in range(n_seeds):
                f = run(p, n_frames=100, seed=s)
                km, kv = ke_stats(f, N)
                km_all.append(km); kv_all.append(kv)
            ke_means.append(np.mean(km_all))
            ke_vars.append(np.mean(kv_all))
            print(f'  eta={eta:5.1f}  KE/N={np.mean(km_all):.4f}  chi={N*np.mean(kv_all):.4f}')
        results[N] = (np.array(eta_vals), np.array(ke_means), np.array(ke_vars))
    return results


results_high = run_sweep(N_VALS, ETA_VALS, C_target=0.78, n_seeds=N_SEEDS, label='high -- caged regime')
results_low  = run_sweep(N_VALS, ETA_VALS, C_target=0.10, n_seeds=N_SEEDS, label='low -- dilute regime')


# =============================================================================
# PLOTS: side-by-side comparison of KE/N and susceptibility for both C values
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Fixed-compactness finite-size scaling\n'
             'C_high=0.78 (caged) vs C_low=0.10 (dilute)', fontsize=11)

def plot_fss(ax_ke, ax_chi, results, N_vals, title_suffix):
    chi = {N: N * results[N][2] for N in N_vals}
    for i, N in enumerate(N_vals):
        eta_arr, ke_m, _ = results[N]
        ax_ke.plot(eta_arr, ke_m, 'o-', color=colors[i], label=f'N={N}', lw=1.5, ms=5)
        ax_chi.plot(eta_arr, chi[N], 'o-', color=colors[i], label=f'N={N}', lw=1.5, ms=5)
    ax_ke.set_xlabel('Noise amplitude eta')
    ax_ke.set_ylabel('KE / N')
    ax_ke.set_title(f'KE/N vs eta  ({title_suffix})')
    ax_ke.legend(fontsize=9)
    ax_chi.set_xlabel('Noise amplitude eta')
    ax_chi.set_ylabel('Susceptibility chi = N * var(KE/N)')
    ax_chi.set_title(f'Susceptibility  ({title_suffix})\nPeak = true phase transition')
    ax_chi.legend(fontsize=9)

plot_fss(axes[0,0], axes[0,1], results_high, N_VALS, 'C=0.78, caged')
plot_fss(axes[1,0], axes[1,1], results_low,  N_VALS, 'C=0.10, dilute')

plt.tight_layout()
plt.savefig('figures/compactness_phase.png', dpi=120)
plt.close()
print('\nSaved: figures/compactness_phase.png')


# =============================================================================
# PRINT SUSCEPTIBILITY PEAK LOCATIONS
# =============================================================================
print('\n--- Susceptibility peaks ---')
for label, results in [('C=0.78 (high)', results_high), ('C=0.10 (low)', results_low)]:
    print(f'\n{label}:')
    for N in N_VALS:
        eta_arr, _, ke_vars = results[N]
        chi_arr = N * ke_vars
        eta_crit = eta_arr[np.argmax(chi_arr)]
        print(f'  N={N:4d}  eta_crit~={eta_crit:.1f}  chi_max={chi_arr.max():.4f}')

print('\nNote: converging eta_crit with N -> true phase transition')
print('      monotone chi with no peak -> crossover')
print('\nCompactness phase analysis complete.')
