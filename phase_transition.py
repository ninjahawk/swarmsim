# phase_transition.py -- Is the solid-to-fluid transition a true phase transition?
#
# The repulsion-only system (alpha=0, v0=0) shows KE rising with noise eta.
# This analysis asks: is there a sharp critical point, or just a smooth crossover?
#
# Methods:
#   - Finite-size scaling: run same eta sweep for multiple N values.
#     A true phase transition shows the curves crossing at a critical point.
#   - Susceptibility: chi = N * var(KE/N) -- diverges at a critical point.
#   - Order parameter: use mean speed as proxy (solid -> agents ~stationary, fluid -> moving)

import os
import numpy as np
import matplotlib.pyplot as plt
from flocking import run, params, kinetic_energy

os.makedirs('figures', exist_ok=True)
SEED = 42
N_SEEDS = 8    # seeds per (N, eta) point

def mean_speed(frames):
    tail = frames[int(0.8*len(frames)):]
    speeds = [np.sqrt(vx**2+vy**2).mean() for _,_,vx,vy in tail]
    return float(np.mean(speeds)), float(np.std(speeds))

def ke_stats(frames, N):
    """Return mean and variance of KE/N in steady state."""
    tail = frames[int(0.8*len(frames)):]
    ke_n = [kinetic_energy(vx,vy)/N for _,_,vx,vy in tail]
    return float(np.mean(ke_n)), float(np.var(ke_n))


# =============================================================================
# FINITE-SIZE SCALING: KE/N vs eta for multiple N
# =============================================================================
print('Finite-size scaling: KE/N vs eta for multiple system sizes')
print('Parameters: alpha=0, v0=0, mu=10, eps=25, r0=0.05')
print()

# eta values -- fine grid around expected transition region
eta_vals = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0]
N_vals   = [25, 50, 100, 200]

# fixed compactness C = pi*N*r0^2 across all N by keeping r0 fixed
# C = pi*25*0.05^2 = 0.196  (moderately packed -- same for all N)

results = {}   # results[N] = (eta_arr, ke_mean, ke_var, speed_mean)

for N in N_vals:
    print(f'N={N}:')
    ke_means = []; ke_vars = []; sp_means = []
    for eta in eta_vals:
        p = params(dict(N=N, n_iter=3000, alpha=0., v0=0., mu=10.,
                        eps=25., r0=0.05, rf=0.1, ramp=float(eta)))
        km_all, kv_all, sm_all = [], [], []
        for s in range(N_SEEDS):
            f = run(p, n_frames=100, seed=s)
            km, kv = ke_stats(f, N)
            sm, _  = mean_speed(f)
            km_all.append(km); kv_all.append(kv); sm_all.append(sm)
        ke_means.append(np.mean(km_all))
        ke_vars.append(np.mean(kv_all))   # mean of within-run variance
        sp_means.append(np.mean(sm_all))
        print(f'  eta={eta:5.1f}  KE/N={np.mean(km_all):.4f}  chi={N*np.mean(kv_all):.4f}  speed={np.mean(sm_all):.4f}')
    results[N] = (np.array(eta_vals), np.array(ke_means),
                  np.array(ke_vars),  np.array(sp_means))

# susceptibility: chi = N * var(KE/N) -- should peak at critical eta
chi = {N: N * results[N][2] for N in N_vals}

# =============================================================================
# PLOTS
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Phase transition analysis: repulsion+noise system (alpha=0, v0=0)', fontsize=11)
colors = ['steelblue', 'firebrick', 'seagreen', 'darkorange']

# Panel 1: KE/N vs eta (finite-size scaling)
for i, N in enumerate(N_vals):
    eta_arr, ke_m, _, _ = results[N]
    axes[0].plot(eta_arr, ke_m, 'o-', color=colors[i], label=f'N={N}', lw=1.5, ms=5)
axes[0].set_xlabel('Noise amplitude eta')
axes[0].set_ylabel('KE / N (per-agent kinetic energy)')
axes[0].set_title('Finite-size scaling: KE/N vs eta')
axes[0].legend(fontsize=9)

# Panel 2: Susceptibility chi = N * var(KE/N)
for i, N in enumerate(N_vals):
    eta_arr = results[N][0]
    axes[1].plot(eta_arr, chi[N], 'o-', color=colors[i], label=f'N={N}', lw=1.5, ms=5)
axes[1].set_xlabel('Noise amplitude eta')
axes[1].set_ylabel('Susceptibility chi = N * var(KE/N)')
axes[1].set_title('Susceptibility -- peaks at critical eta')
axes[1].legend(fontsize=9)

# Panel 3: Mean agent speed vs eta
for i, N in enumerate(N_vals):
    eta_arr, _, _, sp_m = results[N]
    axes[2].plot(eta_arr, sp_m, 'o-', color=colors[i], label=f'N={N}', lw=1.5, ms=5)
axes[2].set_xlabel('Noise amplitude eta')
axes[2].set_ylabel('Mean agent speed')
axes[2].set_title('Mean speed vs eta\n(solid: ~0, fluid: rising)')
axes[2].legend(fontsize=9)

plt.tight_layout()
plt.savefig('figures/phase_transition_scaling.png', dpi=120)
plt.close()
print('\nSaved: figures/phase_transition_scaling.png')

# =============================================================================
# IDENTIFY CRITICAL ETA FROM SUSCEPTIBILITY PEAKS
# =============================================================================
print('\nSusceptibility peaks (estimated critical eta):')
for N in N_vals:
    eta_arr = results[N][0]
    chi_arr = chi[N]
    eta_crit = eta_arr[np.argmax(chi_arr)]
    print(f'  N={N:4d}  eta_crit ~= {eta_crit:.1f}  (chi_max={chi_arr.max():.4f})')

print('\nNote: if this is a true phase transition, eta_crit should converge')
print('as N -> inf. If it drifts with N, it is likely a crossover.')

# =============================================================================
# COLLAPSE PLOT: rescaled axes for data collapse
# =============================================================================
# Try to collapse KE/N curves onto a single master curve by rescaling eta.
# For a phase transition: KE/N = N^(-beta/nu) * f((eta-eta_c)*N^(1/nu))
# We don't know exponents, so just plot (eta - eta_crit(N)) to check alignment.
print('\nGenerating collapse plot...')

# estimate eta_crit per N from susceptibility peak
eta_crits = {}
for N in N_vals:
    eta_arr = results[N][0]
    eta_crits[N] = eta_arr[np.argmax(chi[N])]

fig2, ax2 = plt.subplots(figsize=(7, 5))
for i, N in enumerate(N_vals):
    eta_arr, ke_m, _, _ = results[N]
    eta_c = eta_crits[N]
    ax2.plot(eta_arr - eta_c, ke_m, 'o-', color=colors[i], label=f'N={N}', lw=1.5, ms=5)
ax2.axvline(0, color='gray', ls='--', lw=0.8, label='eta = eta_crit')
ax2.set_xlabel('eta - eta_crit(N)  (shifted)')
ax2.set_ylabel('KE / N')
ax2.set_title('Collapse plot: KE/N vs (eta - eta_crit)\nCurves should overlap if universal scaling holds')
ax2.legend(fontsize=9)
plt.tight_layout()
plt.savefig('figures/phase_transition_collapse.png', dpi=120)
plt.close()
print('Saved: figures/phase_transition_collapse.png')
print('\nPhase transition analysis complete.')
