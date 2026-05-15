# hard_repulsion.py -- Finite-size scaling with harder repulsion exponents
#
# Finding 17 showed no phase transition at any compactness value tested (C = 0.10-0.78)
# in the standard model.  The interpretation was that the soft repulsion potential
# (1 - d/2r0)^1.5 decays smoothly to zero at agent contact and cannot generate the
# diverging spatial correlations required for a true critical point.
#
# This experiment directly tests that conjecture by replacing the exponent 1.5 with
# harder values: n = 1.5 (current), 3, 6, 12.  A hard-core limit (n -> inf) should
# approach the behavior of particles with true excluded-volume repulsion, where a
# solid-to-fluid phase transition is well established (e.g., 2D hard-disc melting).
#
# Protocol (identical to phase_transition.py / Finding 8, for direct comparison):
#   - Repulsion-only system (alpha=0, v0=0, mu=0) -- no flocking, no propulsion
#   - Compactness fixed at C = 0.40 (intermediate, from compactness_search.py)
#     so r0 = sqrt(C/(pi*N)) scales properly with N
#   - Noise sweep eta = 0.5 to 30 (10 values), 8 seeds each
#   - System sizes N = 25, 50, 100, 200
#   - Finite-size scaling: KE/N and chi = N * Var(KE/N)
#
# Expected: harder exponents should produce a steeper KE/N crossover that becomes
# more system-size-dependent, and eventually a chi peak that shifts with N --
# signatures of a true phase transition.

import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('figures', exist_ok=True)

N_VALS   = [25, 50, 100, 200]
ETA_VALS = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0]
N_SEEDS  = 8
N_ITER   = 4000
DT       = 0.01
C        = 0.40          # intermediate compactness
EPS0     = 0.1           # repulsion amplitude (same as baseline)
EXP_VALS = [1.5, 3.0, 6.0, 12.0]   # repulsion exponents to test


def run_repulsion(N, eta, seed, exp_n):
    """Run repulsion-only system with force exponent exp_n.

    Force: F_rep = eps * (1 - d/(2r0))^exp_n / d  [for d < 2r0]
    Returns final mean kinetic energy per agent.
    """
    np.random.seed(seed)
    r0  = np.sqrt(C / (np.pi * N))
    eps = EPS0
    rb  = 2. * r0

    x  = np.zeros(2*N)
    x[:N] = np.random.uniform(0., 1., N)
    x[N:] = np.random.uniform(0., 1., N)
    vx = np.zeros(N)
    vy = np.zeros(N)

    ke_series = []

    for step in range(N_ITER):
        # Build ghost buffer for periodic forces
        nx = x[:N]; ny = x[N:]

        # Inline buffer construction (same logic as flocking.buf)
        real_dx = nx[np.newaxis, :] - nx[:, np.newaxis]
        real_dy = ny[np.newaxis, :] - ny[:, np.newaxis]
        real_dx -= np.round(real_dx); real_dy -= np.round(real_dy)
        d2 = real_dx**2 + real_dy**2

        not_self = ~np.eye(N, dtype=bool)
        rep_mask = (d2 <= rb**2) & not_self & (d2 > 0)
        d_safe   = np.where(rep_mask, np.sqrt(d2), 1.)
        # Hard(er) repulsion: (1 - d/2r0)^exp_n / d
        base_r   = np.where(rep_mask, 1. - d_safe / rb, 0.)
        # Clip to avoid negative bases when d > 2r0 slips through
        base_r   = np.maximum(base_r, 0.)
        strength = np.where(rep_mask, eps * (base_r ** exp_n) / d_safe, 0.)
        repx = (-strength * real_dx).sum(axis=1)
        repy = (-strength * real_dy).sum(axis=1)

        # Random noise (replacing ramp eta)
        frandx = eta * np.random.uniform(-1., 1., N)
        frandy = eta * np.random.uniform(-1., 1., N)

        vx += (repx + frandx) * DT
        vy += (repy + frandy) * DT
        x[:N] = (x[:N] + vx * DT) % 1.
        x[N:] = (x[N:] + vy * DT) % 1.

        if step >= N_ITER // 2:
            ke_series.append(0.5 * (vx**2 + vy**2).mean())

    return np.mean(ke_series)


print('Hard repulsion finite-size scaling')
print('C=%.2f  N=%s  eta=%s' % (C, N_VALS, ETA_VALS))
print('Exponents tested: %s' % EXP_VALS)
print()

# results[exp_n][N][eta] = (mean_ke, std_ke)
results = {}
for exp_n in EXP_VALS:
    results[exp_n] = {}
    for N in N_VALS:
        results[exp_n][N] = {}
        for eta in ETA_VALS:
            ke_vals = [run_repulsion(N, eta, s, exp_n) for s in range(N_SEEDS)]
            ke_arr  = np.array(ke_vals)
            results[exp_n][N][eta] = (ke_arr.mean(), ke_arr.std())
        print('  exp=%.1f  N=%3d  done' % (exp_n, N), flush=True)

print()
print('=== Summary: chi_peak location ===')
print('(chi = N * Var(KE/N); if chi_peak is at eta < 30 and shifts with N -> phase transition)')
print()
for exp_n in EXP_VALS:
    print('Exponent = %.1f:' % exp_n)
    for N in N_VALS:
        ke_means = np.array([results[exp_n][N][eta][0] for eta in ETA_VALS])
        ke_stds  = np.array([results[exp_n][N][eta][1] for eta in ETA_VALS])
        chi_vals = N * (ke_stds**2)
        peak_idx = np.argmax(chi_vals)
        print('  N=%3d: chi_peak = %.4f at eta = %.1f   KE/N range = [%.4f, %.4f]' % (
            N, chi_vals[peak_idx], ETA_VALS[peak_idx],
            ke_means.min() / N, ke_means.max() / N))
    print()

# -----------------------------------------------------------------------
# Figures: 2x2 grid, one panel per exponent; each panel shows chi vs eta for all N
fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True)
axes = axes.flatten()
colors = {25: 'steelblue', 50: 'seagreen', 100: 'darkorange', 200: 'crimson'}

for k, exp_n in enumerate(EXP_VALS):
    ax = axes[k]
    for N in N_VALS:
        ke_means = np.array([results[exp_n][N][eta][0] for eta in ETA_VALS])
        ke_stds  = np.array([results[exp_n][N][eta][1] for eta in ETA_VALS])
        chi_vals = N * (ke_stds**2)
        ax.plot(ETA_VALS, chi_vals, marker='o', label='N=%d' % N,
                color=colors[N], ms=5)
    ax.set_title('Repulsion exponent n = %.1f' % exp_n)
    ax.set_xlabel('Noise amplitude eta')
    ax.set_ylabel('Susceptibility chi = N * Var(KE/N)')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_yscale('log')

fig.suptitle('Finite-size susceptibility chi vs noise eta\nC=0.40, repulsion-only\n'
             'A diverging peak that shifts with N signals a true phase transition',
             fontsize=11)
plt.tight_layout()
plt.savefig('figures/hard_repulsion_1_chi.png', dpi=120)
plt.close()
print('  --> figures/hard_repulsion_1_chi.png')

# Second figure: KE/N vs eta for each exponent (all N overlaid)
fig2, axes2 = plt.subplots(2, 2, figsize=(13, 10), sharex=True)
axes2 = axes2.flatten()
for k, exp_n in enumerate(EXP_VALS):
    ax = axes2[k]
    for N in N_VALS:
        ke_means = np.array([results[exp_n][N][eta][0] for eta in ETA_VALS])
        ke_stds  = np.array([results[exp_n][N][eta][1] for eta in ETA_VALS])
        ax.errorbar(ETA_VALS, ke_means / N, yerr=ke_stds / N,
                    marker='o', label='N=%d' % N,
                    color=colors[N], ms=4, capsize=3)
    ax.set_title('Repulsion exponent n = %.1f' % exp_n)
    ax.set_xlabel('Noise amplitude eta')
    ax.set_ylabel('KE/N')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig2.suptitle('Mean kinetic energy per agent KE/N vs noise eta\nC=0.40, repulsion-only\n'
              'N-independent curves -> crossover;  N-dependent -> phase transition',
              fontsize=11)
plt.tight_layout()
plt.savefig('figures/hard_repulsion_2_ke.png', dpi=120)
plt.close()
print('  --> figures/hard_repulsion_2_ke.png')
