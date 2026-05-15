# langevin_hexatic.py -- Langevin dynamics with hexatic order parameter
#
# Finding 39 showed that the Langevin thermostat works correctly (equipartition
# satisfied) but chi = N*Var(KE/N) cannot detect the KTHNY structural phase
# transition because KTHNY melting is a transition in positional order, not kinetic
# energy.  The right observable is the hexatic order parameter:
#
#   psi6_j = (1/k_j) * sum_{neighbors i} exp(6 * i * theta_{ji})
#
# where theta_{ji} is the angle of the bond from j to i, and k_j is the number
# of neighbors.  |mean(psi6)| is near 1 in a hexagonal solid and drops toward 0
# in the fluid phase.
#
# This experiment replaces chi_KE with chi_psi6 = N * Var(|psi6|) as the
# susceptibility.  A true KTHNY transition would show chi_psi6 peaking at a finite
# kT_c that shifts to LOWER kT as N increases (critical slowing down of the hexatic
# susceptibility).
#
# Protocol (same Langevin parameters as F39):
#   - n=1.5 repulsion, Langevin thermostat, mu=10
#   - C=0.60 (below melting) and C=0.70 (near/above melting)
#   - kT = 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0
#   - N = 25, 50, 100, 200; 8 seeds
#   - Neighbor cutoff: nearest-neighbor distance ~ 2*r0 (using 3*r0 to be safe)
#
# If KTHNY transition exists: chi_psi6 peaks at finite kT_c, chi_psi6_peak grows
# with N, and kT_c shifts slightly downward with N (approach from finite-size scaling).

import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('figures', exist_ok=True)

N_VALS     = [25, 50, 100, 200]
KT_VALS    = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
N_SEEDS    = 8
N_ITER     = 10000    # longer equilibration; measure last quarter
DT         = 0.01
MU         = 10.0
EPS0       = 0.1
EXP_N      = 1.5
C_VALS     = [0.60, 0.70]


def run_langevin_hex(N, kT, seed, C):
    """Run Langevin dynamics. Returns mean |psi6| and mean KE/N over last quarter."""
    np.random.seed(seed)
    r0  = np.sqrt(C / (np.pi * N))
    rb  = 2. * r0
    r_nbr = 3. * r0    # neighbor cutoff for hexatic order

    noise_std = np.sqrt(2. * MU * kT * DT)

    x = np.zeros(2 * N)
    x[:N] = np.random.uniform(0., 1., N)
    x[N:] = np.random.uniform(0., 1., N)
    v_init = np.sqrt(kT) if kT > 0 else 0.
    vx = v_init * np.random.randn(N)
    vy = v_init * np.random.randn(N)

    psi6_series = []
    ke_series   = []

    measure_start = (3 * N_ITER) // 4

    for step in range(N_ITER):
        nx = x[:N]; ny = x[N:]

        # Pairwise distances on torus
        real_dx = nx[np.newaxis, :] - nx[:, np.newaxis]
        real_dy = ny[np.newaxis, :] - ny[:, np.newaxis]
        real_dx -= np.round(real_dx); real_dy -= np.round(real_dy)
        d2 = real_dx**2 + real_dy**2

        not_self = ~np.eye(N, dtype=bool)

        # Repulsion force
        rep_mask = (d2 <= rb**2) & not_self & (d2 > 0)
        d_safe   = np.where(rep_mask, np.sqrt(d2), 1.)
        base_r   = np.maximum(np.where(rep_mask, 1. - d_safe / rb, 0.), 0.)
        strength = np.where(rep_mask, EPS0 * (base_r ** EXP_N) / d_safe, 0.)
        repx = (-strength * real_dx).sum(axis=1)
        repy = (-strength * real_dy).sum(axis=1)

        # Langevin thermostat
        damp_x = -MU * vx; damp_y = -MU * vy
        frandx = noise_std * np.random.randn(N) / DT
        frandy = noise_std * np.random.randn(N) / DT

        vx += (repx + damp_x + frandx) * DT
        vy += (repy + damp_y + frandy) * DT
        x[:N] = (x[:N] + vx * DT) % 1.
        x[N:] = (x[N:] + vy * DT) % 1.

        if step >= measure_start:
            ke_series.append(0.5 * (vx**2 + vy**2).mean())

            # Hexatic order parameter psi6
            nbr_mask = (d2 <= r_nbr**2) & not_self
            angles = np.where(nbr_mask, np.arctan2(real_dy, real_dx), 0.)
            # psi6_j = mean of exp(6i*theta) over neighbors
            cos6 = np.where(nbr_mask, np.cos(6. * angles), 0.).sum(axis=1)
            sin6 = np.where(nbr_mask, np.sin(6. * angles), 0.).sum(axis=1)
            k_j  = nbr_mask.sum(axis=1).clip(min=1)
            psi6_abs = np.sqrt((cos6 / k_j)**2 + (sin6 / k_j)**2)
            psi6_series.append(psi6_abs.mean())

    return np.mean(ke_series), np.mean(psi6_series), np.std(psi6_series)


print('Langevin + hexatic order parameter finite-size scaling')
print('C_vals=%s  n=%.1f  mu=%.1f' % (C_VALS, EXP_N, MU))
print('N=%s  kT=%s' % (N_VALS, KT_VALS))
print()

# results[C][N][kT] = (mean_ke, mean_psi6, std_psi6_timeseries)
results = {}
for C in C_VALS:
    results[C] = {}
    for N in N_VALS:
        results[C][N] = {}
        ke_list   = {}
        psi6_list = {}
        for kT in KT_VALS:
            ke_vals   = []
            psi6_vals = []
            for s in range(N_SEEDS):
                ke, psi6_mean, psi6_std = run_langevin_hex(N, kT, s, C)
                ke_vals.append(ke)
                psi6_vals.append(psi6_mean)
            results[C][N][kT] = (np.mean(ke_vals), np.std(ke_vals),
                                  np.mean(psi6_vals), np.std(psi6_vals))
        print('  C=%.2f  N=%3d  done' % (C, N), flush=True)

print()
print('=== Summary: chi_psi6 (N * Var_seeds(mean_psi6)) and mean_psi6 ===')
print('  KTHNY transition: chi_psi6 peaks at finite kT_c; peak shifts lower with N')
print()
for C in C_VALS:
    print('C = %.2f:' % C)
    for N in N_VALS:
        psi6_means = np.array([results[C][N][kT][2] for kT in KT_VALS])
        psi6_stds  = np.array([results[C][N][kT][3] for kT in KT_VALS])
        chi_psi6   = N * (psi6_stds**2)
        peak_idx   = np.argmax(chi_psi6)
        ke_ref     = results[C][N][0.1][0]
        print('  N=%3d: chi_psi6_peak=%.4f at kT=%.3f  |  psi6(kT=0.001)=%.3f  psi6(kT=5.0)=%.3f  |  KE/N(kT=0.1)=%.4f' % (
            N, chi_psi6[peak_idx], KT_VALS[peak_idx],
            psi6_means[0], psi6_means[-1], ke_ref))
    print()

# -----------------------------------------------------------------------
# Figures: 2 rows (C=0.60, C=0.70), 3 columns (chi_psi6, mean_psi6, KE/N)
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
colors = {25: 'steelblue', 50: 'seagreen', 100: 'darkorange', 200: 'crimson'}
kT_arr = np.array(KT_VALS)

for row, C in enumerate(C_VALS):
    ax_chi  = axes[row, 0]
    ax_psi  = axes[row, 1]
    ax_ke   = axes[row, 2]

    for N in N_VALS:
        ke_means   = np.array([results[C][N][kT][0] for kT in KT_VALS])
        ke_stds    = np.array([results[C][N][kT][1] for kT in KT_VALS])
        psi6_means = np.array([results[C][N][kT][2] for kT in KT_VALS])
        psi6_stds  = np.array([results[C][N][kT][3] for kT in KT_VALS])
        chi_psi6   = N * (psi6_stds**2)

        ax_chi.plot(kT_arr, chi_psi6, marker='o', label='N=%d' % N,
                    color=colors[N], ms=5)
        ax_psi.errorbar(kT_arr, psi6_means,
                        yerr=psi6_stds / np.sqrt(N_SEEDS),
                        marker='o', label='N=%d' % N,
                        color=colors[N], ms=4, capsize=3)
        ax_ke.errorbar(kT_arr, ke_means,
                       yerr=ke_stds / np.sqrt(N_SEEDS),
                       marker='o', label='N=%d' % N,
                       color=colors[N], ms=4, capsize=3)

    ax_ke.plot(kT_arr, kT_arr, 'k--', lw=1, label='KE/N=kT (equip.)')

    ax_chi.set_xlabel('Temperature kT'); ax_chi.set_ylabel('chi_psi6 = N*Var(psi6)')
    ax_chi.set_title('C=%.2f: Hexatic susceptibility' % C)
    ax_chi.legend(fontsize=8); ax_chi.grid(alpha=0.3); ax_chi.set_xscale('log')

    ax_psi.set_xlabel('Temperature kT'); ax_psi.set_ylabel('Mean |psi6|')
    ax_psi.set_title('C=%.2f: Hexatic order (1=solid, 0=fluid)' % C)
    ax_psi.legend(fontsize=8); ax_psi.grid(alpha=0.3); ax_psi.set_xscale('log')

    ax_ke.set_xlabel('Temperature kT'); ax_ke.set_ylabel('Mean KE/N')
    ax_ke.set_title('C=%.2f: KE/N vs kT (equipartition check)' % C)
    ax_ke.legend(fontsize=8); ax_ke.grid(alpha=0.3)
    ax_ke.set_xscale('log'); ax_ke.set_yscale('log')

fig.suptitle('Langevin + hexatic order: finite-size scaling for KTHNY melting\n'
             'n=1.5 repulsion, mu=10, FDT-satisfying noise\n'
             'chi_psi6 peak shifting with N = KTHNY transition signal',
             fontsize=11)
plt.tight_layout()
plt.savefig('figures/langevin_hexatic_1.png', dpi=120)
plt.close()
print('  --> figures/langevin_hexatic_1.png')
