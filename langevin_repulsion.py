# langevin_repulsion.py -- Finite-size scaling with proper thermal (Langevin) dynamics
#
# Finding 38 showed that repulsion hardness does not produce a phase transition.
# The root cause: the model's non-equilibrium driving (uniform random kicks, no
# viscous damping) violates the fluctuation-dissipation theorem (FDT), so the
# system cannot equilibrate into a Boltzmann distribution and cannot exhibit the
# cooperative melting required for a true phase transition.
#
# This experiment replaces the non-equilibrium driving with proper Langevin dynamics:
#   vx_{t+1} = vx_t - mu * vx_t * dt + sqrt(2*mu*kT*dt) * N(0,1)
#   vy_{t+1} = vy_t - mu * vy_t * dt + sqrt(2*mu*kT*dt) * N(0,1)
#
# This satisfies the FDT: the system reaches thermal equilibrium at temperature kT.
# The 2D hard-disc melting (KTHNY transition) is well known to occur near area
# fraction phi_c ~ 0.69-0.72 in equilibrium.  C=0.40 is below this, so we expect
# to be in the solid phase at low kT and the fluid phase at high kT, with a true
# critical point at some kT_c.
#
# Protocol:
#   - Repulsion-only (n=1.5, same as standard model) + Langevin thermostat
#   - Compactness C = 0.40 (same as F38) and C = 0.60 (closer to melting)
#   - kT sweep: 0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0
#   - N = 25, 50, 100, 200; 8 seeds
#   - mu = 10 (damping rate -- sets relaxation timescale tau = 1/mu = 0.1 tu)
#   - Metrics: KE/N (should be kT at equilibrium) and chi = N * Var(KE/N)
#
# If the system thermalizes properly, KE/N = kT at equilibrium (equipartition).
# A true phase transition would show chi peaking at a finite kT_c that shifts with N.

import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('figures', exist_ok=True)

N_VALS     = [25, 50, 100, 200]
KT_VALS    = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
N_SEEDS    = 8
N_ITER     = 8000     # longer to allow equilibration; measure last half
DT         = 0.01
MU         = 10.0     # Langevin damping coefficient
EPS0       = 0.1      # repulsion amplitude
EXP_N      = 1.5      # repulsion exponent (standard)
C_VALS     = [0.60, 0.70]    # bracket the hard-disc melting point (~phi=0.69-0.72)


def run_langevin(N, kT, seed, C):
    """Run Langevin repulsion dynamics. Returns mean KE/N over last half."""
    np.random.seed(seed)
    r0  = np.sqrt(C / (np.pi * N))
    eps = EPS0
    rb  = 2. * r0

    # Noise amplitude satisfying FDT: std = sqrt(2 * mu * kT * dt)
    noise_std = np.sqrt(2. * MU * kT * DT)

    x  = np.zeros(2*N)
    x[:N] = np.random.uniform(0., 1., N)
    x[N:] = np.random.uniform(0., 1., N)
    # Initialize velocities from Maxwell-Boltzmann distribution at temperature kT
    # std(v_x) = sqrt(kT) (unit mass)
    v_init = np.sqrt(kT) if kT > 0 else 0.
    vx = v_init * np.random.randn(N)
    vy = v_init * np.random.randn(N)

    ke_series = []

    for step in range(N_ITER):
        nx = x[:N]; ny = x[N:]

        # Pairwise repulsion on torus
        real_dx = nx[np.newaxis, :] - nx[:, np.newaxis]
        real_dy = ny[np.newaxis, :] - ny[:, np.newaxis]
        real_dx -= np.round(real_dx); real_dy -= np.round(real_dy)
        d2 = real_dx**2 + real_dy**2

        not_self = ~np.eye(N, dtype=bool)
        rep_mask = (d2 <= rb**2) & not_self & (d2 > 0)
        d_safe   = np.where(rep_mask, np.sqrt(d2), 1.)
        base_r   = np.maximum(np.where(rep_mask, 1. - d_safe / rb, 0.), 0.)
        strength = np.where(rep_mask, eps * (base_r ** EXP_N) / d_safe, 0.)
        repx = (-strength * real_dx).sum(axis=1)
        repy = (-strength * real_dy).sum(axis=1)

        # Langevin thermostat: damping + thermal noise
        damp_x = -MU * vx
        damp_y = -MU * vy
        frandx = noise_std * np.random.randn(N) / DT   # per-unit-time noise force
        frandy = noise_std * np.random.randn(N) / DT

        vx += (repx + damp_x + frandx) * DT
        vy += (repy + damp_y + frandy) * DT
        x[:N] = (x[:N] + vx * DT) % 1.
        x[N:] = (x[N:] + vy * DT) % 1.

        if step >= N_ITER // 2:
            ke_series.append(0.5 * (vx**2 + vy**2).mean())

    return np.mean(ke_series)


print('Langevin (thermal) repulsion finite-size scaling')
print('C=0.60 is below hard-disc melting (~0.70); C=0.70 is near melting transition')
print('mu=%.1f  EPS=%.1f  n=%.1f  C_vals=%s' % (MU, EPS0, EXP_N, C_VALS))
print('N=%s  kT=%s' % (N_VALS, KT_VALS))
print()

# results[C][N][kT] = (mean_ke, std_ke)
results = {}
for C in C_VALS:
    results[C] = {}
    for N in N_VALS:
        results[C][N] = {}
        for kT in KT_VALS:
            ke_vals = [run_langevin(N, kT, s, C) for s in range(N_SEEDS)]
            ke_arr  = np.array(ke_vals)
            results[C][N][kT] = (ke_arr.mean(), ke_arr.std())
        print('  C=%.2f  N=%3d  done' % (C, N), flush=True)

print()
print('=== Summary: chi_peak and equipartition check ===')
print('  At thermal equilibrium: KE/N should equal kT (equipartition)')
print('  Phase transition: chi_peak at finite kT_c, shifting with N')
print()
for C in C_VALS:
    print('Compactness C = %.2f:' % C)
    for N in N_VALS:
        ke_means = np.array([results[C][N][kT][0] for kT in KT_VALS])
        ke_stds  = np.array([results[C][N][kT][1] for kT in KT_VALS])
        chi_vals = N * (ke_stds**2)
        peak_idx = np.argmax(chi_vals)
        # Equipartition check: KE/N vs kT at kT=0.1
        kt_ref = 0.1
        ke_ref = results[C][N][kt_ref][0]
        print('  N=%3d: chi_peak=%.4f at kT=%.3f  |  KE/N(kT=0.1)=%.4f vs kT=0.100' % (
            N, chi_vals[peak_idx], KT_VALS[peak_idx], ke_ref))
    print()

# -----------------------------------------------------------------------
# Figures: 2 rows (C=0.40, C=0.60), 2 columns (chi vs kT, KE/N vs kT)
fig, axes = plt.subplots(2, 2, figsize=(13, 10))
colors = {25: 'steelblue', 50: 'seagreen', 100: 'darkorange', 200: 'crimson'}
kT_arr = np.array(KT_VALS)

for row, C in enumerate(C_VALS):
    ax_chi = axes[row, 0]
    ax_ke  = axes[row, 1]

    for N in N_VALS:
        ke_means = np.array([results[C][N][kT][0] for kT in KT_VALS])
        ke_stds  = np.array([results[C][N][kT][1] for kT in KT_VALS])
        chi_vals = N * (ke_stds**2)
        ax_chi.plot(kT_arr, chi_vals, marker='o', label='N=%d' % N,
                    color=colors[N], ms=5)
        ax_ke.errorbar(kT_arr, ke_means, yerr=ke_stds / np.sqrt(N_SEEDS),
                       marker='o', label='N=%d' % N,
                       color=colors[N], ms=4, capsize=3)

    # Equipartition reference line: KE/N = kT
    ax_ke.plot(kT_arr, kT_arr, 'k--', lw=1, label='KE/N = kT (equip.)')

    ax_chi.set_xlabel('Temperature kT'); ax_chi.set_ylabel('chi = N * Var(KE/N)')
    ax_chi.set_title('C=%.2f: Susceptibility' % C)
    ax_chi.legend(fontsize=8); ax_chi.grid(alpha=0.3); ax_chi.set_xscale('log')

    ax_ke.set_xlabel('Temperature kT'); ax_ke.set_ylabel('Mean KE/N')
    ax_ke.set_title('C=%.2f: KE/N vs kT (check: should track KE=kT)' % C)
    ax_ke.legend(fontsize=8); ax_ke.grid(alpha=0.3)
    ax_ke.set_xscale('log'); ax_ke.set_yscale('log')

fig.suptitle('Langevin thermostat: thermal equilibrium finite-size scaling\n'
             'n=1.5 repulsion, mu=10, FDT-satisfying noise\n'
             'A chi_peak shifting with N on left panels = true phase transition',
             fontsize=11)
plt.tight_layout()
plt.savefig('figures/langevin_1.png', dpi=120)
plt.close()
print('  --> figures/langevin_1.png')
