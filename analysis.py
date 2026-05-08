# analysis.py -- Validation and parameter-space exploration for the flocking model.
#
# Phases:
#   2 -- Validation via limiting cases        (saves figures/validate_*.png)
#   3 -- Order parameter and KE over time     (saves figures/phase3_*.png)
#   4 -- Parameter sweeps: eta and alpha      (saves figures/phase4_*.png)

import os
import numpy as np
import matplotlib.pyplot as plt

from flocking import run, params, order_parameter, kinetic_energy

os.makedirs('figures', exist_ok=True)
SEED = 42


# =============================================================================
# HELPERS
# =============================================================================
def final_metrics(frames):
    """(Phi, KE) averaged over last 20% of frames -- steady-state estimate."""
    tail = frames[int(0.8 * len(frames)):]
    phis = [order_parameter(vx, vy) for _, _, vx, vy in tail]
    kes  = [kinetic_energy(vx, vy)  for _, _, vx, vy in tail]
    return float(np.mean(phis)), float(np.mean(kes))

def snap(frames, ax, t_target, p, title=None):
    """Scatter+quiver snapshot at time closest to t_target."""
    frame_step = max(1, p['n_iter'] // len(frames))
    fi = min(int(t_target / (frame_step * p['dt'])), len(frames)-1)
    px, py, vx, vy = frames[fi]
    sp = np.sqrt(vx**2 + vy**2); sp[sp == 0] = 1.
    ax.scatter(px, py, s=2, color='steelblue')
    ax.quiver(px, py, vx/sp, vy/sp, scale=60, width=0.003, color='firebrick')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title or f't = {fi * frame_step * p["dt"]:.1f}', fontsize=9)


# =============================================================================
# PHASE 2 -- VALIDATION: THREE LIMITING CASES
# =============================================================================
print('=' * 60)
print('PHASE 2 -- VALIDATION: LIMITING CASES')
print('=' * 60)

# ---- Case 1: pure random walk (all physical forces off) --------------------
print('\nCase 1: pure random walk  (eps=0, alpha=0, mu=0, v0=0, ramp=1)')
p1 = params(dict(N=100, n_iter=1000, eps=0., alpha=0., mu=0., v0=0., ramp=1.0))
f1 = run(p1, n_frames=100, seed=SEED)
px_f, py_f, vx_f, vy_f = f1[-1]
phi1, _ = final_metrics(f1)
print(f'  Phi = {phi1:.3f}  (expected ~0 -- no preferred direction)')
print(f'  x std={px_f.std():.3f}  y std={py_f.std():.3f}  (uniform ~=0.29)')
print('  PASS' if phi1 < 0.15 else '  WARNING: Phi unexpectedly high')

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle('Validation 1 -- Pure random walk (no forces)', fontsize=11)
for ax, t in zip(axes, [0., 2., 9.9]):
    snap(f1, ax, t, p1)
plt.tight_layout()
plt.savefig('figures/validate_1_random_walk.png', dpi=120)
plt.close()
print('  --> figures/validate_1_random_walk.png')

# ---- Case 2: repulsion + noise only (alpha=0, v0=0) ------------------------
# Reproduces Fig 10.5: hexagonal crystal at low eta, fluid at high eta
print('\nCase 2: repulsion + noise only  (alpha=0, v0=0)')
eta_vals = [1, 3, 10, 30]
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
fig.suptitle('Validation 2 -- Repulsion + noise only (alpha=0, v0=0), snapshot t=20',
             fontsize=10)
for ax, eta in zip(axes, eta_vals):
    p2 = params(dict(N=100, n_iter=2000, alpha=0., v0=0., mu=10.,
                     eps=25., r0=0.05, rf=0.1, ramp=float(eta)))
    f2 = run(p2, n_frames=100, seed=SEED)
    snap(f2, ax, 20., p2, title=f'eta={eta}')
    phi2, _ = final_metrics(f2)
    print(f'  eta={eta:3d}  Phi={phi2:.3f}')
plt.tight_layout()
plt.savefig('figures/validate_2_repulsion_noise.png', dpi=120)
plt.close()
print('  Expected: crystal (ordered) at eta=1, fluid (disordered) at eta=30')
print('  --> figures/validate_2_repulsion_noise.png')

# ---- Case 3: flocking only (eps=0, v0=0) ------------------------------------
# Reproduces Fig 10.6: random -> streaming flock
print('\nCase 3: flocking only  (eps=0, v0=0)')
p3 = params(dict(N=100, n_iter=3000, eps=0., v0=0., mu=10., alpha=1., ramp=0.1))
f3 = run(p3, n_frames=150, seed=SEED)
phi3, _ = final_metrics(f3)
print(f'  Final Phi = {phi3:.3f}  (expected >0.8 -- coherent flock)')
print('  PASS' if phi3 > 0.7 else '  WARNING: flocking weaker than expected')

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
fig.suptitle('Validation 3 -- Flocking only (eps=0, v0=0), flock formation', fontsize=11)
for ax, t in zip(axes.flat, [0.5, 1.0, 2.0, 4.0, 8.0, 29.]):
    snap(f3, ax, t, p3)
plt.tight_layout()
plt.savefig('figures/validate_3_flocking_only.png', dpi=120)
plt.close()
print('  --> figures/validate_3_flocking_only.png')

print('\nValidation complete.')


# =============================================================================
# PHASE 3 -- ORDER PARAMETER AND KE TIME SERIES (default parameters)
# =============================================================================
print('\n' + '=' * 60)
print('PHASE 3 -- TIME SERIES: ORDER PARAMETER AND KE')
print('=' * 60)

p_base = params(dict(N=100, n_iter=2000))
f_base = run(p_base, n_frames=200, seed=SEED)

phi_t = [order_parameter(vx, vy) for _, _, vx, vy in f_base]
ke_t  = [kinetic_energy(vx, vy)  for _, _, vx, vy in f_base]
fs    = max(1, p_base['n_iter'] // len(f_base))
t_arr = np.arange(len(f_base)) * fs * p_base['dt']

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle(f'Time evolution -- N={p_base["N"]}, default parameters', fontsize=10)

ax_a.plot(t_arr, phi_t, color='steelblue')
ax_a.set_xlabel('Time')
ax_a.set_ylabel('Order parameter Phi')
ax_a.set_ylim(0, 1)
ax_a.set_title('Phi = |mean(v_hat)|')
ax_a.axhline(np.mean(phi_t[-40:]), color='gray', ls='--', lw=0.8, label='steady state')
ax_a.legend(fontsize=8)

ax_b.plot(t_arr, ke_t, color='firebrick')
ax_b.set_xlabel('Time')
ax_b.set_ylabel('Kinetic energy')
ax_b.set_title('KE = 0.5 * sum(v^2)')

plt.tight_layout()
plt.savefig('figures/phase3_timeseries.png', dpi=120)
plt.close()

ss_phi, ss_ke = final_metrics(f_base)
print(f'  Steady-state: Phi={ss_phi:.3f}  KE={ss_ke:.2f}')
print('  --> figures/phase3_timeseries.png')


# =============================================================================
# PHASE 4 -- PARAMETER SWEEPS
# =============================================================================
print('\n' + '=' * 60)
print('PHASE 4 -- PARAMETER SWEEPS')
print('=' * 60)

N_sw  = 100
nit   = 2000

# ---- Sweep A: KE vs eta (repulsion+noise only, Exercise 2) -----------------
print('\nSweep A: KE vs eta  (repulsion+noise, alpha=0, v0=0)')
eta_A  = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0]
ke_A   = []
for eta in eta_A:
    pA = params(dict(N=N_sw, n_iter=nit, alpha=0., v0=0.,
                     mu=10., eps=25., r0=0.05, ramp=float(eta)))
    _, ke = final_metrics(run(pA, n_frames=80, seed=SEED))
    ke_A.append(ke)
    print(f'  eta={eta:5.1f}  KE={ke:.2f}')

# ---- Sweep B: Phi vs alpha (flocking amplitude, Exercise 3) ----------------
print('\nSweep B: Phi vs alpha  (flocking only, eps=0, v0=0, eta=0.1)')
alpha_B = [0., 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
phi_B   = []
for alpha in alpha_B:
    pB = params(dict(N=N_sw, n_iter=nit, alpha=float(alpha),
                     eps=0., v0=0., mu=10., ramp=0.1))
    phi, _ = final_metrics(run(pB, n_frames=80, seed=SEED))
    phi_B.append(phi)
    print(f'  alpha={alpha:.2f}  Phi={phi:.3f}')

# ---- Sweep C: Phi and KE vs eta (full model) --------------------------------
print('\nSweep C: Phi and KE vs eta  (full model, all forces active)')
eta_C  = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]
phi_C  = []; ke_C = []
for eta in eta_C:
    pC = params(dict(N=N_sw, n_iter=nit, ramp=float(eta)))
    phi, ke = final_metrics(run(pC, n_frames=80, seed=SEED))
    phi_C.append(phi); ke_C.append(ke)
    print(f'  eta={eta:5.1f}  Phi={phi:.3f}  KE={ke:.2f}')

# ---- Plot all sweeps --------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle(f'Parameter sweeps  (N={N_sw})', fontsize=11)

axes[0].plot(eta_A, ke_A, 'o-', color='steelblue')
axes[0].set_xlabel('Noise amplitude eta')
axes[0].set_ylabel('Steady-state KE')
axes[0].set_title('Sweep A: KE vs eta\n(repulsion+noise only)\nSolid to fluid transition')

axes[1].plot(alpha_B, phi_B, 's-', color='firebrick')
axes[1].axhline(0.5, color='gray', ls='--', lw=0.8, label='Phi=0.5 threshold')
axes[1].set_xlabel('Flocking amplitude alpha')
axes[1].set_ylabel('Order parameter Phi')
axes[1].set_title('Sweep B: Phi vs alpha\n(flocking only, eta=0.1)\nMin alpha for sustained flock')
axes[1].set_ylim(0, 1); axes[1].legend(fontsize=8)

ax2 = axes[2].twinx()
l1, = axes[2].plot(eta_C, phi_C, 'o-', color='steelblue', label='Phi')
l2, = ax2.plot(eta_C, ke_C, 's--', color='firebrick', label='KE')
axes[2].set_xlabel('Noise amplitude eta')
axes[2].set_ylabel('Order parameter Phi', color='steelblue')
ax2.set_ylabel('Kinetic energy', color='firebrick')
axes[2].set_title('Sweep C: Phi and KE vs eta\n(full model)')
axes[2].set_ylim(0, 1)
axes[2].legend(handles=[l1, l2], loc='center right', fontsize=8)

plt.tight_layout()
plt.savefig('figures/phase4_sweeps.png', dpi=120)
plt.close()
print('\n  --> figures/phase4_sweeps.png')

print('\n' + '=' * 60)
print('Analysis complete. All figures in figures/')
print('=' * 60)
