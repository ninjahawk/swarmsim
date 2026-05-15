# flocking3d.py -- 3D flocking model: validation and noise-sweep
#
# Extends the 2D Silverberg/Charbonneau model to three dimensions.
# Agents move on the periodic unit cube [0,1]^3 under the same four forces:
#   repulsion, velocity alignment, self-propulsion, and random noise.
#
# Parameter scaling from 2D to 3D:
#   - r0=0.02: repulsion radius (volume fraction N*(4/3)*pi*r0^3 = 0.012, dilute)
#   - rf=0.20: flocking radius (N*(4/3)*pi*rf^3 = 12.3 expected neighbors, ~same as 2D)
#   - All other parameters unchanged (alpha=1, v0=1, mu=10, dt=0.01)
#
# Finding 41 tests:
#   (1) v_eq = v0 + alpha/mu holds in 3D (analytical result from force balance)
#   (2) Phi vs ramp crossover in 3D vs 2D -- does flocking emerge in 3D?
#   (3) If crossover is similar to 2D, confirms the mechanism is dimension-independent

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('figures', exist_ok=True)

N        = 350
R0       = 0.02
RF       = 0.20
ALPHA    = 1.0
V0       = 1.0
MU       = 10.0
DT       = 0.01
N_ITER   = 4000
N_SEEDS  = 8
EPS      = 0.1
EXP_N    = 1.5
RB       = 2.0 * R0

RAMP_VALS = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]


def run_flock3d(N, ramp, seed):
    """Run 3D flocking. Returns (mean_Phi, std_Phi, mean_speed)."""
    np.random.seed(seed)

    # Positions on [0,1]^3
    pos = np.random.uniform(0.0, 1.0, (3, N))
    # Initial velocities: random 3D unit vectors scaled to V0
    raw = np.random.randn(3, N)
    raw /= np.sqrt((raw**2).sum(axis=0))
    vel = V0 * raw

    phi_series  = []
    spd_series  = []
    measure_start = (3 * N_ITER) // 4

    for step in range(N_ITER):
        # Pairwise displacement on 3D torus
        dx = pos[0, np.newaxis, :] - pos[0, :, np.newaxis]
        dy = pos[1, np.newaxis, :] - pos[1, :, np.newaxis]
        dz = pos[2, np.newaxis, :] - pos[2, :, np.newaxis]
        dx -= np.round(dx); dy -= np.round(dy); dz -= np.round(dz)
        d2 = dx**2 + dy**2 + dz**2

        not_self = ~np.eye(N, dtype=bool)

        # --- Repulsion ---
        rep_mask = (d2 <= RB**2) & not_self & (d2 > 0)
        d_safe   = np.where(rep_mask, np.sqrt(d2), 1.0)
        base_r   = np.maximum(np.where(rep_mask, 1.0 - d_safe / RB, 0.0), 0.0)
        strength = np.where(rep_mask, EPS * (base_r ** EXP_N) / d_safe, 0.0)
        fx = (-strength * dx).sum(axis=1)
        fy = (-strength * dy).sum(axis=1)
        fz = (-strength * dz).sum(axis=1)

        # --- Flocking (alignment) ---
        flock_mask = (d2 <= RF**2) & not_self
        n_nbr = flock_mask.sum(axis=1)  # number of neighbors per agent
        sum_vx = (vel[0] * flock_mask).sum(axis=1)
        sum_vy = (vel[1] * flock_mask).sum(axis=1)
        sum_vz = (vel[2] * flock_mask).sum(axis=1)
        vbar_mag = np.sqrt(sum_vx**2 + sum_vy**2 + sum_vz**2)
        # Only apply flocking force where at least one neighbor exists
        has_nbr = (n_nbr > 0)
        vbar_mag_safe = np.where(has_nbr, vbar_mag, 1.0)
        fx += np.where(has_nbr, ALPHA * sum_vx / vbar_mag_safe, 0.0)
        fy += np.where(has_nbr, ALPHA * sum_vy / vbar_mag_safe, 0.0)
        fz += np.where(has_nbr, ALPHA * sum_vz / vbar_mag_safe, 0.0)

        # --- Self-propulsion ---
        speed = np.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
        speed = np.maximum(speed, 1e-10)
        prop = MU * (V0 - speed) / speed
        fx += prop * vel[0]
        fy += prop * vel[1]
        fz += prop * vel[2]

        # --- Random noise ---
        if ramp > 0:
            fx += ramp * np.random.uniform(-1.0, 1.0, N)
            fy += ramp * np.random.uniform(-1.0, 1.0, N)
            fz += ramp * np.random.uniform(-1.0, 1.0, N)

        # --- Integrate ---
        vel[0] += fx * DT
        vel[1] += fy * DT
        vel[2] += fz * DT
        pos = (pos + vel * DT) % 1.0

        if step >= measure_start:
            spd_now = np.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
            spd_series.append(spd_now.mean())
            vhat_x = vel[0] / np.maximum(spd_now, 1e-10)
            vhat_y = vel[1] / np.maximum(spd_now, 1e-10)
            vhat_z = vel[2] / np.maximum(spd_now, 1e-10)
            phi_series.append(np.sqrt(
                vhat_x.mean()**2 + vhat_y.mean()**2 + vhat_z.mean()**2
            ))

    return np.mean(phi_series), np.std(phi_series), np.mean(spd_series)


print('3D flocking: N=%d  r0=%.3f  rf=%.2f  alpha=%.1f  v0=%.1f  mu=%.1f' % (
      N, R0, RF, ALPHA, V0, MU))
print('Expected neighbors at rf: N*(4/3)*pi*rf^3 = %.1f' % (
      N * (4.0/3.0) * np.pi * RF**3))
print('Expected v_eq = v0 + alpha/mu = %.3f' % (V0 + ALPHA / MU))
print()
print('ramp sweep: %s' % RAMP_VALS)
print()

results = {}
for ramp in RAMP_VALS:
    phi_vals = []; spd_vals = []
    for s in range(N_SEEDS):
        phi_m, phi_s, spd_m = run_flock3d(N, ramp, s)
        phi_vals.append(phi_m)
        spd_vals.append(spd_m)
    results[ramp] = (np.mean(phi_vals), np.std(phi_vals), np.mean(spd_vals))
    print('  ramp=%.1f: Phi=%.4f +/- %.4f  |v|=%.4f' % (
          ramp, results[ramp][0], results[ramp][1], results[ramp][2]))

print()
v_eq_pred = V0 + ALPHA / MU
print('Predicted v_eq = %.3f' % v_eq_pred)
print('Measured v_eq at ramp=0: %.4f' % results[0.0][2])
print()

# -----------------------------------------------------------------------
ramp_arr = np.array(RAMP_VALS)
phi_m    = np.array([results[r][0] for r in RAMP_VALS])
phi_s    = np.array([results[r][1] for r in RAMP_VALS])
spd_m    = np.array([results[r][2] for r in RAMP_VALS])

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.errorbar(ramp_arr, phi_m, yerr=phi_s, marker='o', capsize=4,
            color='steelblue', label='3D N=%d' % N)
ax.axhline(0, ls='--', color='gray', lw=1)
ax.axhline(1, ls='--', color='gray', lw=1)
ax.set_xlabel('Noise amplitude (ramp)')
ax.set_ylabel('Order parameter Phi = |mean(v_hat)|')
ax.set_title('3D flocking: Phi vs noise\nN=%d, rf=%.2f, alpha=%.1f, mu=%.1f' % (
             N, RF, ALPHA, MU))
ax.legend(); ax.grid(alpha=0.3)

ax2 = axes[1]
ax2.plot(ramp_arr, spd_m, marker='s', color='darkorange', label='Measured |v|')
ax2.axhline(V0 + ALPHA/MU, ls='--', color='black', lw=1.5,
            label='v_eq = v0+alpha/mu = %.3f' % (V0 + ALPHA/MU))
ax2.axhline(V0, ls=':', color='gray', lw=1, label='v0 = %.1f' % V0)
ax2.set_xlabel('Noise amplitude (ramp)')
ax2.set_ylabel('Mean agent speed |v|')
ax2.set_title('3D flocking: equilibrium speed check\nEquipartition prediction v_eq = v0+alpha/mu')
ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

fig.suptitle('3D flocking on [0,1]^3 torus: validation\n'
             'r0=%.3f  rf=%.2f  alpha=%.1f  v0=%.1f  mu=%.1f  N=%d  %d seeds' % (
             R0, RF, ALPHA, V0, MU, N, N_SEEDS), fontsize=10)
plt.tight_layout()
plt.savefig('figures/flocking3d_1.png', dpi=120)
plt.close()
print('  --> figures/flocking3d_1.png')
