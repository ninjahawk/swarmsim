# flocking3d_noise.py -- 3D flocking: extended noise sweep and 2D comparison
#
# Finding 41 showed that 3D flocking is coherent up to ramp=10 (Phi=0.84) but the
# crossover to the disordered phase is at ramp >> 10.  This experiment:
#   (1) Extends the 3D ramp sweep to ramp=30 to find the full transition region
#   (2) Runs the equivalent 2D sweep at matched parameters for direct comparison
#   (3) Computes susceptibility chi = N * Var_seeds(mean_Phi) to test for a finite-
#       size dependent peak (true transition vs smooth crossover)
#
# Hypothesis: the 3D crossover is also smooth (no diverging chi_peak), mirroring
# the 2D result but with the transition shifted to higher ramp (~20-25 vs 2D ~20-25).
# If the 3D model has a true Vicsek-type phase transition, chi_peak would grow with N
# and shift to a finite ramp_c.  Both 2D and 3D Vicsek models are known to have a
# continuous noise-driven transition; this experiment tests the same for the force-based
# Charbonneau/Silverberg model in 3D.

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('figures', exist_ok=True)

N_VALS_3D  = [100, 200, 350]
N_2D       = 350
N_SEEDS    = 8
N_ITER     = 4000
DT         = 0.01

# 3D parameters
R0_3D  = 0.02
RF_3D  = 0.20
EPS    = 0.1
EXP_N  = 1.5
ALPHA  = 1.0
V0     = 1.0
MU     = 10.0

# Ramp sweep: extend to 30 to fully bracket the transition
RAMP_VALS = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0]


def run_flock3d(N, ramp, seed):
    """Run 3D flocking. Returns mean_Phi over last quarter."""
    np.random.seed(seed)
    RB = 2.0 * R0_3D

    pos = np.random.uniform(0.0, 1.0, (3, N))
    raw = np.random.randn(3, N)
    raw /= np.sqrt((raw**2).sum(axis=0))
    vel = V0 * raw

    phi_series = []
    measure_start = (3 * N_ITER) // 4

    for step in range(N_ITER):
        dx = pos[0, np.newaxis, :] - pos[0, :, np.newaxis]
        dy = pos[1, np.newaxis, :] - pos[1, :, np.newaxis]
        dz = pos[2, np.newaxis, :] - pos[2, :, np.newaxis]
        dx -= np.round(dx); dy -= np.round(dy); dz -= np.round(dz)
        d2 = dx**2 + dy**2 + dz**2
        not_self = ~np.eye(N, dtype=bool)

        rep_mask = (d2 <= RB**2) & not_self & (d2 > 0)
        d_safe   = np.where(rep_mask, np.sqrt(d2), 1.0)
        base_r   = np.maximum(np.where(rep_mask, 1.0 - d_safe / RB, 0.0), 0.0)
        strength = np.where(rep_mask, EPS * (base_r ** EXP_N) / d_safe, 0.0)
        fx = (-strength * dx).sum(axis=1)
        fy = (-strength * dy).sum(axis=1)
        fz = (-strength * dz).sum(axis=1)

        flock_mask = (d2 <= RF_3D**2) & not_self
        n_nbr  = flock_mask.sum(axis=1)
        svx = (vel[0] * flock_mask).sum(axis=1)
        svy = (vel[1] * flock_mask).sum(axis=1)
        svz = (vel[2] * flock_mask).sum(axis=1)
        vmag = np.sqrt(svx**2 + svy**2 + svz**2)
        has  = (n_nbr > 0)
        vsafe = np.where(has, vmag, 1.0)
        fx += np.where(has, ALPHA * svx / vsafe, 0.0)
        fy += np.where(has, ALPHA * svy / vsafe, 0.0)
        fz += np.where(has, ALPHA * svz / vsafe, 0.0)

        spd  = np.maximum(np.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2), 1e-10)
        prop = MU * (V0 - spd) / spd
        fx += prop * vel[0]; fy += prop * vel[1]; fz += prop * vel[2]

        if ramp > 0:
            fx += ramp * np.random.uniform(-1.0, 1.0, N)
            fy += ramp * np.random.uniform(-1.0, 1.0, N)
            fz += ramp * np.random.uniform(-1.0, 1.0, N)

        vel[0] += fx * DT; vel[1] += fy * DT; vel[2] += fz * DT
        pos = (pos + vel * DT) % 1.0

        if step >= measure_start:
            spd2  = np.maximum(np.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2), 1e-10)
            phi_series.append(np.sqrt(
                (vel[0]/spd2).mean()**2 + (vel[1]/spd2).mean()**2 + (vel[2]/spd2).mean()**2
            ))

    return np.mean(phi_series)


def run_flock2d(N, ramp, seed, r0=0.005, rf=0.1):
    """Run 2D flocking for comparison. Returns mean_Phi over last quarter."""
    np.random.seed(seed)
    RB = 2.0 * r0

    x = np.random.uniform(0.0, 1.0, N)
    y = np.random.uniform(0.0, 1.0, N)
    ang = np.random.uniform(0.0, 2.0 * np.pi, N)
    vx = V0 * np.cos(ang)
    vy = V0 * np.sin(ang)

    phi_series = []
    measure_start = (3 * N_ITER) // 4

    for step in range(N_ITER):
        rdx = x[np.newaxis, :] - x[:, np.newaxis]
        rdy = y[np.newaxis, :] - y[:, np.newaxis]
        rdx -= np.round(rdx); rdy -= np.round(rdy)
        d2 = rdx**2 + rdy**2
        not_self = ~np.eye(N, dtype=bool)

        rep_mask = (d2 <= RB**2) & not_self & (d2 > 0)
        d_safe   = np.where(rep_mask, np.sqrt(d2), 1.0)
        base_r   = np.maximum(np.where(rep_mask, 1.0 - d_safe / RB, 0.0), 0.0)
        strength = np.where(rep_mask, EPS * (base_r ** EXP_N) / d_safe, 0.0)
        fx = (-strength * rdx).sum(axis=1)
        fy = (-strength * rdy).sum(axis=1)

        flock_mask = (d2 <= rf**2) & not_self
        n_nbr = flock_mask.sum(axis=1)
        svx = (vx * flock_mask).sum(axis=1)
        svy = (vy * flock_mask).sum(axis=1)
        vmag = np.sqrt(svx**2 + svy**2)
        has  = (n_nbr > 0)
        vsafe = np.where(has, vmag, 1.0)
        fx += np.where(has, ALPHA * svx / vsafe, 0.0)
        fy += np.where(has, ALPHA * svy / vsafe, 0.0)

        spd  = np.maximum(np.sqrt(vx**2 + vy**2), 1e-10)
        prop = MU * (V0 - spd) / spd
        fx += prop * vx; fy += prop * vy

        if ramp > 0:
            fx += ramp * np.random.uniform(-1.0, 1.0, N)
            fy += ramp * np.random.uniform(-1.0, 1.0, N)

        vx += fx * DT; vy += fy * DT
        x = (x + vx * DT) % 1.0; y = (y + vy * DT) % 1.0

        if step >= measure_start:
            spd2 = np.maximum(np.sqrt(vx**2 + vy**2), 1e-10)
            phi_series.append(np.sqrt(
                (vx/spd2).mean()**2 + (vy/spd2).mean()**2
            ))

    return np.mean(phi_series)


print('3D flocking extended noise sweep + 2D comparison')
print('3D: N=%s  r0=%.3f  rf=%.2f  (%.1f nbrs)' % (
      N_VALS_3D, R0_3D, RF_3D, N_VALS_3D[-1]*(4/3)*np.pi*RF_3D**3))
print('2D: N=%d  r0=%.3f  rf=%.2f  (%.1f nbrs)' % (
      N_2D, 0.005, 0.1, N_2D*np.pi*0.01))
print('ramp=%s' % RAMP_VALS)
print()

# 3D sweep at multiple N
res3d = {}
for N in N_VALS_3D:
    res3d[N] = {}
    for ramp in RAMP_VALS:
        phi_vals = [run_flock3d(N, ramp, s) for s in range(N_SEEDS)]
        res3d[N][ramp] = (np.mean(phi_vals), np.std(phi_vals))
    print('  3D N=%3d: done' % N, flush=True)

# 2D sweep at N=350
res2d = {}
for ramp in RAMP_VALS:
    phi_vals = [run_flock2d(N_2D, ramp, s) for s in range(N_SEEDS)]
    res2d[ramp] = (np.mean(phi_vals), np.std(phi_vals))
print('  2D N=350: done', flush=True)
print()

# Summary: chi = N * Var_seeds(Phi)
print('=== chi = N * Var_seeds(Phi) for 3D (finite-size scaling) ===')
ramp_arr = np.array(RAMP_VALS)
for N in N_VALS_3D:
    phi_means = np.array([res3d[N][r][0] for r in RAMP_VALS])
    phi_stds  = np.array([res3d[N][r][1] for r in RAMP_VALS])
    chi       = N * (phi_stds**2)
    peak_idx  = np.argmax(chi)
    print('  N=%3d: chi_peak=%.4f at ramp=%.1f  |  Phi(ramp=0.5)=%.4f  Phi(ramp=20)=%.4f' % (
          N, chi[peak_idx], RAMP_VALS[peak_idx],
          res3d[N][0.5][0], res3d[N][20.0][0]))
print()
print('=== 2D N=350 (matched parameters) ===')
phi2_means = np.array([res2d[r][0] for r in RAMP_VALS])
chi2 = N_2D * (np.array([res2d[r][1] for r in RAMP_VALS])**2)
peak2 = np.argmax(chi2)
print('  chi_peak=%.4f at ramp=%.1f  |  Phi(ramp=0.5)=%.4f  Phi(ramp=20)=%.4f' % (
      chi2[peak2], RAMP_VALS[peak2], res2d[0.5][0], res2d[20.0][0]))

# -----------------------------------------------------------------------
colors3d = {100: 'steelblue', 200: 'seagreen', 350: 'darkorange'}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
for N in N_VALS_3D:
    phi_m = np.array([res3d[N][r][0] for r in RAMP_VALS])
    phi_s = np.array([res3d[N][r][1] for r in RAMP_VALS])
    ax.errorbar(ramp_arr, phi_m, yerr=phi_s, marker='o', capsize=3,
                color=colors3d[N], label='3D N=%d' % N, ms=5)
phi2m = np.array([res2d[r][0] for r in RAMP_VALS])
phi2s = np.array([res2d[r][1] for r in RAMP_VALS])
ax.errorbar(ramp_arr, phi2m, yerr=phi2s, marker='s', capsize=3,
            color='crimson', label='2D N=350', ms=5, ls='--')
ax.set_xlabel('Noise amplitude (ramp)')
ax.set_ylabel('Order parameter Phi')
ax.set_title('3D vs 2D: Phi vs noise (extended sweep)')
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.set_ylim(-0.05, 1.05)

ax2 = axes[1]
for N in N_VALS_3D:
    phi_s = np.array([res3d[N][r][1] for r in RAMP_VALS])
    chi   = N * (phi_s**2)
    ax2.plot(ramp_arr, chi, marker='o', color=colors3d[N], label='3D N=%d' % N, ms=5)
ax2.plot(ramp_arr, chi2, marker='s', color='crimson', label='2D N=350', ms=5, ls='--')
ax2.set_xlabel('Noise amplitude (ramp)')
ax2.set_ylabel('chi = N * Var_seeds(Phi)')
ax2.set_title('Susceptibility chi vs noise\n(peak at finite ramp_c = transition signal)')
ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

fig.suptitle('3D vs 2D flocking: extended noise sweep (Finding 42)\n'
             '3D: r0=0.02, rf=0.20 (~12 nbrs)  |  2D: r0=0.005, rf=0.10 (~11 nbrs)\n'
             'N_SEEDS=%d  N_ITER=%d  dt=%.2f' % (N_SEEDS, N_ITER, DT), fontsize=10)
plt.tight_layout()
plt.savefig('figures/flocking3d_noise_1.png', dpi=120)
plt.close()
print()
print('  --> figures/flocking3d_noise_1.png')
