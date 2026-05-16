# renc_scaling.py -- Encirclement radius scaling with flock size
#
# Finding 28 showed the encirclement floor RISES at large N because R_enc=0.15
# was tuned to N=350.  At N=1000 the flock is broader (Rg > 0.15), so
# R_enc=0.15 places predators inside the flock and they lose disruptive power.
#
# Direct test: sweep R_enc at N=1000 (and N=350 for reference) and look for
# the optimal R_enc that maximises disruption.  Hypothesis: optimal R_enc
# should scale with Rg.  Also measure Rg directly so we can plot R_enc / Rg.

import os
import numpy as np
import matplotlib.pyplot as plt
from flocking import buffer as buf_fn, order_parameter
from model import _periodic_com, _periodic_disp

os.makedirs('figures', exist_ok=True)
N_SEEDS = 4

BASE = dict(
    r0=0.005, eps=0.1, rf=0.1,
    alpha=1.0, v0=0.02, mu=10.0, ramp=0.1, dt=0.01,
    n_iter=3000,
)
PRED = dict(v0=0.05, mu=10.0, alpha=5.0, r0=0.1, eps=2.0, ramp=1.0)

N_PRED = 10


def rg(px, py):
    cx, cy = _periodic_com(px), _periodic_com(py)
    rel_x = (px - cx + 0.5) % 1.0 - 0.5
    rel_y = (py - cy + 0.5) % 1.0 - 0.5
    return float(np.sqrt((rel_x**2 + rel_y**2).mean()))


def run(N, R_enc, seed):
    np.random.seed(seed)
    p = BASE.copy()
    dt = p['dt']; n_iter = p['n_iter']
    r0, eps, rf = p['r0'], p['eps'], p['rf']
    v0, mu = p['v0'], p['mu']
    ramp = p['ramp']

    x  = np.zeros(2*N)
    x[:N] = np.random.uniform(0., 1., N)
    x[N:] = np.random.uniform(0., 1., N)
    vx = np.random.uniform(-1., 1., N) * v0
    vy = np.random.uniform(-1., 1., N) * v0

    pred_x  = np.random.uniform(0., 1., N_PRED)
    pred_y  = np.random.uniform(0., 1., N_PRED)
    pred_vx = np.zeros(N_PRED); pred_vy = np.zeros(N_PRED)
    pred_angles = np.radians(np.arange(N_PRED) * 360.0 / N_PRED)

    rb = max(r0, rf)
    phi_t=[]; rg_t=[]; mind_t=[]

    for i in range(n_iter):
        nb, xb, yb, vxb, vyb = buf_fn(rb, x, vx, vy, N)
        dx = xb[:nb][np.newaxis, :] - x[:N][:, np.newaxis]
        dy = yb[:nb][np.newaxis, :] - x[N:][:, np.newaxis]
        d2 = dx**2 + dy**2

        not_self = np.ones((N, nb), dtype=bool)
        idx = np.arange(min(N, nb))
        not_self[idx, idx] = False

        flock_mask = (d2 <= rf**2) & not_self
        flx = np.where(flock_mask, vxb[:nb], 0.).sum(axis=1)
        fly = np.where(flock_mask, vyb[:nb], 0.).sum(axis=1)
        nfl = flock_mask.sum(axis=1)
        nrm = np.sqrt(flx**2 + fly**2)
        nrm[nfl == 0] = 1.
        flockx = p['alpha'] * flx / nrm
        flocky = p['alpha'] * fly / nrm

        rep_mask = (d2 <= (2*r0)**2) & not_self & (d2 > 0)
        d_safe   = np.where(rep_mask, np.sqrt(d2), 1.)
        base     = np.where(rep_mask, 1. - d_safe/(2.*r0), 0.)
        strength = np.where(rep_mask, eps * base**1.5 / d_safe, 0.)
        repx = (-strength * dx).sum(axis=1)
        repy = (-strength * dy).sum(axis=1)

        vnorm  = np.sqrt(vx**2 + vy**2)
        vnorms = np.where(vnorm == 0, 1., vnorm)
        fpropx = mu * (v0 - vnorm) * vx / vnorms
        fpropy = mu * (v0 - vnorm) * vy / vnorms

        frandx = ramp * np.random.uniform(-1., 1., N)
        frandy = ramp * np.random.uniform(-1., 1., N)

        fx_total = flockx + repx + fpropx + frandx
        fy_total = flocky + repy + fpropy + frandy

        min_d = 1.0
        for k in range(N_PRED):
            ddx = _periodic_disp(pred_x[k], x[:N])
            ddy = _periodic_disp(pred_y[k], x[N:])
            d = np.sqrt(ddx**2 + ddy**2)
            min_d = min(min_d, d.min())
            mask_p = (d > 0) & (d <= PRED['r0'])
            if mask_p.any():
                strength_p = PRED['eps'] * (1. - d[mask_p]/PRED['r0'])**1.5 / d[mask_p]
                fx_total[mask_p] -= strength_p * ddx[mask_p]
                fy_total[mask_p] -= strength_p * ddy[mask_p]

        vx += fx_total * dt
        vy += fy_total * dt
        x[:N] = (x[:N] + vx*dt) % 1.
        x[N:] = (x[N:] + vy*dt) % 1.

        cx, cy = _periodic_com(x[:N]), _periodic_com(x[N:])
        for k in range(N_PRED):
            tx = (cx + R_enc * np.cos(pred_angles[k])) % 1.
            ty = (cy + R_enc * np.sin(pred_angles[k])) % 1.
            ddx = _periodic_disp(tx, pred_x[k])
            ddy = _periodic_disp(ty, pred_y[k])
            dist = np.sqrt(ddx**2 + ddy**2)
            if dist > 0:
                ddx /= dist; ddy /= dist
            sp_p = np.sqrt(pred_vx[k]**2 + pred_vy[k]**2)
            if sp_p > 0:
                pfx = PRED['alpha']*ddx + PRED['mu']*(PRED['v0']-sp_p)*pred_vx[k]/sp_p
                pfy = PRED['alpha']*ddy + PRED['mu']*(PRED['v0']-sp_p)*pred_vy[k]/sp_p
            else:
                pfx = PRED['alpha']*ddx; pfy = PRED['alpha']*ddy
            pfx += PRED['ramp']*np.random.uniform(-1., 1.)
            pfy += PRED['ramp']*np.random.uniform(-1., 1.)
            pred_vx[k] += pfx*dt; pred_vy[k] += pfy*dt
            pred_x[k] = (pred_x[k] + pred_vx[k]*dt) % 1.
            pred_y[k] = (pred_y[k] + pred_vy[k]*dt) % 1.

        if i >= 1000 and i % 50 == 0:
            phi_t.append(order_parameter(vx, vy))
            rg_t.append(rg(x[:N], x[N:]))
            mind_t.append(min_d)

    return np.mean(phi_t), np.mean(rg_t), np.mean(mind_t)


print('R_enc scaling at n_pred=10')
R_enc_vals = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]
N_vals = [350, 1000]

results = {N: {} for N in N_vals}
for N in N_vals:
    print('N = %d' % N)
    for R in R_enc_vals:
        phis=[]; rgs=[]; minds=[]
        for s in range(N_SEEDS):
            phi, rg_v, mind = run(N, R, s)
            phis.append(phi); rgs.append(rg_v); minds.append(mind)
        results[N][R] = dict(phi=np.mean(phis), phi_std=np.std(phis),
                              rg=np.mean(rgs), mind=np.mean(minds))
        print('  R_enc=%.2f  Phi=%.3f +/- %.3f  Rg=%.3f  R_enc/Rg=%.2f  mind=%.4f' %
              (R, results[N][R]['phi'], results[N][R]['phi_std'],
               results[N][R]['rg'], R/results[N][R]['rg'], results[N][R]['mind']),
              flush=True)

# Find Phi minimum for each N
print('\nPhi minimum:')
for N in N_vals:
    R_min = min(R_enc_vals, key=lambda R: results[N][R]['phi'])
    print('  N=%d  optimal R_enc=%.2f  R_enc/Rg=%.2f  Phi_min=%.3f' %
          (N, R_min, R_min/results[N][R_min]['rg'], results[N][R_min]['phi']))


# Plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Encirclement radius scaling (n_pred=10, %d seeds)' % N_SEEDS, fontsize=11)

ax = axes[0]
for N, color in [(350, 'steelblue'), (1000, 'crimson')]:
    phis = [results[N][R]['phi']     for R in R_enc_vals]
    stds = [results[N][R]['phi_std'] for R in R_enc_vals]
    ax.errorbar(R_enc_vals, phis, yerr=stds, fmt='o-', color=color,
                lw=2, capsize=4, label='N=%d' % N)
ax.set_xlabel('R_enc'); ax.set_ylabel('Steady-state Phi')
ax.set_ylim(0, 1.05); ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.set_title('Phi vs R_enc')

ax = axes[1]
for N, color in [(350, 'steelblue'), (1000, 'crimson')]:
    rgs = [results[N][R]['rg'] for R in R_enc_vals]
    phis = [results[N][R]['phi'] for R in R_enc_vals]
    r_over_rg = np.array(R_enc_vals) / np.array(rgs)
    ax.plot(r_over_rg, phis, 'o-', color=color, lw=2, label='N=%d' % N)
ax.set_xlabel('R_enc / Rg'); ax.set_ylabel('Steady-state Phi')
ax.set_ylim(0, 1.05); ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.set_title('Collapse: Phi vs R_enc / Rg')

plt.tight_layout()
plt.savefig('figures/renc_scaling_1.png', dpi=120)
plt.close()
print('  --> figures/renc_scaling_1.png')

print('\nR_enc scaling complete.')
