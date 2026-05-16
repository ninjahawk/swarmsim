# large_N_encirclement.py -- Does the encirclement floor hold at large N?
#
# Finding 15 conjectured that the Phi ~ 0.67 floor at n_pred = 10 is set by
# ANGULAR COVERAGE (predators per 360 degrees), not by predator-to-prey ratio
# or absolute predator count.  Both N=100 and N=350 converged to the same floor.
# This script tests the conjecture by going to N=1000.  If angular coverage
# is the relevant variable, the floor at n_pred=10 should still be ~0.67.
#
# Setup: slow-prey predator regime (v0=0.02, ramp=0.1).  n_pred = 6, 10, 14.
# Three N values for comparison: 350, 700, 1000.
#
# Cost: N=1000 has ~3x the force computations of N=350.  Use 4 seeds and
# 3000 steps to keep the run feasible (~30 min total).

import os
import numpy as np
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from flocking import buffer as buf_fn, order_parameter
from model import _periodic_com, _periodic_disp

os.makedirs('figures', exist_ok=True)
N_SEEDS = 4

BASE = dict(
    r0=0.005, eps=0.1, rf=0.1,
    alpha=1.0, v0=0.02, mu=10.0, ramp=0.1, dt=0.01,
    n_iter=3000,
)

PRED = dict(v0=0.05, mu=10.0, alpha=5.0, r0=0.1, eps=2.0, ramp=1.0,
            enc_radius=0.15)


def run(N, n_pred, seed):
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

    pred_x  = np.random.uniform(0., 1., n_pred)
    pred_y  = np.random.uniform(0., 1., n_pred)
    pred_vx = np.zeros(n_pred); pred_vy = np.zeros(n_pred)
    pred_angles = np.radians(np.arange(n_pred) * 360.0 / n_pred)

    rb = max(r0, rf)
    phi_t=[]; mind_t=[]
    record_every = 50

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

        # Predator force on prey
        min_d = 1.0
        for k in range(n_pred):
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

        # Predator update (encirclement)
        cx, cy = _periodic_com(x[:N]), _periodic_com(x[N:])
        for k in range(n_pred):
            tx = (cx + PRED['enc_radius'] * np.cos(pred_angles[k])) % 1.
            ty = (cy + PRED['enc_radius'] * np.sin(pred_angles[k])) % 1.
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

        if i >= 1000 and i % record_every == 0:
            phi_t.append(order_parameter(vx, vy))
            mind_t.append(min_d)

    return np.mean(phi_t), np.mean(mind_t)


N_vals      = [350, 700, 1000]
n_pred_vals = [6, 10, 14]
results = {}
for N in N_vals:
    for n_pred in n_pred_vals:
        phis=[]; minds=[]
        for s in range(N_SEEDS):
            phi, mind = run(N, n_pred, s)
            phis.append(phi); minds.append(mind)
            print('  N=%4d n_pred=%2d seed=%d -> Phi=%.3f mind=%.4f' % (
                N, n_pred, s, phi, mind), flush=True)
        results[(N, n_pred)] = dict(phi=np.mean(phis), phi_std=np.std(phis),
                                     mind=np.mean(minds))
        print('N=%4d n_pred=%2d:  Phi=%.3f +/- %.3f  mind=%.4f' % (
            N, n_pred, results[(N, n_pred)]['phi'], results[(N, n_pred)]['phi_std'],
            results[(N, n_pred)]['mind']), flush=True)


fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Encirclement at large N (%d seeds)' % N_SEEDS, fontsize=11)

ax = axes[0]
colors = {350:'steelblue', 700:'darkorange', 1000:'crimson'}
for N in N_vals:
    phis = [results[(N, n)]['phi']     for n in n_pred_vals]
    stds = [results[(N, n)]['phi_std'] for n in n_pred_vals]
    ax.errorbar(n_pred_vals, phis, yerr=stds, fmt='o-', color=colors[N],
                lw=2, capsize=4, label='N=%d' % N)
ax.set_xlabel('n_pred'); ax.set_ylabel('Steady-state Phi')
ax.set_ylim(0, 1.05); ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.set_title('Phi vs n_pred at three flock sizes')
ax.axhline(0.67, ls=':', color='gray', alpha=0.5, label='Finding 15 floor')

ax = axes[1]
for N in N_vals:
    minds = [results[(N, n)]['mind'] for n in n_pred_vals]
    ax.plot(n_pred_vals, minds, 'o-', color=colors[N], lw=2, label='N=%d' % N)
ax.set_xlabel('n_pred'); ax.set_ylabel('Mean min pred-prey distance')
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.set_title('Evasion distance')

plt.tight_layout()
plt.savefig('figures/large_N_1_summary.png', dpi=120)
plt.close()
print('  --> figures/large_N_1_summary.png')

print('\nLarge-N encirclement complete.')
