# long_encirclement.py -- Long-time steady state under sustained encirclement
#
# Finding 16 showed encirclement divides the flock into coherent sub-flocks
# (n_clusters jumps from 1 to ~4-5, largest_frac drops to ~0.41).
# Finding 22 showed those divisions reunite within ~10 time units of predator
# REMOVAL.  Open question: with predators STILL ACTIVE, does the division
# persist indefinitely, or do sub-flocks eventually re-merge despite ongoing
# pressure?
#
# Long-time encirclement (n_pred=10, R_enc=0.20 -> optimal for N=1000 from Finding 31)
# for 30000 steps (= 300 time units, ~10x longer than typical attack).
# Track n_clusters and largest_frac over time, look for either:
#   - persistent fragmentation (sub-flock identities stable)
#   - intermittent fragmentation (sub-flocks merge and re-split)
#   - eventual reunification (sub-flocks find a configuration that defeats encirclement)

import os
import numpy as np
import matplotlib.pyplot as plt
from flocking import buffer as buf_fn, order_parameter
from model import _periodic_com, _periodic_disp

os.makedirs('figures', exist_ok=True)
N_SEEDS = 4

BASE = dict(
    N=350, r0=0.005, eps=0.1, rf=0.1,
    alpha=1.0, v0=0.02, mu=10.0, ramp=0.1, dt=0.01,
    n_iter=30000,
)
PRED = dict(v0=0.05, mu=10.0, alpha=5.0, r0=0.1, eps=2.0, ramp=1.0,
            enc_radius=0.15)
N_PRED = 10
RECORD_EVERY = 50


def find_clusters(px, py, rf):
    N = len(px)
    dx = px[np.newaxis, :] - px[:, np.newaxis]
    dy = py[np.newaxis, :] - py[:, np.newaxis]
    dx -= np.round(dx); dy -= np.round(dy)
    d2 = dx**2 + dy**2
    adj = (d2 <= rf**2) & (d2 > 0)

    labels = -np.ones(N, dtype=int)
    cur = 0
    for i in range(N):
        if labels[i] >= 0:
            continue
        stack = [i]
        while stack:
            j = stack.pop()
            if labels[j] >= 0:
                continue
            labels[j] = cur
            for k in np.where(adj[j])[0]:
                if labels[k] < 0:
                    stack.append(k)
        cur += 1
    return labels, cur


def run(seed):
    np.random.seed(seed)
    p = BASE.copy()
    N = p['N']; dt = p['dt']; n_iter = p['n_iter']
    r0, eps, rf = p['r0'], p['eps'], p['rf']
    v0, mu = p['v0'], p['mu']

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
    phi_t=[]; n_clust_t=[]; lfrac_t=[]; t_t=[]

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

        frandx = p['ramp'] * np.random.uniform(-1., 1., N)
        frandy = p['ramp'] * np.random.uniform(-1., 1., N)

        fx_total = flockx + repx + fpropx + frandx
        fy_total = flocky + repy + fpropy + frandy

        for k in range(N_PRED):
            ddx = _periodic_disp(pred_x[k], x[:N])
            ddy = _periodic_disp(pred_y[k], x[N:])
            d = np.sqrt(ddx**2 + ddy**2)
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

        if i % RECORD_EVERY == 0:
            phi_t.append(order_parameter(vx, vy))
            labels, n_cl = find_clusters(x[:N], x[N:], rf)
            sizes = np.bincount(labels)
            lfrac_t.append(float(sizes.max() / N))
            n_clust_t.append(n_cl)
            t_t.append(i * dt)

    return np.array(t_t), np.array(phi_t), np.array(n_clust_t), np.array(lfrac_t)


print('Long-time encirclement: n_pred=%d, %d steps = %d time units, %d seeds' %
      (N_PRED, BASE['n_iter'], int(BASE['n_iter']*BASE['dt']), N_SEEDS))

all_runs = []
for s in range(N_SEEDS):
    print('  seed %d ...' % s, flush=True)
    all_runs.append(run(s))

t = all_runs[0][0]
phi_arr   = np.array([r[1] for r in all_runs])
clust_arr = np.array([r[2] for r in all_runs])
lfrac_arr = np.array([r[3] for r in all_runs])


# Steady-state statistics (last 1/3 of run)
ss_start = int(len(t) * 2 / 3)
phi_ss   = phi_arr[:, ss_start:].mean(axis=1)
clust_ss = clust_arr[:, ss_start:].mean(axis=1)
lfrac_ss = lfrac_arr[:, ss_start:].mean(axis=1)

print('\n=== Long-time steady state (last 1/3 = t > %.1f) ===' % t[ss_start])
print('Phi          mean=%.3f +/- %.3f (across seeds)' % (phi_ss.mean(), phi_ss.std()))
print('n_clusters   mean=%.2f +/- %.2f' % (clust_ss.mean(), clust_ss.std()))
print('largest_frac mean=%.3f +/- %.3f' % (lfrac_ss.mean(), lfrac_ss.std()))

# Variance over time to detect intermittent reunifications
phi_temporal_std = phi_arr[:, ss_start:].std(axis=1)
print('Phi temporal-std per seed: %s' % phi_temporal_std)


# Plot
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
ax = axes[0]
for r in all_runs:
    ax.plot(r[0], r[1], alpha=0.3, lw=0.6)
ax.plot(t, phi_arr.mean(0), color='navy', lw=2, label='mean')
ax.axhline(0.667, ls=':', color='gray', alpha=0.5, label='Finding 16 short-time floor')
ax.set_ylabel('Phi'); ax.set_ylim(0, 1.05)
ax.set_title('Long-time encirclement: 30000 steps, %d seeds' % N_SEEDS)
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1]
for r in all_runs:
    ax.plot(r[0], r[2], alpha=0.3, lw=0.6)
ax.plot(t, clust_arr.mean(0), color='darkgreen', lw=2)
ax.set_ylabel('n_clusters')
ax.grid(alpha=0.3)

ax = axes[2]
for r in all_runs:
    ax.plot(r[0], r[3], alpha=0.3, lw=0.6)
ax.plot(t, lfrac_arr.mean(0), color='chocolate', lw=2)
ax.set_ylabel('largest cluster fraction')
ax.set_xlabel('Time')
ax.set_ylim(0, 1.05)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/long_encircle_1.png', dpi=120)
plt.close()
print('  --> figures/long_encircle_1.png')
