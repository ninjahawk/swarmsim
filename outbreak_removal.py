# outbreak_removal.py -- Does contagion persist after compression is removed?
#
# Finding 22 showed encirclement damage is reversible: sub-flocks reunite within
# ~10 time units of predator removal.  Finding 25 showed SIS contagion has a
# threshold at beta/gamma ~ 1.  Finding 29 showed encirclement shifts the threshold
# by ~4%.
#
# Hypothesis: if we run an outbreak that is AMPLIFIED into supercriticality by
# encirclement (i.e., bare beta is sub-threshold but encirclement-amplified beta
# is super-threshold), removing the predators mid-run should let the contagion
# die out as the local <k> drops.  This combines kinematic reversibility with
# epidemiological reversibility.
#
# Three-phase protocol:
#   Phase 1 (0..n_warmup):           pure flock, no contagion
#   Phase 2 (n_warmup..n_remove):    encirclement + SIS contagion (induce outbreak)
#   Phase 3 (n_remove..n_total):     predators removed, contagion left alone
#
# We pick beta = 1.5, gamma = 2.0 (beta/gamma = 0.75, just below bare threshold of
# 0.96 from Finding 29) -- contagion should be on the edge.

import os
import numpy as np
import matplotlib.pyplot as plt
from flocking import buffer as buf_fn, order_parameter
from model import _periodic_com, _periodic_disp

os.makedirs('figures', exist_ok=True)
N_SEEDS = 5

BASE = dict(
    N=350, r0=0.005, eps=0.1, rf=0.1,
    alpha=1.0, v0=0.02, mu=10.0, ramp=0.1, dt=0.01,
)
PANIC_ALPHA = 0.1
PANIC_RAMP  = 10.0
R_CONT      = 0.05
BETA        = 1.5
GAMMA       = 2.0

PRED = dict(v0=0.05, mu=10.0, alpha=5.0, r0=0.1, eps=2.0, ramp=1.0,
            enc_radius=0.15)

N_WARMUP = 1000
N_REMOVE = 5000   # 4000 steps of attack
N_TOTAL  = 10000  # 5000 steps of recovery


def run(seed):
    np.random.seed(seed)
    p = BASE.copy()
    N = p['N']; dt = p['dt']
    r0, eps, rf = p['r0'], p['eps'], p['rf']
    v0, mu = p['v0'], p['mu']

    x  = np.zeros(2*N)
    x[:N] = np.random.uniform(0., 1., N)
    x[N:] = np.random.uniform(0., 1., N)
    vx = np.random.uniform(-1., 1., N) * v0
    vy = np.random.uniform(-1., 1., N) * v0

    is_panicked = np.zeros(N, dtype=bool)

    n_pred = 6
    pred_x  = np.random.uniform(0., 1., n_pred)
    pred_y  = np.random.uniform(0., 1., n_pred)
    pred_vx = np.zeros(n_pred); pred_vy = np.zeros(n_pred)
    pred_angles = np.radians(np.arange(n_pred) * 60.0)

    p_recover = 1. - np.exp(-GAMMA * dt)
    rb = max(r0, rf, R_CONT)

    f_t=[]; phi_t=[]; t_t=[]
    record_every = 50

    for i in range(N_TOTAL):
        encirclement_on = (i >= N_WARMUP) and (i < N_REMOVE)
        contagion_on    = (i >= N_WARMUP)

        # seed initial panic at start of attack
        if i == N_WARMUP:
            n0 = max(1, round(0.05 * N))
            idx0 = np.random.choice(N, size=n0, replace=False)
            is_panicked[idx0] = True

        nb, xb, yb, vxb, vyb = buf_fn(rb, x, vx, vy, N)
        dx = xb[:nb][np.newaxis, :] - x[:N][:, np.newaxis]
        dy = yb[:nb][np.newaxis, :] - x[N:][:, np.newaxis]
        d2 = dx**2 + dy**2

        not_self = np.ones((N, nb), dtype=bool)
        idx = np.arange(min(N, nb))
        not_self[idx, idx] = False

        alpha_arr = np.where(is_panicked, PANIC_ALPHA, p['alpha'])
        ramp_arr  = np.where(is_panicked, PANIC_RAMP,  p['ramp'])

        flock_mask = (d2 <= rf**2) & not_self
        flx = np.where(flock_mask, vxb[:nb], 0.).sum(axis=1)
        fly = np.where(flock_mask, vyb[:nb], 0.).sum(axis=1)
        nfl = flock_mask.sum(axis=1)
        nrm = np.sqrt(flx**2 + fly**2)
        nrm[nfl == 0] = 1.
        flockx = alpha_arr * flx / nrm
        flocky = alpha_arr * fly / nrm

        rep_mask = (d2 <= (2*r0)**2) & not_self & (d2 > 0)
        d_safe   = np.where(rep_mask, np.sqrt(d2), 1.)
        base_r   = np.where(rep_mask, 1. - d_safe/(2.*r0), 0.)
        strength = np.where(rep_mask, eps * base_r**1.5 / d_safe, 0.)
        repx = (-strength * dx).sum(axis=1)
        repy = (-strength * dy).sum(axis=1)

        vnorm  = np.sqrt(vx**2 + vy**2)
        vnorms = np.where(vnorm == 0, 1., vnorm)
        fpropx = mu * (v0 - vnorm) * vx / vnorms
        fpropy = mu * (v0 - vnorm) * vy / vnorms

        frandx = ramp_arr * np.random.uniform(-1., 1., N)
        frandy = ramp_arr * np.random.uniform(-1., 1., N)

        fx_total = flockx + repx + fpropx + frandx
        fy_total = flocky + repy + fpropy + frandy

        if encirclement_on:
            for k in range(n_pred):
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

        if encirclement_on:
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

        if contagion_on and is_panicked.any() and (~is_panicked).any():
            real_dx = x[:N][np.newaxis, :] - x[:N][:, np.newaxis]
            real_dy = x[N:][np.newaxis, :] - x[N:][:, np.newaxis]
            real_dx -= np.round(real_dx); real_dy -= np.round(real_dy)
            rd2 = real_dx**2 + real_dy**2
            within = (rd2 <= R_CONT**2) & (rd2 > 0)
            k_arr = within @ is_panicked.astype(np.int32)
            calm_idx = np.where(~is_panicked)[0]
            if calm_idx.size:
                k_calm = k_arr[calm_idx]
                p_trans = 1. - np.exp(-BETA * k_calm * dt)
                r = np.random.uniform(0., 1., calm_idx.size)
                flipped = calm_idx[r < p_trans]
                if flipped.size:
                    is_panicked[flipped] = True
        if contagion_on and is_panicked.any():
            panic_idx = np.where(is_panicked)[0]
            r = np.random.uniform(0., 1., panic_idx.size)
            recovered = panic_idx[r < p_recover]
            if recovered.size:
                is_panicked[recovered] = False

        if i % record_every == 0:
            f_t.append(is_panicked.mean())
            phi_t.append(order_parameter(vx, vy))
            t_t.append(i * dt)

    return np.array(t_t), np.array(phi_t), np.array(f_t)


print('Outbreak-then-removal: encirclement+SIS for steps %d..%d, removed %d..%d' %
      (N_WARMUP, N_REMOVE, N_REMOVE, N_TOTAL))
print('beta=%.2f gamma=%.2f (beta/gamma=%.2f), %d seeds' % (BETA, GAMMA, BETA/GAMMA, N_SEEDS))

all_runs = []
for s in range(N_SEEDS):
    print('  seed %d...' % s, flush=True)
    all_runs.append(run(s))

t = all_runs[0][0]
phi_arr = np.array([r[1] for r in all_runs])
f_arr   = np.array([r[2] for r in all_runs])

dt_val = BASE['dt']
t_attack = N_WARMUP * dt_val
t_remove = N_REMOVE * dt_val

def window(arr, t0, t1):
    mask = (t >= t0) & (t < t1)
    return arr[:, mask].mean(axis=1)

f_during  = window(f_arr,  t_remove - 5.0, t_remove)
f_post    = window(f_arr,  t[-1] - 10.0, t[-1] + 1)
phi_during= window(phi_arr,t_remove - 5.0, t_remove)
phi_post  = window(phi_arr,t[-1] - 10.0, t[-1] + 1)

print('\n=== Phase summary (mean +/- std across %d seeds) ===' % N_SEEDS)
print('During attack (t < %.1f): f = %.3f +/- %.3f   Phi = %.3f +/- %.3f' %
      (t_remove, f_during.mean(), f_during.std(), phi_during.mean(), phi_during.std()))
print('Post-removal (t > %.1f):  f = %.3f +/- %.3f   Phi = %.3f +/- %.3f' %
      (t[-1] - 10, f_post.mean(), f_post.std(), phi_post.mean(), phi_post.std()))

# Plot
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax = axes[0]
ax.axvspan(t_attack, t_remove, color='red', alpha=0.10, label='encirclement active')
for r in all_runs:
    ax.plot(r[0], r[1], alpha=0.3, lw=0.6)
ax.plot(t, phi_arr.mean(0), color='navy', lw=2, label='mean Phi')
ax.set_ylabel('Phi'); ax.set_ylim(0, 1.05)
ax.set_title('Outbreak-then-removal: beta=%.2f gamma=%.2f' % (BETA, GAMMA))
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1]
ax.axvspan(t_attack, t_remove, color='red', alpha=0.10)
for r in all_runs:
    ax.plot(r[0], r[2], alpha=0.3, lw=0.6)
ax.plot(t, f_arr.mean(0), color='crimson', lw=2, label='mean f')
ax.set_ylabel('Panic fraction f')
ax.set_xlabel('Time')
ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('figures/outbreak_removal_1.png', dpi=120)
plt.close()
print('  --> figures/outbreak_removal_1.png')
