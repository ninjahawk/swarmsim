# predictive_encirclement.py -- can predators deepen encirclement disruption by
# ANTICIPATING the flock's motion rather than targeting the current CoM?
#
# F11 (naive predators co-localize at CoM, helping the flock) and F14 (fixed-angle
# encirclement disrupts to Phi=0.77 at n_pred=6) used predators that targeted
# either the flock's current CoM or fixed compass offsets from it. F33 showed the
# FLOCK does not steer toward gaps (no global escape-route detection). The natural
# adversarial question is whether PREDATORS can detect and exploit the flock's
# escape direction. The simplest predator intelligence is anticipation: target
# CoM + lead_time * v_mean, with v_mean the flock's mean velocity.
#
# At lead_time = 0 this reproduces F14 (Phi ~ 0.77 at n_pred=6, R_enc=0.15).
# As lead_time grows the encirclement ring is placed AHEAD of the flock, so the
# flock encounters it sooner and the predators in its heading direction are now
# directly in its path. Prediction: anticipation should deepen disruption below
# the F14/F35 floor, with the strongest effect at lead_time ~ R_enc / v_mean
# (the time the flock would take to traverse one encirclement radius).
#
# Sweep lead_time in [0, 0.5, 1, 2, 5, 10] tu. n_pred=6, R_enc=0.15, slow-prey
# regime (v0=0.02, ramp=0.1) to match the legacy predator findings.

import os
import numpy as np
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from flocking import buffer as buf_fn, order_parameter
from model import _periodic_com, _periodic_disp

os.makedirs('figures', exist_ok=True)
N_SEEDS = 4

BASE = dict(N=350, r0=0.005, eps=0.1, rf=0.1,
            alpha=1.0, v0=0.02, mu=10.0, ramp=0.1, dt=0.01)
PRED = dict(v0=0.05, mu=10.0, alpha=5.0, r0=0.1, eps=2.0, ramp=1.0,
            enc_radius=0.15)
N_WARMUP = 1000   # flock equilibrates without predators first
N_ITER   = 5000   # 4000 steps of encirclement

NPRED = 6
LEAD_TIMES = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]


def run(lead_time, seed):
    rng = np.random.RandomState(seed)
    p = BASE.copy()
    N = p['N']; dt = p['dt']
    r0, eps, rf = p['r0'], p['eps'], p['rf']
    v0, mu = p['v0'], p['mu']

    x  = np.zeros(2*N)
    x[:N] = rng.uniform(0., 1., N); x[N:] = rng.uniform(0., 1., N)
    vx = rng.uniform(-1., 1., N) * v0
    vy = rng.uniform(-1., 1., N) * v0

    n_pred = NPRED
    pred_x  = rng.uniform(0., 1., n_pred)
    pred_y  = rng.uniform(0., 1., n_pred)
    pred_vx = np.zeros(n_pred); pred_vy = np.zeros(n_pred)
    pred_angles = np.radians(np.arange(n_pred) * 360.0 / n_pred)

    rb = max(r0, rf)
    phi_record = []

    for i in range(N_ITER):
        encirclement_on = (i >= N_WARMUP)

        nb, xb, yb, vxb, vyb = buf_fn(rb, x, vx, vy, N)
        dx = xb[:nb][np.newaxis, :] - x[:N][:, np.newaxis]
        dy = yb[:nb][np.newaxis, :] - x[N:][:, np.newaxis]
        d2 = dx**2 + dy**2
        not_self = np.ones((N, nb), dtype=bool)
        idx = np.arange(min(N, nb)); not_self[idx, idx] = False

        flock_mask = (d2 <= rf**2) & not_self
        flx = np.where(flock_mask, vxb[:nb], 0.).sum(axis=1)
        fly = np.where(flock_mask, vyb[:nb], 0.).sum(axis=1)
        nfl = flock_mask.sum(axis=1)
        nrm = np.sqrt(flx**2 + fly**2); nrm[nfl == 0] = 1.
        flockx = p['alpha'] * flx / nrm; flocky = p['alpha'] * fly / nrm

        rep_mask = (d2 <= (2*r0)**2) & not_self & (d2 > 0)
        d_safe   = np.where(rep_mask, np.sqrt(d2), 1.)
        base_r   = np.where(rep_mask, 1. - d_safe/(2.*r0), 0.)
        strength = np.where(rep_mask, eps * base_r**1.5 / d_safe, 0.)
        repx = (-strength * dx).sum(axis=1); repy = (-strength * dy).sum(axis=1)

        vnorm = np.sqrt(vx**2 + vy**2)
        vnorms = np.where(vnorm == 0, 1., vnorm)
        fpropx = mu * (v0 - vnorm) * vx / vnorms
        fpropy = mu * (v0 - vnorm) * vy / vnorms

        frandx = p['ramp'] * rng.uniform(-1., 1., N)
        frandy = p['ramp'] * rng.uniform(-1., 1., N)

        fx_total = flockx + repx + fpropx + frandx
        fy_total = flocky + repy + fpropy + frandy

        if encirclement_on:
            for k in range(n_pred):
                ddx = _periodic_disp(pred_x[k], x[:N])
                ddy = _periodic_disp(pred_y[k], x[N:])
                d = np.sqrt(ddx**2 + ddy**2)
                mask_p = (d > 0) & (d <= PRED['r0'])
                if mask_p.any():
                    sp_p = PRED['eps'] * (1. - d[mask_p]/PRED['r0'])**1.5 / d[mask_p]
                    fx_total[mask_p] -= sp_p * ddx[mask_p]
                    fy_total[mask_p] -= sp_p * ddy[mask_p]

        vx += fx_total * dt; vy += fy_total * dt
        x[:N] = (x[:N] + vx*dt) % 1.; x[N:] = (x[N:] + vy*dt) % 1.

        if encirclement_on:
            cx, cy = _periodic_com(x[:N]), _periodic_com(x[N:])
            # mean velocity (predictive lead term)
            vmx = vx.mean(); vmy = vy.mean()
            for k in range(n_pred):
                tx = (cx + PRED['enc_radius']*np.cos(pred_angles[k]) + lead_time*vmx) % 1.
                ty = (cy + PRED['enc_radius']*np.sin(pred_angles[k]) + lead_time*vmy) % 1.
                ddx = _periodic_disp(tx, pred_x[k]); ddy = _periodic_disp(ty, pred_y[k])
                dist = np.sqrt(ddx**2 + ddy**2)
                if dist > 0:
                    ddx /= dist; ddy /= dist
                sp_p = np.sqrt(pred_vx[k]**2 + pred_vy[k]**2)
                if sp_p > 0:
                    pfx = PRED['alpha']*ddx + PRED['mu']*(PRED['v0']-sp_p)*pred_vx[k]/sp_p
                    pfy = PRED['alpha']*ddy + PRED['mu']*(PRED['v0']-sp_p)*pred_vy[k]/sp_p
                else:
                    pfx = PRED['alpha']*ddx; pfy = PRED['alpha']*ddy
                pfx += PRED['ramp']*rng.uniform(-1., 1.); pfy += PRED['ramp']*rng.uniform(-1., 1.)
                pred_vx[k] += pfx*dt; pred_vy[k] += pfy*dt
                pred_x[k] = (pred_x[k] + pred_vx[k]*dt) % 1.
                pred_y[k] = (pred_y[k] + pred_vy[k]*dt) % 1.

            phi_record.append(order_parameter(vx, vy))

    # average Phi over attack phase, dropping first 500 transient steps
    arr = np.array(phi_record[500:]) if len(phi_record) > 500 else np.array(phi_record)
    return float(arr.mean()), float(arr.std())


print('Predictive encirclement -- does anticipating the flock motion deepen F14 disruption?')
print('  N=%d  n_pred=%d  R_enc=%.2f  v0_prey=%.2f  %d seeds' %
      (BASE['N'], NPRED, PRED['enc_radius'], BASE['v0'], N_SEEDS))
print('  Reference: F14 (lead=0) gave Phi~0.77 at n_pred=6\n')

results = {}
for lt in LEAD_TIMES:
    phis = []; stds = []
    for s in range(N_SEEDS):
        m, sd = run(lt, s)
        phis.append(m); stds.append(sd)
    results[lt] = (np.mean(phis), np.std(phis), np.mean(stds))
    print('  lead_time=%4.1f tu   mean Phi=%.3f +/- %.3f (across seeds)   intra-run std=%.3f' %
          (lt, np.mean(phis), np.std(phis), np.mean(stds)))

best_lt = min(LEAD_TIMES, key=lambda L: results[L][0])
print('\n  Best lead_time = %.1f tu, Phi=%.3f (vs lead=0 Phi=%.3f)' %
      (best_lt, results[best_lt][0], results[0.0][0]))

# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle('Predictive encirclement: predators anticipate the flock by lead_time * v_mean '
             '(n_pred=%d, R_enc=%.2f, %d seeds)' % (NPRED, PRED['enc_radius'], N_SEEDS),
             fontsize=10)
lts = np.array(LEAD_TIMES)
phis = np.array([results[L][0] for L in LEAD_TIMES])
errs = np.array([results[L][1] for L in LEAD_TIMES])
ax.errorbar(lts, phis, yerr=errs, marker='o', capsize=4, lw=2, color='crimson')
ax.axhline(results[0.0][0], ls='--', color='gray', alpha=0.7, label='lead=0 (F14 baseline)')
ax.set_xlabel('lead_time (tu)'); ax.set_ylabel('mean Phi during attack')
ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('figures/predictive_encirclement_1.png', dpi=120)
plt.close()
print('\n  --> figures/predictive_encirclement_1.png')
print('\nPredictive encirclement analysis complete.')
