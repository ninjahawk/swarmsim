# local_escape.py -- does the F70 collective-escape counter survive LOCAL
# (realistic) predator sensing, or does it require the global predator centroid?
#
# F70 gave every prey the GLOBAL predator centroid and showed that a committed
# collective flee (w_escape >= ~alpha) fully defeats predictive encirclement
# (Phi -> 1.0). That assumes each prey knows where all predators are. Real prey
# sense only nearby predators. This experiment replaces the global signal with a
# per-prey LOCAL escape rule: prey i flees the average direction of predators
# within a sensing radius r_sense, and feels no escape force if none are in range.
#
# escape_i = normalize( sum_{k: dist(i,k) <= r_sense} unit(pos_i - pos_k) )
# force_i += w_escape * escape_i
#
# This is the prey-side analog of F19 (predator sensing threshold). w_escape is
# fixed at 2.0 (the F70 value that gave full escape with global sensing). Sweep
# r_sense from 0.05 (very local) to 1.0 (effectively global -> recovers F70).
#
# Question: is there a sensing threshold above which local escape recovers the
# global F70 result? If local sensing works at modest r_sense, the F70 counter is
# robust and realistic; if it needs near-global r_sense, the counter depends on
# unrealistic information and is fragile.

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
N_WARMUP = 1000
N_ITER   = 5000
NPRED    = 6
LEAD_PRED = 2.0
W_ESCAPE = 2.0   # F70 value that gave full escape with global sensing
RSENSE_VALS = [0.05, 0.10, 0.20, 0.40, 1.0]   # 1.0 ~ global (box half-diagonal 0.71)


def run(r_sense, seed):
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
        on = (i >= N_WARMUP)

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

        if on:
            # predator repulsion + LOCAL escape, accumulated over predators
            ex_acc = np.zeros(N); ey_acc = np.zeros(N)
            for k in range(n_pred):
                ddx = _periodic_disp(pred_x[k], x[:N])   # prey - pred (toward prey)
                ddy = _periodic_disp(pred_y[k], x[N:])
                d = np.sqrt(ddx**2 + ddy**2)
                # repulsion (close range)
                mask_p = (d > 0) & (d <= PRED['r0'])
                if mask_p.any():
                    sp_p = PRED['eps'] * (1. - d[mask_p]/PRED['r0'])**1.5 / d[mask_p]
                    fx_total[mask_p] -= sp_p * ddx[mask_p]
                    fy_total[mask_p] -= sp_p * ddy[mask_p]
                # local escape sensing (out to r_sense): sum unit(prey-pred)
                sense = (d > 1e-9) & (d <= r_sense)
                if sense.any():
                    ex_acc[sense] += ddx[sense] / d[sense]
                    ey_acc[sense] += ddy[sense] / d[sense]
            en = np.sqrt(ex_acc**2 + ey_acc**2)
            has_e = en > 1e-9
            fx_total[has_e] += W_ESCAPE * ex_acc[has_e] / en[has_e]
            fy_total[has_e] += W_ESCAPE * ey_acc[has_e] / en[has_e]

        vx += fx_total * dt; vy += fy_total * dt
        x[:N] = (x[:N] + vx*dt) % 1.; x[N:] = (x[N:] + vy*dt) % 1.

        if on:
            cx, cy = _periodic_com(x[:N]), _periodic_com(x[N:])
            vmx = vx.mean(); vmy = vy.mean()
            for k in range(n_pred):
                tx = (cx + PRED['enc_radius']*np.cos(pred_angles[k]) + LEAD_PRED*vmx) % 1.
                ty = (cy + PRED['enc_radius']*np.sin(pred_angles[k]) + LEAD_PRED*vmy) % 1.
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

    arr = np.array(phi_record[500:]) if len(phi_record) > 500 else np.array(phi_record)
    return float(arr.mean()), float(arr.std())


print('Local escape sensing vs predictive encirclement (prey-side analog of F19)')
print('  N=%d  n_pred=%d  w_escape=%.1f (F70 winning value)  lead=%.1f  %d seeds' %
      (BASE['N'], NPRED, W_ESCAPE, LEAD_PRED, N_SEEDS))
print('  Reference: F70 global escape (w=2) Phi=1.000; no escape (F66) Phi=0.530\n')
results = {}
for rs in RSENSE_VALS:
    phis = []; stds = []
    for s in range(N_SEEDS):
        m, sd = run(rs, s)
        phis.append(m); stds.append(sd)
    results[rs] = (np.mean(phis), np.std(phis), np.mean(stds))
    print('  r_sense=%.2f  Phi=%.3f +/- %.3f  intra-std=%.3f' %
          (rs, np.mean(phis), np.std(phis), np.mean(stds)))

# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
rarr = np.array(RSENSE_VALS)
phis_arr = np.array([results[r][0] for r in RSENSE_VALS])
errs_arr = np.array([results[r][1] for r in RSENSE_VALS])
ax.errorbar(rarr, phis_arr, yerr=errs_arr, marker='o', capsize=4, lw=2, color='teal')
ax.axhline(1.000, ls='--', color='teal', alpha=0.5, label='F70 global escape (w=2)')
ax.axhline(0.530, ls='--', color='crimson', alpha=0.5, label='F66 no escape')
ax.set_xlabel('prey escape sensing radius r_sense')
ax.set_ylabel('mean Phi during attack')
ax.set_title('Local escape sensing vs predictive encirclement '
             '(w_escape=%.1f, n_pred=%d, %d seeds)' % (W_ESCAPE, NPRED, N_SEEDS), fontsize=10)
ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('figures/local_escape_1.png', dpi=120)
plt.close()
print('\n  --> figures/local_escape_1.png')
print('\nLocal escape sensing analysis complete.')
