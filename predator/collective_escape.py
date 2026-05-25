# collective_escape.py -- the arms-race step: does PREY collective escape
# intelligence counter PREDATOR predictive encirclement (F66)?
#
# F66-F69 gave predators a global signal (the flock's mean velocity v_mean) and
# showed predictive placement deepens disruption (Phi 0.83 -> 0.53). The
# symmetric question is whether prey can use the dual global signal -- the
# predator centroid -- to flee collectively. F33 showed the flock cannot detect
# escape directions on its own. Under SYMMETRIC encirclement the predator
# centroid coincides with the flock CoM, so there is no escape gradient. But
# under PREDICTIVE encirclement (F66) the predators mass AHEAD of the flock, so
# the predator centroid is displaced in the heading direction and "flee the
# predator centroid" becomes a well-defined backward escape.
#
# Collective escape intelligence: every prey adds a force w_escape * e_hat where
# e_hat = unit vector from the predator centroid toward the flock CoM (i.e. away
# from where predators are massed). This is the prey-side dual of v_mean: a
# global signal shared by the whole flock. Sweep w_escape from 0 (F66) upward.
#
# Predator: F66 predictive encirclement at lead=2 tu (the hardest predator found).
# Question: does escape intelligence restore coherence (raise Phi back toward 1)?
# If yes, prey global intelligence counters predator global intelligence and the
# arms race favors whoever holds the global signal. If no (Phi stays low), the
# predator's forward projection beats the prey's centroid-flee even when both
# have global information.

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
LEAD_PRED = 2.0   # F66 hardest predator
W_ESCAPE_VALS = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0]


def run(w_escape, seed):
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
            # predator repulsion on prey
            for k in range(n_pred):
                ddx = _periodic_disp(pred_x[k], x[:N])
                ddy = _periodic_disp(pred_y[k], x[N:])
                d = np.sqrt(ddx**2 + ddy**2)
                mask_p = (d > 0) & (d <= PRED['r0'])
                if mask_p.any():
                    sp_p = PRED['eps'] * (1. - d[mask_p]/PRED['r0'])**1.5 / d[mask_p]
                    fx_total[mask_p] -= sp_p * ddx[mask_p]
                    fy_total[mask_p] -= sp_p * ddy[mask_p]

            # collective escape intelligence: flee the predator centroid
            if w_escape > 0:
                cx0, cy0 = _periodic_com(x[:N]), _periodic_com(x[N:])
                pcx = _periodic_com(pred_x); pcy = _periodic_com(pred_y)
                ex = _periodic_disp(cx0, pcx)  # from predator centroid toward CoM
                ey = _periodic_disp(cy0, pcy)
                en = np.sqrt(ex**2 + ey**2)
                if en > 1e-9:
                    ex /= en; ey /= en
                    fx_total += w_escape * ex
                    fy_total += w_escape * ey

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


print('Collective escape intelligence vs predictive encirclement (F66 lead=2 tu)')
print('  N=%d  n_pred=%d  %d seeds  prey flee the predator centroid with weight w_escape' %
      (BASE['N'], NPRED, N_SEEDS))
print('  Reference: w=0 is F66 (Phi=0.530); prey alignment force alpha=1.0 sets the scale\n')
results = {}
for w in W_ESCAPE_VALS:
    phis = []; stds = []
    for s in range(N_SEEDS):
        m, sd = run(w, s)
        phis.append(m); stds.append(sd)
    results[w] = (np.mean(phis), np.std(phis), np.mean(stds))
    print('  w_escape=%.2f  Phi=%.3f +/- %.3f  intra-std=%.3f' %
          (w, np.mean(phis), np.std(phis), np.mean(stds)))

best = max(W_ESCAPE_VALS, key=lambda w: results[w][0])
print('\n  Highest Phi at w_escape=%.2f (Phi=%.3f vs w=0 Phi=%.3f)' %
      (best, results[best][0], results[0.0][0]))

# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
warr = np.array(W_ESCAPE_VALS)
phis_arr = np.array([results[w][0] for w in W_ESCAPE_VALS])
errs_arr = np.array([results[w][1] for w in W_ESCAPE_VALS])
ax.errorbar(warr, phis_arr, yerr=errs_arr, marker='o', capsize=4, lw=2, color='teal')
ax.axhline(results[0.0][0], ls='--', color='crimson', alpha=0.6,
           label='w=0 (F66 predictive, no escape intel)')
ax.axhline(0.825, ls='--', color='gray', alpha=0.5, label='F14 fixed-encirclement baseline')
ax.set_xlabel('collective escape weight w_escape (prey alpha=1.0)')
ax.set_ylabel('mean Phi during attack')
ax.set_title('Prey collective escape intelligence vs predictive encirclement '
             '(n_pred=%d, %d seeds)' % (NPRED, N_SEEDS), fontsize=10)
ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('figures/collective_escape_1.png', dpi=120)
plt.close()
print('\n  --> figures/collective_escape_1.png')
print('\nCollective escape intelligence analysis complete.')
