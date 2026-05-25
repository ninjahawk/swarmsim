# predictive_adaptive_encirclement.py -- do the two predator-side adaptations
# (predictive position from F66, adaptive radius from F35) COMPOSE?
#
# F14: fixed encirclement, n_pred=6, R_enc=0.15 -> Phi ~ 0.77-0.83
# F35: adaptive R_enc = 0.5 * live_Rg -> Phi ~ 0.713 (modest improvement)
# F66: predictive lead = 2 tu, fixed R_enc -> Phi = 0.530 (best so far)
# F67 (this): predictive lead = 2 tu AND adaptive R_enc = 0.5 * live_Rg -> ?
#
# Both adaptations operate on different geometric degrees of freedom (one shifts
# the ring centre, the other scales the ring radius). The simplest prediction is
# they compose: combined Phi should be lower than either alone. The alternative
# is interference -- e.g., predictive placement biases the geometry so that
# adaptive R_enc no longer hits its optimum -- in which case combined Phi might
# match the better of the two singletons but not improve further.
#
# Compares four conditions at matched n_pred=6, 4 seeds, attack 4000 steps:
#   fixed-fixed     -- F14 reproduction
#   fixed-adaptive  -- F35 reproduction (R_enc = 0.5 * live_Rg)
#   predictive-fixed-- F66 reproduction (R_enc=0.15, lead=2 tu)
#   predictive-adaptive -- new (R_enc = 0.5*live_Rg AND lead=2 tu)

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
PRED = dict(v0=0.05, mu=10.0, alpha=5.0, r0=0.1, eps=2.0, ramp=1.0)
N_WARMUP = 1000
N_ITER   = 5000
NPRED    = 6
RENC_FIXED = 0.15
RENC_RATIO = 0.5     # adaptive: R_enc = 0.5 * Rg (F35 universal optimum)
LEAD_FIXED = 0.0
LEAD_PRED  = 2.0     # F66 optimum

CONFIGS = [
    ('fixed-fixed',         RENC_FIXED, 'fixed',    LEAD_FIXED),
    ('fixed-adaptive',      None,        'adaptive', LEAD_FIXED),
    ('predictive-fixed',    RENC_FIXED, 'fixed',    LEAD_PRED),
    ('predictive-adaptive', None,        'adaptive', LEAD_PRED),
]


def _periodic_rg(px, py, cx, cy):
    dx = px - cx; dx -= np.round(dx)
    dy = py - cy; dy -= np.round(dy)
    return float(np.sqrt((dx**2 + dy**2).mean()))


def run(renc_fixed, renc_mode, lead_time, seed):
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
            # adaptive R_enc per F35
            if renc_mode == 'adaptive':
                rg = _periodic_rg(x[:N], x[N:], cx, cy)
                renc_now = RENC_RATIO * rg
            else:
                renc_now = renc_fixed
            # predictive lead per F66
            vmx = vx.mean(); vmy = vy.mean()
            for k in range(n_pred):
                tx = (cx + renc_now*np.cos(pred_angles[k]) + lead_time*vmx) % 1.
                ty = (cy + renc_now*np.sin(pred_angles[k]) + lead_time*vmy) % 1.
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


print('Predictive + adaptive encirclement -- do the two predator adaptations compose?')
print('  N=%d  n_pred=%d  %d seeds  lead_predictive=%.1f tu  R_enc/Rg=%.2f (adaptive)' %
      (BASE['N'], NPRED, N_SEEDS, LEAD_PRED, RENC_RATIO))
print()
results = {}
for label, renc_f, renc_m, lt in CONFIGS:
    phis = []; stds = []
    for s in range(N_SEEDS):
        m, sd = run(renc_f, renc_m, lt, s)
        phis.append(m); stds.append(sd)
    results[label] = (np.mean(phis), np.std(phis), np.mean(stds))
    print('  %-22s  Phi=%.3f +/- %.3f  intra-std=%.3f' %
          (label, np.mean(phis), np.std(phis), np.mean(stds)))

best = min(results.keys(), key=lambda L: results[L][0])
print('\n  Best: %s  (Phi=%.3f)' % (best, results[best][0]))

# advantage of combined over each singleton
combined = results['predictive-adaptive'][0]
print('  Combined vs F14 fixed-fixed:        %+.3f' % (results['fixed-fixed'][0] - combined))
print('  Combined vs F35 fixed-adaptive:     %+.3f' % (results['fixed-adaptive'][0] - combined))
print('  Combined vs F66 predictive-fixed:   %+.3f' % (results['predictive-fixed'][0] - combined))

# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
labels = [c[0] for c in CONFIGS]
means  = [results[L][0] for L in labels]
errs   = [results[L][1] for L in labels]
colors = ['gray', 'steelblue', 'crimson', 'darkviolet']
xp = np.arange(len(labels))
ax.bar(xp, means, yerr=errs, capsize=5, color=colors, edgecolor='black', alpha=0.85)
for i, m in enumerate(means):
    ax.text(i, m + 0.03, '%.3f' % m, ha='center', fontsize=9)
ax.set_xticks(xp); ax.set_xticklabels(labels, rotation=12, ha='right')
ax.set_ylabel('mean Phi during attack')
ax.set_ylim(0, 1.05); ax.grid(alpha=0.3, axis='y')
ax.set_title('Do the two predator-side adaptations (predictive position + adaptive radius) compose?\n'
             'n_pred=%d, R_enc fixed=%.2f, R_enc adaptive=%.1f*Rg, lead predictive=%.1f tu, %d seeds' %
             (NPRED, RENC_FIXED, RENC_RATIO, LEAD_PRED, N_SEEDS), fontsize=10)
plt.tight_layout()
plt.savefig('figures/predictive_adaptive_encirclement_1.png', dpi=120)
plt.close()
print('\n  --> figures/predictive_adaptive_encirclement_1.png')
print('\nPredictive + adaptive encirclement analysis complete.')
