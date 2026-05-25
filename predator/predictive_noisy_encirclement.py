# predictive_noisy_encirclement.py -- informational stress test for the F66
# predictive encirclement mechanism: how robust is it to noisy v_mean estimates?
#
# F66: predictive encirclement (predators target CoM + lead*v_mean) deepens F14
# to Phi=0.530 at lead=2 tu. F67: combining with adaptive R_enc adds nothing;
# placement is the dominant lever. F67 closed by noting the remaining questions
# are predator-side INFORMATIONAL: noisy v_mean, delayed updates, partial
# visibility. This experiment is the noisy-estimate version -- the F60 analog
# for predator intelligence.
#
# Setup: same as predictive_encirclement.py at lead_time = 2 tu, but the
# predators see v_mean_hat = v_mean + N(0, sigma_obs) per step (independent
# Gaussian noise on each component). At sigma_obs = 0 reproduces F66 (Phi~0.53).
# As sigma_obs grows the predictive signal is increasingly drowned in noise;
# the policy should degrade to ~F14 baseline (Phi~0.83) when sigma_obs >> |v_mean|.
#
# |v_mean| ~ v_eq = v0 + alpha/mu = 0.12 (Finding 1). Sweep sigma_obs in
# absolute velocity units: 0.0, 0.03, 0.06, 0.12, 0.24, 0.48 (= 0, 25%, 50%,
# 100%, 200%, 400% of |v_mean|).
#
# Prediction (analogous to F60): policy robust up to sigma_obs ~ |v_mean|,
# then graceful degradation toward F14 baseline.

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
LEAD_PRED = 2.0   # F66 optimum

SIGMA_VALS = [0.0, 0.03, 0.06, 0.12, 0.24, 0.48]   # noise on v_mean estimate


def run(sigma_obs, seed):
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
            # noisy estimate of v_mean
            vmx_true = vx.mean(); vmy_true = vy.mean()
            if sigma_obs > 0:
                vmx_hat = vmx_true + sigma_obs * rng.standard_normal()
                vmy_hat = vmy_true + sigma_obs * rng.standard_normal()
            else:
                vmx_hat = vmx_true; vmy_hat = vmy_true
            for k in range(n_pred):
                tx = (cx + PRED['enc_radius']*np.cos(pred_angles[k]) + LEAD_PRED*vmx_hat) % 1.
                ty = (cy + PRED['enc_radius']*np.sin(pred_angles[k]) + LEAD_PRED*vmy_hat) % 1.
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


print('Predictive encirclement under noisy v_mean estimate -- F60 analog for predator intel')
print('  N=%d  n_pred=%d  R_enc=%.2f  lead=%.1f tu  %d seeds' %
      (BASE['N'], NPRED, PRED['enc_radius'], LEAD_PRED, N_SEEDS))
print('  Reference: F66 (sigma_obs=0) Phi=0.530; F14 baseline (no predictive) Phi=0.825')
print('  v_mean magnitude ~ v_eq = 0.12; sigma_obs values span 0 to 400%% of |v_mean|\n')

results = {}
for sig in SIGMA_VALS:
    phis = []; stds = []
    for s in range(N_SEEDS):
        m, sd = run(sig, s)
        phis.append(m); stds.append(sd)
    results[sig] = (np.mean(phis), np.std(phis), np.mean(stds))
    print('  sigma_obs=%.2f  (%3d%% of v_mean)  Phi=%.3f +/- %.3f  intra-std=%.3f' %
          (sig, int(round(100*sig/0.12)), np.mean(phis), np.std(phis), np.mean(stds)))

# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
sig_arr = np.array(SIGMA_VALS)
phis_arr = np.array([results[s][0] for s in SIGMA_VALS])
errs_arr = np.array([results[s][1] for s in SIGMA_VALS])
ax.errorbar(sig_arr, phis_arr, yerr=errs_arr, marker='o', capsize=4, lw=2, color='crimson')
ax.axhline(results[0.0][0], ls='--', color='crimson', alpha=0.5, label='sigma_obs=0 (F66, predictive)')
ax.axhline(0.825, ls='--', color='gray', alpha=0.5, label='F14 baseline (no predictive)')
ax.axvline(0.12, ls=':', color='black', alpha=0.5, label='|v_mean| ~ 0.12')
ax.set_xlabel('sigma_obs (absolute velocity units)')
ax.set_ylabel('mean Phi during attack')
ax.set_title('Predictive encirclement robustness to noisy v_mean estimate '
             '(lead=%.1f tu, n_pred=%d, %d seeds)' % (LEAD_PRED, NPRED, N_SEEDS), fontsize=10)
ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('figures/predictive_noisy_encirclement_1.png', dpi=120)
plt.close()
print('\n  --> figures/predictive_noisy_encirclement_1.png')
print('\nPredictive encirclement noise-tolerance analysis complete.')
