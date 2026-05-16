# critical_shift.py -- Does encirclement shift the SIS epidemic threshold?
#
# Finding 26 showed that at beta/gamma = 0.33 (well sub-threshold), encirclement
# amplifies but does not tip contagion.  Open question: how close to threshold
# must contagion be for encirclement's compression-amplification to MATTER?
#
# Method: sweep beta at fixed gamma=2.0 with and without encirclement.  Find the
# apparent threshold beta_c (above which f_ss > 0.5) under both conditions.
# Encirclement's effect = the leftward shift of beta_c.
#
# Five beta values bracketing the no-encirclement threshold:
#   beta = 1.0, 1.5, 2.0, 2.5, 3.0   (beta/gamma = 0.50, 0.75, 1.00, 1.25, 1.50)

import os
import numpy as np
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from flocking import buffer as buf_fn, order_parameter
from model import _periodic_com, _periodic_disp

os.makedirs('figures', exist_ok=True)
N_SEEDS = 5

BASE = dict(
    N=350, r0=0.005, eps=0.1, rf=0.1,
    alpha=1.0, v0=0.02, mu=10.0, ramp=0.1, dt=0.01,
    n_iter=5000,
)
PANIC_ALPHA = 0.1
PANIC_RAMP  = 10.0
R_CONT      = 0.05
GAMMA       = 2.0
PRED = dict(v0=0.05, mu=10.0, alpha=5.0, r0=0.1, eps=2.0, ramp=1.0,
            enc_radius=0.15)


def run(beta, use_encircle, seed):
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

    is_panicked = np.zeros(N, dtype=bool)
    n0 = max(1, round(0.05 * N))
    idx0 = np.random.choice(N, size=n0, replace=False)
    is_panicked[idx0] = True

    if use_encircle:
        n_pred = 6
        pred_x  = np.random.uniform(0., 1., n_pred)
        pred_y  = np.random.uniform(0., 1., n_pred)
        pred_vx = np.zeros(n_pred); pred_vy = np.zeros(n_pred)
        pred_angles = np.radians(np.arange(n_pred) * 60.0)
    else:
        n_pred = 0

    p_recover_per_step = 1. - np.exp(-GAMMA * dt)
    rb = max(r0, rf, R_CONT)

    f_t = []
    for i in range(n_iter):
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

        if use_encircle:
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

        if use_encircle:
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

        if is_panicked.any() and (~is_panicked).any():
            real_dx = x[:N][np.newaxis, :] - x[:N][:, np.newaxis]
            real_dy = x[N:][np.newaxis, :] - x[N:][:, np.newaxis]
            real_dx -= np.round(real_dx); real_dy -= np.round(real_dy)
            rd2 = real_dx**2 + real_dy**2
            within = (rd2 <= R_CONT**2) & (rd2 > 0)
            k_arr = within @ is_panicked.astype(np.int32)
            calm_idx = np.where(~is_panicked)[0]
            if calm_idx.size:
                k_calm = k_arr[calm_idx]
                p_trans = 1. - np.exp(-beta * k_calm * dt)
                r = np.random.uniform(0., 1., calm_idx.size)
                flipped = calm_idx[r < p_trans]
                if flipped.size:
                    is_panicked[flipped] = True
        if is_panicked.any():
            panic_idx = np.where(is_panicked)[0]
            r = np.random.uniform(0., 1., panic_idx.size)
            recovered = panic_idx[r < p_recover_per_step]
            if recovered.size:
                is_panicked[recovered] = False

        if i % 50 == 0:
            f_t.append(is_panicked.mean())

    # steady state = last 20% of recorded points
    f_ss = np.mean(f_t[-len(f_t)//5:])
    return f_ss


print('Critical shift: SIS gamma=2.0, beta sweep with vs without encirclement')
print('  %d seeds' % N_SEEDS)
beta_vals = [1.0, 1.5, 2.0, 2.5, 3.0]
results = {True: {}, False: {}}
for ue in (False, True):
    for b in beta_vals:
        fs = []
        for s in range(N_SEEDS):
            fs.append(run(b, ue, s))
        results[ue][b] = (np.mean(fs), np.std(fs))
        print('  encircle=%s  beta=%.2f  f_ss=%.3f +/- %.3f' % (
            ue, b, results[ue][b][0], results[ue][b][1]))

# Compute apparent threshold (linear interp to f_ss = 0.5)
def threshold(curve, betas):
    fs = [curve[b][0] for b in betas]
    for i in range(len(betas)-1):
        if fs[i] < 0.5 <= fs[i+1]:
            return betas[i] + (0.5 - fs[i])/(fs[i+1]-fs[i]) * (betas[i+1]-betas[i])
    return None

bc_no  = threshold(results[False], beta_vals)
bc_yes = threshold(results[True],  beta_vals)
print('\nApparent thresholds:')
print('  without encirclement: beta_c = %s' % (bc_no,))
print('  with    encirclement: beta_c = %s' % (bc_yes,))
if bc_no and bc_yes:
    print('  shift = %.3f  (=%.1f%% of bc_no)' % (
        bc_no - bc_yes, 100*(bc_no - bc_yes)/bc_no))

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
for ue, color, label in [(False, 'steelblue', 'no encirclement'),
                          (True, 'crimson', 'encirclement')]:
    fs   = [results[ue][b][0] for b in beta_vals]
    fstd = [results[ue][b][1] for b in beta_vals]
    ax.errorbar(beta_vals, fs, yerr=fstd, fmt='o-', color=color,
                lw=2, capsize=4, label=label)
ax.axhline(0.5, ls=':', color='k', alpha=0.4)
ax.set_xlabel('Contagion rate beta (gamma=2.0)')
ax.set_ylabel('Steady-state panic fraction f_ss')
ax.set_title('Critical shift under encirclement (%d seeds)' % N_SEEDS)
ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/critical_shift_1.png', dpi=120)
plt.close()
print('  --> figures/critical_shift_1.png')
