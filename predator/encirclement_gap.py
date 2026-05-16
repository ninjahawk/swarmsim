# encirclement_gap.py -- Incomplete encirclement: does the flock escape through a gap?
#
# Findings 14-16 used n_pred predators at equally spaced angles theta_k =
# 2*pi*k/n_pred -- a full encirclement.  Biologically, predators may have
# gaps: a wolf pack approaches from upwind, leaving the downwind side open.
# Does the flock detect and exploit the gap?
#
# Strategy: start with the full n_pred = 6 encirclement.  Remove predators
# one at a time, recording (a) steady-state Phi, (b) flock CoM drift direction.
# If the flock detects gaps it should drift toward them; if not, it should
# stay roughly centered.

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
    n_iter=4000,
)
PRED = dict(v0=0.05, mu=10.0, alpha=5.0, r0=0.1, eps=2.0, ramp=1.0)


def run(n_active_pred, n_total_pred, gap_pattern, R_enc, seed):
    """
    n_active_pred : number of predators that act
    n_total_pred  : the geometrical layout assumes this many angular slots
    gap_pattern   : 'contiguous' (remove last (n_total - n_active) slots, gap on one side)
                  or 'distributed' (uniformly remove)
    """
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

    all_angles = np.radians(np.arange(n_total_pred) * 360.0 / n_total_pred)
    if gap_pattern == 'contiguous':
        active_idx = list(range(n_active_pred))
    elif gap_pattern == 'distributed':
        active_idx = list(np.linspace(0, n_total_pred-1, n_active_pred, dtype=int))
    else:
        raise ValueError(gap_pattern)
    angles = all_angles[active_idx]
    n_pred = len(angles)

    # Gap center (for measuring drift direction)
    gap_angles = np.array([a for k, a in enumerate(all_angles) if k not in active_idx])
    if gap_angles.size:
        # mean angle (circular)
        gap_dir = np.arctan2(np.sin(gap_angles).mean(), np.cos(gap_angles).mean())
    else:
        gap_dir = None

    pred_x  = np.random.uniform(0., 1., n_pred)
    pred_y  = np.random.uniform(0., 1., n_pred)
    pred_vx = np.zeros(n_pred); pred_vy = np.zeros(n_pred)

    rb = max(r0, rf)
    phi_t=[]; com_traj=[]

    initial_cx, initial_cy = _periodic_com(x[:N]), _periodic_com(x[N:])

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

        cx, cy = _periodic_com(x[:N]), _periodic_com(x[N:])
        for k in range(n_pred):
            tx = (cx + R_enc * np.cos(angles[k])) % 1.
            ty = (cy + R_enc * np.sin(angles[k])) % 1.
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
            com_traj.append((_periodic_disp(cx, initial_cx),
                             _periodic_disp(cy, initial_cy)))

    com_traj = np.array(com_traj)
    final_dx = com_traj[-1, 0]; final_dy = com_traj[-1, 1]
    drift_dir = np.arctan2(final_dy, final_dx) if (final_dx**2+final_dy**2) > 1e-8 else None
    drift_mag = np.sqrt(final_dx**2 + final_dy**2)

    if gap_dir is not None and drift_dir is not None:
        # Angle between drift direction and gap direction (in [0, pi])
        angle_diff = abs(drift_dir - gap_dir)
        if angle_diff > np.pi:
            angle_diff = 2*np.pi - angle_diff
    else:
        angle_diff = None

    return np.mean(phi_t), drift_mag, angle_diff


# Sweep n_active_pred from 6 (full) down to 1, at n_total=6 (60-degree slots).
print('Encirclement gap test: n_total=6, R_enc=0.15')
n_total = 6
R_enc = 0.15
n_active_vals = [6, 5, 4, 3]

results_contig = {}
results_distr  = {}
for n_act in n_active_vals:
    print('  n_active=%d  contiguous gap:' % n_act, end='', flush=True)
    phis=[]; mags=[]; angs=[]
    for s in range(N_SEEDS):
        phi, mag, ang = run(n_act, n_total, 'contiguous', R_enc, s)
        phis.append(phi); mags.append(mag); angs.append(ang)
    results_contig[n_act] = dict(
        phi=np.mean(phis), phi_std=np.std(phis),
        drift_mag=np.mean(mags),
        ang_diff=np.nanmean([a for a in angs if a is not None]) if any(a is not None for a in angs) else None
    )
    print('  Phi=%.3f +/- %.3f  drift_mag=%.3f  ang_diff=%s' % (
        results_contig[n_act]['phi'], results_contig[n_act]['phi_std'],
        results_contig[n_act]['drift_mag'],
        results_contig[n_act]['ang_diff']))

    print('  n_active=%d  distributed gap:' % n_act, end='', flush=True)
    phis=[]; mags=[]
    for s in range(N_SEEDS):
        phi, mag, _ = run(n_act, n_total, 'distributed', R_enc, s)
        phis.append(phi); mags.append(mag)
    results_distr[n_act] = dict(phi=np.mean(phis), phi_std=np.std(phis),
                                drift_mag=np.mean(mags))
    print('  Phi=%.3f +/- %.3f  drift_mag=%.3f' % (
        results_distr[n_act]['phi'], results_distr[n_act]['phi_std'],
        results_distr[n_act]['drift_mag']))


# Figure
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Incomplete encirclement: n_total=6 slots, gap pattern matters (%d seeds)'
             % N_SEEDS, fontsize=11)

ax = axes[0]
phi_c = [results_contig[n]['phi'] for n in n_active_vals]
phi_d = [results_distr[n]['phi']  for n in n_active_vals]
std_c = [results_contig[n]['phi_std'] for n in n_active_vals]
std_d = [results_distr[n]['phi_std']  for n in n_active_vals]
ax.errorbar(n_active_vals, phi_c, yerr=std_c, fmt='o-', color='crimson',
            lw=2, capsize=4, label='contiguous gap')
ax.errorbar(n_active_vals, phi_d, yerr=std_d, fmt='s-', color='steelblue',
            lw=2, capsize=4, label='distributed gap')
ax.set_xlabel('n_active predators (out of 6 slots)')
ax.set_ylabel('Steady-state Phi')
ax.set_ylim(0, 1.05); ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.set_title('Phi vs number of active predators')

ax = axes[1]
drifts_c = [results_contig[n]['drift_mag'] for n in n_active_vals]
drifts_d = [results_distr[n]['drift_mag']  for n in n_active_vals]
ax.plot(n_active_vals, drifts_c, 'o-', color='crimson', lw=2, label='contiguous gap')
ax.plot(n_active_vals, drifts_d, 's-', color='steelblue', lw=2, label='distributed gap')
ax.set_xlabel('n_active predators (out of 6 slots)')
ax.set_ylabel('CoM drift magnitude (over n_iter steps)')
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.set_title('Did the flock escape?')

plt.tight_layout()
plt.savefig('figures/gap_encirclement_1.png', dpi=120)
plt.close()
print('  --> figures/gap_encirclement_1.png')
