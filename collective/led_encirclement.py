# led_encirclement.py -- can an informed minority hold the flock together and steer it
# while predators ENCIRCLE it? (the predator thread meets the leadership thread)
#
# F14 showed encirclement (n_pred=6, R_enc=0.15) is the one predator strategy that breaks 2D
# flock coherence (Phi ~ 0.77, fragmenting into sub-flocks, F16). F70 showed prey can counter
# PREDICTIVE encirclement with a shared collective-escape direction. Here we ask the leadership
# version: if a fraction rho of prey carry a shared GOAL direction (+x) -- not fleeing the
# predators, just committed to a heading -- does that shared signal (a) restore coherence the
# encirclement destroyed, and (b) steer the flock toward the goal despite the predators?
#
# Slow-prey predator regime (matches F14 calibration: PREY_DEFAULT v0=0.02, ramp=0.1, N=100),
# self-contained (no import of the guard-less legacy predator modules). Encirclement predators
# placed on the F14 ring around the live prey CoM each step (sign verified: prey pushed AWAY).
#
# Two columns: predators OFF (pure leadership in the slow regime, calibration) and predators ON
# (encirclement). Sweep informed fraction rho. Metrics: Phi (coherence) and accuracy toward +x.

import os
import numpy as np
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from flocking import buffer as buf_fn, order_parameter

os.makedirs('figures', exist_ok=True)
N_SEEDS = 6

# slow-prey predator regime (F14 calibration)
PREY = dict(N=100, r0=0.005, eps=0.1, rf=0.1, alpha=1.0, v0=0.02, mu=10.0, ramp=0.1, dt=0.01)
PRED = dict(v0_pred=0.05, mu_pred=10.0, alpha_pred=5.0, r0_pred=0.1, eps_pred=2.0, ramp_pred=1.0)
N_ITER  = 4000
N_WARM  = 1500
W_LEAD  = 0.5            # leader bias (slow regime: v_eq=0.12, alpha=1.0; 0.5 steers without dominating)
G_HAT   = np.array([1.0, 0.0])
N_PRED  = 6
R_ENC   = 0.15


def circ_com(a):
    """circular mean of positions in [0,1) (periodic)."""
    return np.arctan2(np.sin(2*np.pi*a).mean(), np.cos(2*np.pi*a).mean()) / (2*np.pi) % 1.


def run(rho, predators, seed):
    rng = np.random.RandomState(seed)
    p = PREY; N = p['N']; dt = p['dt']
    r0, eps, rf = p['r0'], p['eps'], p['rf']
    v0, mu, alpha = p['v0'], p['mu'], p['alpha']

    n_inf = int(round(rho * N))
    informed = np.zeros(N, dtype=bool)
    if n_inf > 0:
        informed[rng.choice(N, size=n_inf, replace=False)] = True

    x = np.zeros(2*N)
    x[:N] = rng.uniform(0., 1., N); x[N:] = rng.uniform(0., 1., N)
    vx = rng.uniform(-1., 1., N) * v0
    vy = rng.uniform(-1., 1., N) * v0

    angles = np.array([2*np.pi*k/N_PRED for k in range(N_PRED)])
    pred_x = (0.5 + 0.45*np.cos(angles)) % 1.
    pred_y = (0.5 + 0.45*np.sin(angles)) % 1.
    pred_vx = np.zeros(N_PRED); pred_vy = np.zeros(N_PRED)

    rb = max(r0, rf)
    acc_rec = []; phi_rec = []
    for i in range(N_ITER):
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
        flockx = alpha * flx / nrm; flocky = alpha * fly / nrm

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

        fx = flockx + repx + fpropx + frandx
        fy = flocky + repy + fpropy + frandy

        if predators:
            # repulsion from each predator onto prey (min image), sign: push prey AWAY
            for ip in range(N_PRED):
                ddx = pred_x[ip] - x[:N]; ddx -= np.round(ddx)
                ddy = pred_y[ip] - x[N:]; ddy -= np.round(ddy)
                dd = np.sqrt(ddx**2 + ddy**2)
                m = (dd > 0) & (dd <= PRED['r0_pred'])
                base = np.where(m, 1. - dd/PRED['r0_pred'], 0.)   # >=0 where in range, else 0 (no neg^1.5)
                s = np.where(m, PRED['eps_pred'] * base**1.5 / np.where(m, dd, 1.), 0.)
                fx -= s * ddx; fy -= s * ddy   # fx[j] -= s*(pred-prey) = +s*(prey-pred): repulsion

        if n_inf > 0:
            fx[informed] += W_LEAD * G_HAT[0]; fy[informed] += W_LEAD * G_HAT[1]

        vx += fx * dt; vy += fy * dt
        x[:N] = (x[:N] + vx*dt) % 1.; x[N:] = (x[N:] + vy*dt) % 1.

        if predators:
            cx = circ_com(x[:N]); cy = circ_com(x[N:])
            for ip in range(N_PRED):
                txg = (cx + R_ENC*np.cos(angles[ip])) % 1.
                tyg = (cy + R_ENC*np.sin(angles[ip])) % 1.
                tx = txg - pred_x[ip]; tx -= round(tx)
                ty = tyg - pred_y[ip]; ty -= round(ty)
                dist = np.sqrt(tx**2 + ty**2)
                if dist > 0: tx /= dist; ty /= dist
                sp = np.hypot(pred_vx[ip], pred_vy[ip])
                pfx = PRED['alpha_pred']*tx; pfy = PRED['alpha_pred']*ty
                if sp > 0:
                    pfx += PRED['mu_pred']*(PRED['v0_pred']-sp)*pred_vx[ip]/sp
                    pfy += PRED['mu_pred']*(PRED['v0_pred']-sp)*pred_vy[ip]/sp
                pfx += PRED['ramp_pred']*rng.uniform(-1., 1.)
                pfy += PRED['ramp_pred']*rng.uniform(-1., 1.)
                pred_vx[ip] += pfx*dt; pred_vy[ip] += pfy*dt
                pred_x[ip] = (pred_x[ip] + pred_vx[ip]*dt) % 1.
                pred_y[ip] = (pred_y[ip] + pred_vy[ip]*dt) % 1.

        if i >= N_WARM:
            mvx = vx.mean(); mvy = vy.mean(); mn = np.hypot(mvx, mvy)
            acc_rec.append((mvx*G_HAT[0] + mvy*G_HAT[1])/mn if mn > 1e-9 else 0.0)
            phi_rec.append(order_parameter(vx, vy))
    return float(np.mean(acc_rec)), float(np.mean(phi_rec))


print('Leadership vs encirclement: can an informed minority hold and steer an encircled flock?')
print('  slow-prey regime N=%d  n_pred=%d  R_enc=%.2f  w_lead=%.1f  %d seeds\n'
      % (PREY['N'], N_PRED, R_ENC, W_LEAD, N_SEEDS))

RHO_VALS = [0.0, 0.05, 0.10, 0.20, 0.40]
results = {}
for predators in [False, True]:
    tag = 'predators ON (encircle n=6)' if predators else 'predators OFF (pure leadership)'
    print('== %s ==' % tag)
    for rho in RHO_VALS:
        accs = []; phis = []
        for s in range(N_SEEDS):
            a, ph = run(rho, predators, s)
            accs.append(a); phis.append(ph)
        results[(predators, rho)] = (np.mean(accs), np.std(accs), np.mean(phis), np.std(phis))
        print('   rho=%.2f (%2d informed)  accuracy=%+.3f +/- %.3f  Phi=%.3f +/- %.3f'
              % (rho, int(round(rho*PREY['N'])), np.mean(accs), np.std(accs),
                 np.mean(phis), np.std(phis)))
    print()

# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Leadership under encirclement (slow-prey, n_pred=%d, R_enc=%.2f, %d seeds)'
             % (N_PRED, R_ENC, N_SEEDS), fontsize=12)
ax = axes[0]
for predators, col, lab in [(False, 'steelblue', 'no predators'), (True, 'crimson', 'encirclement')]:
    ph = [results[(predators, r)][2] for r in RHO_VALS]
    es = [results[(predators, r)][3] for r in RHO_VALS]
    ax.errorbar(RHO_VALS, ph, yerr=es, marker='o', capsize=4, lw=2, color=col, label=lab)
ax.set_xlabel('informed fraction rho'); ax.set_ylabel('order parameter Phi')
ax.set_title('Coherence: does leadership restore what encirclement broke?')
ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=9)
ax = axes[1]
for predators, col, lab in [(False, 'steelblue', 'no predators'), (True, 'crimson', 'encirclement')]:
    ac = [results[(predators, r)][0] for r in RHO_VALS]
    es = [results[(predators, r)][1] for r in RHO_VALS]
    ax.errorbar(RHO_VALS, ac, yerr=es, marker='o', capsize=4, lw=2, color=col, label=lab)
ax.axhline(0, ls=':', color='gray')
ax.set_xlabel('informed fraction rho'); ax.set_ylabel('accuracy toward goal (+x)')
ax.set_title('Steering: can leaders aim the flock through the ring?')
ax.set_ylim(-0.3, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('figures/led_encirclement_1.png', dpi=120)
plt.close()
print('  --> figures/led_encirclement_1.png')
print('\nLeadership-under-encirclement analysis complete.')
