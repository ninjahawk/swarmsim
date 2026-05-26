# moving_goal.py -- can an informed minority steer a TURNING flock? (steering bandwidth)
#
# F72-F76 all used a FIXED goal direction. Real navigation requires turning. Here the goal
# direction rotates at angular velocity omega (rad/tu): g_hat(t) = (cos(omega t), sin(omega t)).
# The informed minority always biases toward the CURRENT goal. Questions: how well does the flock
# track a turning target, how much does it lag, and is there a critical turning rate above which
# tracking collapses -- a steering bandwidth set by the F75 response time (~few tu, so we expect
# omega_crit ~ 1/response ~ 0.1-0.3 rad/tu)? And do more leaders raise the bandwidth?
#
# Metrics over the measurement window:
#   track = mean cos(flock_heading - goal_angle)  in [-1,1]   (1 = perfect tracking)
#   lag   = mean signed angle by which the flock heading trails the goal (deg)
#   Phi   = order parameter

import os
import numpy as np
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from flocking import buffer as buf_fn, order_parameter

os.makedirs('figures', exist_ok=True)
N_SEEDS = 5

BASE = dict(N=350, r0=0.005, eps=0.1, rf=0.1,
            alpha=1.0, v0=1.0, mu=10.0, ramp=0.5, dt=0.01)
N_WARMUP = 1500          # let the steady lag establish
N_ITER   = 5000
W_LEAD   = 1.0


def run(rho, omega, seed):
    """omega = goal angular velocity (rad/tu). Informed agents bias toward the live goal."""
    rng = np.random.RandomState(seed)
    p = BASE.copy()
    N = p['N']; dt = p['dt']
    r0, eps, rf = p['r0'], p['eps'], p['rf']
    v0, mu = p['v0'], p['mu']

    n_inf = int(round(rho * N))
    informed = np.zeros(N, dtype=bool)
    if n_inf > 0:
        informed[rng.choice(N, size=n_inf, replace=False)] = True

    x  = np.zeros(2 * N)
    x[:N] = rng.uniform(0., 1., N); x[N:] = rng.uniform(0., 1., N)
    vx = rng.uniform(-1., 1., N) * v0
    vy = rng.uniform(-1., 1., N) * v0

    rb = max(r0, rf)
    track_rec = []; lag_rec = []; phi_rec = []
    for i in range(N_ITER):
        t = i * dt
        ga = omega * t                          # goal angle
        gx, gy = np.cos(ga), np.sin(ga)

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

        rep_mask = (d2 <= (2 * r0)**2) & not_self & (d2 > 0)
        d_safe   = np.where(rep_mask, np.sqrt(d2), 1.)
        base_r   = np.where(rep_mask, 1. - d_safe / (2. * r0), 0.)
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

        if n_inf > 0:
            fx[informed] += W_LEAD * gx; fy[informed] += W_LEAD * gy

        vx += fx * dt; vy += fy * dt
        x[:N] = (x[:N] + vx * dt) % 1.; x[N:] = (x[N:] + vy * dt) % 1.

        if i >= N_WARMUP:
            ha = np.arctan2(vy.mean(), vx.mean())
            d = (ha - ga + np.pi) % (2 * np.pi) - np.pi   # signed heading - goal
            track_rec.append(np.cos(d))
            lag_rec.append(np.degrees(d))
            phi_rec.append(order_parameter(vx, vy))
    return float(np.mean(track_rec)), float(np.mean(lag_rec)), float(np.mean(phi_rec))


print('Steering a TURNING flock: goal rotates at angular velocity omega (rad/tu)')
print('  N=%d  pure-flock  %d seeds  w_lead=%.1f\n' % (BASE['N'], N_SEEDS, W_LEAD))

OMEGA_VALS = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
RHO_VALS = [0.10, 0.20]
results = {}
for rho in RHO_VALS:
    print('== rho=%.2f (%d informed) ==' % (rho, int(round(rho * BASE['N']))))
    for omega in OMEGA_VALS:
        tr = []; lg = []; ph = []
        for s in range(N_SEEDS):
            a, l, p_ = run(rho, omega, s)
            tr.append(a); lg.append(l); ph.append(p_)
        results[(rho, omega)] = (np.mean(tr), np.std(tr), np.mean(lg), np.mean(ph))
        print('   omega=%.2f rad/tu  track=%+.3f +/- %.3f  lag=%+6.1f deg  Phi=%.3f'
              % (omega, np.mean(tr), np.std(tr), np.mean(lg), np.mean(ph)))
    print()

# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Steering a turning flock: tracking vs goal turning rate (N=%d, %d seeds)'
             % (BASE['N'], N_SEEDS), fontsize=12)
ax = axes[0]
for rho, col in zip(RHO_VALS, ['steelblue', 'crimson']):
    tr = [results[(rho, o)][0] for o in OMEGA_VALS]
    er = [results[(rho, o)][1] for o in OMEGA_VALS]
    ax.errorbar(OMEGA_VALS, tr, yerr=er, marker='o', capsize=4, lw=2, color=col,
                label='rho=%.2f' % rho)
ax.axhline(0, ls=':', color='gray')
ax.set_xlabel('goal turning rate omega (rad/tu)'); ax.set_ylabel('tracking cos(heading - goal)')
ax.set_title('Tracking vs turning rate'); ax.set_ylim(-0.3, 1.05)
ax.grid(alpha=0.3); ax.legend(fontsize=10)

ax = axes[1]
for rho, col in zip(RHO_VALS, ['steelblue', 'crimson']):
    lg = [results[(rho, o)][2] for o in OMEGA_VALS]
    ax.plot(OMEGA_VALS, lg, 'o-', lw=2, color=col, label='rho=%.2f' % rho)
ax.axhline(0, ls=':', color='gray')
ax.set_xlabel('goal turning rate omega (rad/tu)'); ax.set_ylabel('heading lag behind goal (deg)')
ax.set_title('Lag vs turning rate'); ax.grid(alpha=0.3); ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('figures/moving_goal_1.png', dpi=120)
plt.close()
print('  --> figures/moving_goal_1.png')
print('\nMoving-goal (steering bandwidth) analysis complete.')
