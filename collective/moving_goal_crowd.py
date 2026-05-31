# moving_goal_crowd.py -- does the many-wrongs average (F82) help a crowd track a
# MOVING goal, and does per-agent noise cost steering BANDWIDTH (F77)?
#
# F77 found the flock has a finite steering bandwidth: an informed minority can
# drive a turn only below a critical rate omega_crit ~ 1/(response time), which
# scales with the pull. F82 found that when every agent holds a noisy private goal
# estimate, alignment averages them to a 1/sqrt(N) accurate heading. This combines
# the two (the open question left by F83): the goal direction rotates at omega and
# every agent biases toward the CURRENT goal carrying its own FIXED angular offset,
# g_hat_i(t) = rotate(goal(t), phi_i), phi_i ~ N(0, sigma_pref).
#
# Key question: does the noise lower the tracking bandwidth, and if so, how? The
# many-wrongs average is SPATIAL (over agents, instantaneous each step), while the
# bandwidth is a TEMPORAL response. Hypothesis: they are independent -- the averaged
# bias still points at the current goal (the static offsets cancel) and rotates
# cleanly with it, so noise adds NO lag. It lowers bandwidth ONLY by shrinking the
# averaged-bias MAGNITUDE, which is w*exp(-sigma_pref^2/2) (the F82 Exp2 law), i.e.
# exactly like reducing the pull strength w. So omega_crit should drop with
# sigma_pref in proportion to exp(-sigma_pref^2/2), not by an extra phase lag.
#
# Sweep omega x sigma_pref. Metric: tracking accuracy = time-avg cos(flock heading
# minus goal angle) over the measurement window; Phi.

import os
import numpy as np
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from flocking import buffer as buf_fn, order_parameter

os.makedirs('figures', exist_ok=True)
N_SEEDS = 8

BASE = dict(r0=0.005, eps=0.1, rf=0.1, alpha=1.0, v0=1.0, mu=10.0, ramp=0.5, dt=0.01)
N_WARMUP = 1000
N_ITER   = 4000
W_BIAS   = 0.5
N_FIX    = 250


def run(omega, sigma_pref, seed):
    rng = np.random.RandomState(seed)
    p = BASE; dt = p['dt']
    r0, eps, rf = p['r0'], p['eps'], p['rf']
    v0, mu, alpha = p['v0'], p['mu'], p['alpha']
    N = N_FIX

    phi = rng.normal(0.0, sigma_pref, N)   # fixed per-agent offset from the goal

    x = np.zeros(2*N)
    x[:N] = rng.uniform(0., 1., N); x[N:] = rng.uniform(0., 1., N)
    vx = rng.uniform(-1., 1., N) * v0
    vy = rng.uniform(-1., 1., N) * v0

    rb = max(r0, rf)
    acc_rec = []; phi_rec = []
    for i in range(N_ITER):
        goal_ang = omega * (i * dt)              # goal rotates from +x at rate omega
        gx = np.cos(goal_ang + phi); gy = np.sin(goal_ang + phi)

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

        fx = flockx + repx + fpropx + frandx + W_BIAS * gx
        fy = flocky + repy + fpropy + frandy + W_BIAS * gy

        vx += fx * dt; vy += fy * dt
        x[:N] = (x[:N] + vx*dt) % 1.; x[N:] = (x[N:] + vy*dt) % 1.

        if i >= N_WARMUP:
            mvx = vx.mean(); mvy = vy.mean(); mn = np.hypot(mvx, mvy)
            head = np.arctan2(mvy, mvx)
            acc_rec.append(np.cos(head - goal_ang) if mn > 1e-9 else 0.0)
            phi_rec.append(order_parameter(vx, vy))
    return float(np.mean(acc_rec)), float(np.std(acc_rec)), float(np.mean(phi_rec))


def sweep_seeds(omega, sigma_pref):
    accs = []; phis = []
    for s in range(N_SEEDS):
        a, _, ph = run(omega, sigma_pref, s)
        accs.append(a); phis.append(ph)
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(phis))


print('Many-wrongs tracking of a MOVING goal: does noise cost steering bandwidth?')
print('  N=%d  w_bias=%.1f  %d seeds  pure-flock\n' % (N_FIX, W_BIAS, N_SEEDS))

OMEGA_VALS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]
SIGMA_VALS = [0.0, 0.5, 1.0, 1.5]
res = {}
for sg in SIGMA_VALS:
    mag = np.exp(-sg**2/2.0)   # F82 averaged-bias magnitude factor
    print('== sigma_pref=%.2f rad  (avg-bias magnitude factor exp(-s^2/2)=%.3f) ==' % (sg, mag))
    for om in OMEGA_VALS:
        acc, sd, phi = sweep_seeds(om, sg)
        res[(sg, om)] = (acc, sd, phi)
        print('   omega=%.2f rad/tu  tracking acc=%+.3f +/- %.3f  Phi=%.3f' % (om, acc, sd, phi))
    print()

# crude bandwidth: largest omega with tracking accuracy >= 0.5
print('Bandwidth (largest omega with tracking acc >= 0.5), vs exp(-s^2/2) prediction:')
om0 = None
for sg in SIGMA_VALS:
    band = 0.0
    for om in OMEGA_VALS:
        if res[(sg, om)][0] >= 0.5:
            band = om
    if sg == 0.0:
        om0 = band if band > 0 else OMEGA_VALS[-1]
    pred = om0 * np.exp(-sg**2/2.0)
    print('   sigma=%.2f  bandwidth~%.2f rad/tu   exp(-s^2/2)-scaled prediction~%.2f' % (sg, band, pred))
print()

# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Many-wrongs tracking of a moving goal (N=%d, w_bias=%.1f, %d seeds)'
             % (N_FIX, W_BIAS, N_SEEDS), fontsize=11)
ax = axes[0]
cols = ['steelblue', 'seagreen', 'darkorange', 'crimson']
for sg, col in zip(SIGMA_VALS, cols):
    acc = [res[(sg, om)][0] for om in OMEGA_VALS]
    err = [res[(sg, om)][1] for om in OMEGA_VALS]
    ax.errorbar(OMEGA_VALS, acc, yerr=err, marker='o', capsize=3, lw=2, color=col,
                label='sigma=%.1f' % sg)
ax.axhline(0.5, ls=':', color='gray')
ax.set_xlabel('goal turning rate omega (rad/tu)'); ax.set_ylabel('tracking accuracy')
ax.set_title('Tracking vs turning rate: noise shifts the bandwidth down')
ax.set_ylim(-0.3, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=9, title='per-agent noise')

ax = axes[1]
# accuracy vs sigma at fixed slow omega (static-penalty view) and the magnitude factor
sg_fine = np.linspace(0, 1.5, 50)
ax.plot(sg_fine, np.exp(-sg_fine**2/2.0), '--', color='gray', lw=1.5,
        label='exp(-sigma^2/2) (avg-bias magnitude)')
for om, col in zip([0.0, 0.05, 0.10], ['steelblue', 'darkorange', 'crimson']):
    acc = [res[(sg, om)][0] for sg in SIGMA_VALS]
    ax.plot(SIGMA_VALS, acc, 'o-', lw=2, color=col, label='omega=%.2f' % om)
ax.set_xlabel('per-agent noise sigma_pref (rad)'); ax.set_ylabel('tracking accuracy / magnitude factor')
ax.set_title('Noise penalty tracks the magnitude factor exp(-s^2/2)')
ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('figures/moving_goal_crowd_1.png', dpi=120)
plt.close()
print('  --> figures/moving_goal_crowd_1.png')
print('\nMoving-goal crowd-tracking analysis complete.')
