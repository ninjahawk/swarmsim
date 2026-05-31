# correlated_estimates.py -- F81 and F82 are two ends of ONE axis: the
# CORRELATION between agents' goal-estimate errors. Does correlated sensing
# error cap the wisdom of crowds?
#
# F81: leaders share an EXACT vector -> no group-size amplification (per-capita pull).
# F82: agents hold INDEPENDENT noisy estimates -> RMS heading error ~ sigma/sqrt(N)
#      (1/sqrt(N) wisdom of crowds).
# These look opposite but are the rho_c -> 1 and rho_c -> 0 limits of a single
# parameter: how correlated the agents' estimate errors are. Real collectives
# sit in between -- animals reading the same misleading cue share part of their error.
#
# Construct per-agent preferred-direction angle as
#     phi_i = sqrt(rho_c) * c + sqrt(1 - rho_c) * e_i ,   c, e_i ~ N(0, sigma_pref)
# so Var(phi_i) = sigma_pref^2 and Corr(phi_i, phi_j) = rho_c (one shared draw c
# per run, independent private e_i). rho_c=0 reproduces F82; rho_c=1 gives every
# agent the SAME (shared but wrong) direction -- an F81-like shared signal.
#
# Prediction (small-angle): the alignment-averaged heading error has cross-seed
# variance rho_c*sigma^2 + (1-rho_c)*sigma^2/N, i.e.
#     RMS heading error = sigma_pref * sqrt(rho_c + (1 - rho_c)/N).
# The private part averages away as 1/sqrt(N); the shared part does NOT -- it is a
# floor of sigma_pref*sqrt(rho_c) that no group size can beat. Crossover N* ~ 1/rho_c.
#
# Sweep rho_c x N at fixed sigma_pref. Metric: cross-seed RMS steady heading error
# (deg), accuracy = cos(error), Phi. Many seeds (the shared draw c dominates the
# cross-seed spread when rho_c>0, so RMS needs averaging over many c).

import os
import numpy as np
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from flocking import buffer as buf_fn, order_parameter

os.makedirs('figures', exist_ok=True)
N_SEEDS = 12

BASE = dict(r0=0.005, eps=0.1, rf=0.1, alpha=1.0, v0=1.0, mu=10.0, ramp=0.5, dt=0.01)
N_WARMUP = 1000
N_ITER   = 2500
W_BIAS   = 0.5
SIGMA    = 1.0     # rad, per-agent preferred-direction error (matches F82 Exp1)


def run(N, rho_c, seed):
    rng = np.random.RandomState(seed)
    p = BASE; dt = p['dt']
    r0, eps, rf = p['r0'], p['eps'], p['rf']
    v0, mu, alpha = p['v0'], p['mu'], p['alpha']

    # correlated per-agent preferred directions
    c = rng.normal(0.0, SIGMA)                 # one shared error draw per run
    e = rng.normal(0.0, SIGMA, N)              # independent private errors
    phi = np.sqrt(rho_c) * c + np.sqrt(1.0 - rho_c) * e
    gx = np.cos(phi); gy = np.sin(phi)

    x = np.zeros(2*N)
    x[:N] = rng.uniform(0., 1., N); x[N:] = rng.uniform(0., 1., N)
    vx = rng.uniform(-1., 1., N) * v0
    vy = rng.uniform(-1., 1., N) * v0

    rb = max(r0, rf)
    mvx_rec = []; mvy_rec = []; acc_rec = []; phi_rec = []
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

        fx = flockx + repx + fpropx + frandx + W_BIAS * gx
        fy = flocky + repy + fpropy + frandy + W_BIAS * gy

        vx += fx * dt; vy += fy * dt
        x[:N] = (x[:N] + vx*dt) % 1.; x[N:] = (x[N:] + vy*dt) % 1.

        if i >= N_WARMUP:
            mvx = vx.mean(); mvy = vy.mean(); mn = np.hypot(mvx, mvy)
            mvx_rec.append(mvx); mvy_rec.append(mvy)
            acc_rec.append(mvx/mn if mn > 1e-9 else 0.0)
            phi_rec.append(order_parameter(vx, vy))
    err = np.arctan2(np.mean(mvy_rec), np.mean(mvx_rec))
    return float(err), float(np.mean(acc_rec)), float(np.mean(phi_rec))


def sweep_seeds(N, rho_c):
    errs = []; accs = []; phis = []
    for s in range(N_SEEDS):
        e, a, ph = run(N, rho_c, s)
        errs.append(e); accs.append(a); phis.append(ph)
    errs = np.array(errs)
    rms_deg = np.degrees(np.sqrt(np.mean(errs**2)))
    return rms_deg, float(np.mean(accs)), float(np.std(accs)), float(np.mean(phis))


print('Correlated estimates: does shared sensing error cap the wisdom of crowds?')
print('  w_bias=%.1f  sigma_pref=%.2f rad  %d seeds  pure-flock\n' % (W_BIAS, SIGMA, N_SEEDS))

RHO_VALS = [0.0, 0.1, 0.3, 1.0]
N_VALS   = [30, 125, 500]
res = {}
for rho_c in RHO_VALS:
    print('== correlation rho_c=%.2f ==' % rho_c)
    floor_pred = np.degrees(SIGMA * np.sqrt(rho_c))
    print('   predicted N->inf floor = sigma*sqrt(rho_c) = %.1f deg' % floor_pred)
    for N in N_VALS:
        rms, acc, accsd, phi = sweep_seeds(N, rho_c)
        pred = np.degrees(SIGMA * np.sqrt(rho_c + (1.0 - rho_c)/N))
        res[(rho_c, N)] = (rms, acc, accsd, phi, pred)
        print('   N=%4d  RMS err=%5.1f deg (predict %5.1f)  accuracy=%+.3f +/- %.3f  Phi=%.3f'
              % (N, rms, pred, acc, accsd, phi))
    print()

print('Reading: rho_c=0 reproduces F82 (error falls ~1/sqrt(N)); any rho_c>0 imposes')
print('an N-independent floor sigma*sqrt(rho_c) -- correlated error caps collective accuracy.')
print('rho_c=1 (every agent the same wrong direction) is the F81-like shared-signal limit:')
print('no group-size benefit at all.\n')

# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Correlated goal estimates: shared sensing error caps the wisdom of crowds '
             '(sigma_pref=%.1f rad, w_bias=%.1f, %d seeds)' % (SIGMA, W_BIAS, N_SEEDS), fontsize=11)
ax = axes[0]
cols = ['crimson', 'darkorange', 'steelblue', 'seagreen']
for rho_c, col in zip(RHO_VALS, cols):
    rms = [res[(rho_c, N)][0] for N in N_VALS]
    ax.loglog(N_VALS, rms, 'o-', lw=2, color=col, label='rho_c=%.1f' % rho_c)
    floor = np.degrees(SIGMA * np.sqrt(rho_c))
    if rho_c > 0:
        ax.axhline(floor, ls=':', color=col, alpha=0.6)
ax.set_xlabel('flock size N'); ax.set_ylabel('cross-seed RMS heading error (deg)')
ax.set_title('Error vs N: rho_c=0 falls, rho_c>0 plateaus at sigma*sqrt(rho_c)')
ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=9, title='error correlation')

ax = axes[1]
for N, col in zip(N_VALS, ['plum', 'mediumpurple', 'indigo']):
    acc = [res[(rho_c, N)][1] for rho_c in RHO_VALS]
    ax.plot(RHO_VALS, acc, 'o-', lw=2, color=col, label='N=%d' % N)
ax.set_xlabel('estimate-error correlation rho_c'); ax.set_ylabel('accuracy toward true goal')
ax.set_title('Accuracy collapses with correlation, N-independent at high rho_c')
ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('figures/correlated_estimates_1.png', dpi=120)
plt.close()
print('  --> figures/correlated_estimates_1.png')
print('\nCorrelated-estimates analysis complete.')
