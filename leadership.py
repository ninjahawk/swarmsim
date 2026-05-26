# leadership.py -- can a small INFORMED MINORITY steer the whole flock?
#
# Every prior result concerns robustness or disruption. This opens the classic
# collective-decision question (Couzin et al. 2005, "Effective leadership and
# decision-making in animal groups on the move"): if a fraction rho of agents
# have a preferred travel direction and the rest are naive followers, how
# accurately does the whole flock travel in the preferred direction, and how
# small can rho be?
#
# This connects to the F70/F71 theme that the flock acts collectively only on a
# SHARED directional signal. Here the signal is held by a minority; the question
# is whether alignment PROPAGATES it to the majority.
#
# Informed agents get an extra force w_lead * g_hat (g_hat = preferred unit
# vector, +x here) added each step; naive agents get only the usual four forces.
# Pure-flock regime (v0=1.0, no predators, no contagion), default parameters.
#
# Metrics (averaged over the measurement window):
#   accuracy  = (flock mean velocity . g_hat) / |flock mean velocity|  in [-1,1]
#               (1 = flock travels exactly in the preferred direction)
#   Phi       = order parameter (does cohesion survive the steering force?)
# Sweep rho = informed fraction; two leader strengths w_lead.

import os
import numpy as np
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from flocking import buffer as buf_fn, order_parameter

os.makedirs('figures', exist_ok=True)
N_SEEDS = 4

BASE = dict(N=350, r0=0.005, eps=0.1, rf=0.1,
            alpha=1.0, v0=1.0, mu=10.0, ramp=0.5, dt=0.01)
N_WARMUP = 1000
N_ITER   = 4000
G_HAT = np.array([1.0, 0.0])   # preferred direction (+x)
RHO_VALS = [0.0, 0.02, 0.05, 0.10, 0.20, 0.50]
WLEAD_VALS = [0.5, 1.0]        # leader bias strength (alpha=1.0 reference)


def run(rho, w_lead, seed):
    rng = np.random.RandomState(seed)
    p = BASE.copy()
    N = p['N']; dt = p['dt']
    r0, eps, rf = p['r0'], p['eps'], p['rf']
    v0, mu = p['v0'], p['mu']

    n_inf = int(round(rho * N))
    informed = np.zeros(N, dtype=bool)
    if n_inf > 0:
        informed[rng.choice(N, size=n_inf, replace=False)] = True

    x  = np.zeros(2*N)
    x[:N] = rng.uniform(0., 1., N); x[N:] = rng.uniform(0., 1., N)
    vx = rng.uniform(-1., 1., N) * v0
    vy = rng.uniform(-1., 1., N) * v0

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

        fx = flockx + repx + fpropx + frandx
        fy = flocky + repy + fpropy + frandy

        # leadership bias on informed agents
        if n_inf > 0 and w_lead > 0:
            fx[informed] += w_lead * G_HAT[0]
            fy[informed] += w_lead * G_HAT[1]

        vx += fx * dt; vy += fy * dt
        x[:N] = (x[:N] + vx*dt) % 1.; x[N:] = (x[N:] + vy*dt) % 1.

        if i >= N_WARMUP:
            mvx = vx.mean(); mvy = vy.mean()
            mnorm = np.hypot(mvx, mvy)
            acc = (mvx*G_HAT[0] + mvy*G_HAT[1]) / mnorm if mnorm > 1e-9 else 0.0
            acc_rec.append(acc)
            phi_rec.append(order_parameter(vx, vy))
    return float(np.mean(acc_rec)), float(np.mean(phi_rec))


print('Informed-minority leadership: can a fraction rho steer the flock toward +x?')
print('  N=%d  pure-flock regime (v0=1.0)  %d seeds  alpha=1.0\n' % (BASE['N'], N_SEEDS))
results = {}
for w_lead in WLEAD_VALS:
    print('  == w_lead=%.1f ==' % w_lead)
    for rho in RHO_VALS:
        accs = []; phis = []
        for s in range(N_SEEDS):
            a, ph = run(rho, w_lead, s)
            accs.append(a); phis.append(ph)
        results[(w_lead, rho)] = (np.mean(accs), np.std(accs), np.mean(phis))
        print('     rho=%.2f (%3d informed)  accuracy=%+.3f +/- %.3f  Phi=%.3f' %
              (rho, int(round(rho*BASE['N'])), np.mean(accs), np.std(accs), np.mean(phis)))

# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Informed-minority leadership: directional accuracy vs informed fraction '
             '(N=%d, %d seeds)' % (BASE['N'], N_SEEDS), fontsize=11)
ax = axes[0]
for w_lead, col in zip(WLEAD_VALS, ['steelblue', 'crimson']):
    acc = [results[(w_lead, r)][0] for r in RHO_VALS]
    err = [results[(w_lead, r)][1] for r in RHO_VALS]
    ax.errorbar(RHO_VALS, acc, yerr=err, marker='o', capsize=4, lw=2, color=col,
                label='w_lead=%.1f' % w_lead)
ax.axhline(0, ls=':', color='gray')
ax.set_xlabel('informed fraction rho'); ax.set_ylabel('directional accuracy (cos angle to goal)')
ax.set_title('Accuracy vs informed fraction'); ax.set_ylim(-0.1, 1.05)
ax.grid(alpha=0.3); ax.legend(fontsize=9)

ax = axes[1]
for w_lead, col in zip(WLEAD_VALS, ['steelblue', 'crimson']):
    phi = [results[(w_lead, r)][2] for r in RHO_VALS]
    ax.plot(RHO_VALS, phi, 'o-', color=col, lw=2, label='w_lead=%.1f' % w_lead)
ax.set_xlabel('informed fraction rho'); ax.set_ylabel('order parameter Phi')
ax.set_title('Cohesion vs informed fraction'); ax.set_ylim(0, 1.05)
ax.grid(alpha=0.3); ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('figures/leadership_1.png', dpi=120)
plt.close()
print('\n  --> figures/leadership_1.png')
print('\nInformed-minority leadership analysis complete.')
