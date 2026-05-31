# leader_scaling.py -- how does the leadership requirement scale with GROUP SIZE?
#
# F72 found a fixed FRACTION rho~0.05 steers the flock and noted, citing Couzin et al. (2005),
# that "the informed fraction needed decreases as the group grows." That was asserted from the
# single-N sweep but never measured directly. Here we vary N and fix the informed NUMBER n_lead,
# asking: does a fixed handful of leaders suffice regardless of flock size (so the required
# FRACTION falls as ~n_lead/N), or does the absolute number needed grow with N?
#
# Couzin's prediction: a fixed small number of informed individuals achieves accurate group
# navigation in arbitrarily large groups -- accuracy at fixed n_lead should be roughly N-independent
# (or improve with N), so the fraction needed drops like 1/N.
#
# Sweep N x informed-number. Metric: directional accuracy toward +x, and Phi.

import os
import numpy as np
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from flocking import buffer as buf_fn, order_parameter

os.makedirs('figures', exist_ok=True)
N_SEEDS = 4

BASE = dict(r0=0.005, eps=0.1, rf=0.1, alpha=1.0, v0=1.0, mu=10.0, ramp=0.5, dt=0.01)
N_WARMUP = 1000
N_ITER   = 4000
W_LEAD   = 1.0
G_HAT    = np.array([1.0, 0.0])


def run(N, n_lead, seed):
    rng = np.random.RandomState(seed)
    p = BASE; dt = p['dt']
    r0, eps, rf = p['r0'], p['eps'], p['rf']
    v0, mu, alpha = p['v0'], p['mu'], p['alpha']

    informed = np.zeros(N, dtype=bool)
    if n_lead > 0:
        informed[rng.choice(N, size=min(n_lead, N), replace=False)] = True

    x = np.zeros(2*N)
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

        if n_lead > 0:
            fx[informed] += W_LEAD * G_HAT[0]; fy[informed] += W_LEAD * G_HAT[1]

        vx += fx * dt; vy += fy * dt
        x[:N] = (x[:N] + vx*dt) % 1.; x[N:] = (x[N:] + vy*dt) % 1.

        if i >= N_WARMUP:
            mvx = vx.mean(); mvy = vy.mean(); mn = np.hypot(mvx, mvy)
            acc_rec.append((mvx*G_HAT[0] + mvy*G_HAT[1])/mn if mn > 1e-9 else 0.0)
            phi_rec.append(order_parameter(vx, vy))
    return float(np.mean(acc_rec)), float(np.std(acc_rec)), float(np.mean(phi_rec))


print('Leadership scaling with group size: does a fixed NUMBER of leaders suffice as N grows?')
print('  pure-flock  w_lead=%.1f  %d seeds\n' % (W_LEAD, N_SEEDS))

N_VALS = [100, 250, 500]
NLEAD_VALS = [5, 10, 20]
results = {}
for N in N_VALS:
    print('== N=%d ==' % N)
    for nl in NLEAD_VALS:
        accs = []; phis = []
        for s in range(N_SEEDS):
            a, _, ph = run(N, nl, s)
            accs.append(a); phis.append(ph)
        results[(N, nl)] = (np.mean(accs), np.std(accs), np.mean(phis))
        print('   n_lead=%2d (frac=%.3f)  accuracy=%+.3f +/- %.3f  Phi=%.3f'
              % (nl, nl/N, np.mean(accs), np.std(accs), np.mean(phis)))
    print()

# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Leadership scaling with group size (pure-flock, w_lead=%.1f, %d seeds)'
             % (W_LEAD, N_SEEDS), fontsize=12)
ax = axes[0]
for nl, col in zip(NLEAD_VALS, ['steelblue', 'darkorange', 'crimson']):
    acc = [results[(N, nl)][0] for N in N_VALS]
    err = [results[(N, nl)][1] for N in N_VALS]
    ax.errorbar(N_VALS, acc, yerr=err, marker='o', capsize=4, lw=2, color=col,
                label='n_lead=%d' % nl)
ax.set_xlabel('flock size N'); ax.set_ylabel('accuracy toward goal')
ax.set_title('Accuracy at FIXED leader NUMBER vs N')
ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=9)

ax = axes[1]
for N, col in zip(N_VALS, ['seagreen', 'purple', 'firebrick']):
    fracs = [nl/N for nl in NLEAD_VALS]
    acc = [results[(N, nl)][0] for nl in NLEAD_VALS]
    ax.plot(fracs, acc, 'o-', lw=2, color=col, label='N=%d' % N)
ax.set_xlabel('informed FRACTION n_lead/N'); ax.set_ylabel('accuracy toward goal')
ax.set_title('Accuracy vs fraction (curves shift left as N grows)')
ax.set_ylim(0, 1.05); ax.set_xscale('log'); ax.grid(alpha=0.3); ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('figures/leader_scaling_1.png', dpi=120)
plt.close()
print('  --> figures/leader_scaling_1.png')
print('\nLeadership-scaling analysis complete.')
