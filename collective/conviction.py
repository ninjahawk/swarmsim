# conviction.py -- numbers vs CONVICTION in collective decisions.
#
# F73 (conflicting_leaders.py) showed that when two opposed informed subgroups differ in
# SIZE, the larger subgroup wins -- group direction is an effective majority vote. But all
# leaders there had equal bias strength w_lead. The natural next question: does a smaller
# but more strongly committed subgroup beat a larger weakly committed one? Is the decision
# set by headcount, or by total "pull" = count * strength?
#
# Two opposed subgroups (theta = 180 deg, g1 = +x, g2 = -x). Accuracy toward g1 = cos of the
# flock heading (+1 = g1 wins, -1 = g2 wins, 0 = tie / random pick).
#
# Exp A -- equal numbers (18 vs 18), sweep the conviction ratio w1/w2 (w2 fixed at 1.0).
#          Does stronger conviction win at equal numbers?
# Exp B -- the PRODUCT-LAW test: group 2 fixed at n2=26 weak agents (w2=1.0, total pull 26);
#          group 1 has only n1=10 agents, sweep its strength w1. If the decision is set by
#          total pull, the tie (accuracy = 0) should fall near n1*w1 = n2*w2 -> w1 = 2.6, and
#          a strong enough minority (w1 > 2.6) should overcome its 10-vs-26 numbers deficit.

import os
import numpy as np
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from flocking import buffer as buf_fn, order_parameter

os.makedirs('figures', exist_ok=True)
N_SEEDS = 6

BASE = dict(N=350, r0=0.005, eps=0.1, rf=0.1,
            alpha=1.0, v0=1.0, mu=10.0, ramp=0.5, dt=0.01)
N_WARMUP = 1000
N_ITER   = 4000


def run(n1, w1, n2, w2, seed):
    """Two opposed subgroups: n1 agents biased +x at strength w1, n2 biased -x at w2."""
    rng = np.random.RandomState(seed)
    p = BASE.copy()
    N = p['N']; dt = p['dt']
    r0, eps, rf = p['r0'], p['eps'], p['rf']
    v0, mu = p['v0'], p['mu']

    g1 = np.array([1.0, 0.0]); g2 = np.array([-1.0, 0.0])
    perm = rng.permutation(N)
    grp1 = np.zeros(N, dtype=bool); grp2 = np.zeros(N, dtype=bool)
    grp1[perm[:n1]] = True
    grp2[perm[n1:n1 + n2]] = True

    x  = np.zeros(2 * N)
    x[:N] = rng.uniform(0., 1., N); x[N:] = rng.uniform(0., 1., N)
    vx = rng.uniform(-1., 1., N) * v0
    vy = rng.uniform(-1., 1., N) * v0

    rb = max(r0, rf)
    hx_rec = []; hy_rec = []; phi_rec = []
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

        fx[grp1] += w1 * g1[0]; fy[grp1] += w1 * g1[1]
        fx[grp2] += w2 * g2[0]; fy[grp2] += w2 * g2[1]

        vx += fx * dt; vy += fy * dt
        x[:N] = (x[:N] + vx * dt) % 1.; x[N:] = (x[N:] + vy * dt) % 1.

        if i >= N_WARMUP:
            hx_rec.append(vx.mean()); hy_rec.append(vy.mean())
            phi_rec.append(order_parameter(vx, vy))

    mhx = np.mean(hx_rec); mhy = np.mean(hy_rec)
    mnorm = np.hypot(mhx, mhy)
    acc = mhx / mnorm if mnorm > 1e-9 else 0.0   # cos of heading = accuracy toward +x (g1)
    return acc, float(np.mean(phi_rec))


print('Conviction vs numbers in collective decisions (theta=180 opposition)')
print('  N=%d  pure-flock (v0=1.0)  %d seeds\n' % (BASE['N'], N_SEEDS))

# Exp A -- equal numbers, vary conviction ratio
print('== Exp A: EQUAL numbers (18 vs 18), vary conviction w1/w2 (w2=1.0) ==')
n_each = int(round(0.05 * BASE['N']))
WRATIO = [1.0, 1.5, 2.0, 3.0, 5.0]
expA = {}
for w1 in WRATIO:
    accs = []; phis = []
    for s in range(N_SEEDS):
        a, ph = run(n_each, w1, n_each, 1.0, s)
        accs.append(a); phis.append(ph)
    expA[w1] = (np.mean(accs), np.std(accs), np.mean(phis))
    print('   w1/w2=%.1f  accuracy_to_g1=%+.3f +/- %.3f  Phi=%.3f'
          % (w1, np.mean(accs), np.std(accs), np.mean(phis)))

# Exp B -- product-law test: 10 strong vs 26 weak
print('\n== Exp B: PRODUCT-LAW test -- n1=10 (strong) vs n2=26 (weak, w2=1.0) ==')
print('   total pull balances at n1*w1 = n2*w2 -> w1 = 26/10 = 2.6')
n1B, n2B = 10, 26
W1B = [1.0, 1.8, 2.6, 3.5, 5.0]
expB = {}
for w1 in W1B:
    accs = []; phis = []
    for s in range(N_SEEDS):
        a, ph = run(n1B, w1, n2B, 1.0, s)
        accs.append(a); phis.append(ph)
    pull1 = n1B * w1; pull2 = n2B * 1.0
    expB[w1] = (np.mean(accs), np.std(accs), np.mean(phis), pull1)
    print('   w1=%.1f  pull1=%4.1f vs pull2=%4.1f  accuracy_to_g1=%+.3f +/- %.3f  Phi=%.3f'
          % (w1, pull1, pull2, np.mean(accs), np.std(accs), np.mean(phis)))

# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Conviction vs numbers in collective decisions (N=%d, %d seeds, theta=180)'
             % (BASE['N'], N_SEEDS), fontsize=12)

ax = axes[0]
acc = [expA[w][0] for w in WRATIO]; err = [expA[w][1] for w in WRATIO]
ax.axhline(0, ls=':', color='gray')
ax.errorbar(WRATIO, acc, yerr=err, marker='o', capsize=4, lw=2, color='darkorange')
ax.set_xlabel('conviction ratio w1/w2 (equal numbers, 18 vs 18)')
ax.set_ylabel('accuracy toward stronger group (g1)')
ax.set_title('Exp A: does stronger conviction win at equal numbers?')
ax.set_ylim(-0.2, 1.05); ax.grid(alpha=0.3)

ax = axes[1]
pulls = [expB[w][3] for w in W1B]
acc = [expB[w][0] for w in W1B]; err = [expB[w][1] for w in W1B]
ax.axhline(0, ls=':', color='gray')
ax.axvline(26, ls='--', color='red', alpha=0.6, label='pull balance (n1*w1 = n2*w2 = 26)')
ax.errorbar(pulls, acc, yerr=err, marker='s', capsize=4, lw=2, color='navy')
ax.set_xlabel('total pull of minority group 1 (n1*w1), vs group 2 pull = 26')
ax.set_ylabel('accuracy toward minority (g1, 10 strong agents)')
ax.set_title('Exp B: can 10 strong beat 26 weak? product-law test')
ax.set_ylim(-1.05, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('figures/conviction_1.png', dpi=120)
plt.close()
print('\n  --> figures/conviction_1.png')
print('\nConviction-vs-numbers analysis complete.')
