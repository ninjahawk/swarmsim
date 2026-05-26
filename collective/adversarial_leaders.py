# adversarial_leaders.py -- SABOTEUR leaders: denial vs capture.
#
# F73/F74 showed two opposed informed subgroups resolve by majority/total-pull. Here we frame it
# adversarially: a fixed set of TRUE leaders (fraction rho_true) steer toward a goal (+x), and a
# swept set of SABOTEURS (rho_sab) push toward a trap (-x). Two distinct adversarial objectives:
#   DENIAL  -- stop the flock reaching its goal (drive accuracy-toward-goal to ~0); a deadlock.
#   CAPTURE -- actively drive the flock to the trap (accuracy-toward-trap > 0).
# Question: are these equally hard? Hypothesis from F73/F74: DENIAL needs only PARITY of pull
# (equal saboteurs tie the flock into a random/deadlocked heading), but CAPTURE needs a MAJORITY
# (the saboteur pull must exceed the true pull to actually win the consensus). If so, "denial is
# cheaper than capture" -- a security asymmetry: a small adversary can paralyze a led flock long
# before it can hijack it.
#
# Fixed rho_true=0.10 toward +x; sweep rho_sab toward -x. Equal leader strength w=1.0.
# Metric: accuracy toward TRUE goal (+x) = cos(heading); negative => heading toward trap.

import os
import numpy as np
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from flocking import buffer as buf_fn, order_parameter

os.makedirs('figures', exist_ok=True)
N_SEEDS = 6

BASE = dict(N=350, r0=0.005, eps=0.1, rf=0.1, alpha=1.0, v0=1.0, mu=10.0, ramp=0.5, dt=0.01)
N_WARMUP = 1000
N_ITER   = 4000
W_LEAD   = 1.0
RHO_TRUE = 0.10
G_TRUE = np.array([1.0, 0.0])
G_TRAP = np.array([-1.0, 0.0])


def run(rho_sab, seed):
    rng = np.random.RandomState(seed)
    p = BASE; N = p['N']; dt = p['dt']
    r0, eps, rf = p['r0'], p['eps'], p['rf']
    v0, mu, alpha = p['v0'], p['mu'], p['alpha']

    n_true = int(round(RHO_TRUE * N)); n_sab = int(round(rho_sab * N))
    perm = rng.permutation(N)
    true_m = np.zeros(N, dtype=bool); sab_m = np.zeros(N, dtype=bool)
    true_m[perm[:n_true]] = True
    sab_m[perm[n_true:n_true + n_sab]] = True

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

        fx[true_m] += W_LEAD * G_TRUE[0]; fy[true_m] += W_LEAD * G_TRUE[1]
        if n_sab > 0:
            fx[sab_m] += W_LEAD * G_TRAP[0]; fy[sab_m] += W_LEAD * G_TRAP[1]

        vx += fx * dt; vy += fy * dt
        x[:N] = (x[:N] + vx*dt) % 1.; x[N:] = (x[N:] + vy*dt) % 1.

        if i >= N_WARMUP:
            mvx = vx.mean(); mvy = vy.mean(); mn = np.hypot(mvx, mvy)
            acc_rec.append((mvx*G_TRUE[0] + mvy*G_TRUE[1])/mn if mn > 1e-9 else 0.0)
            phi_rec.append(order_parameter(vx, vy))
    return float(np.mean(acc_rec)), float(np.std(acc_rec)), float(np.mean(phi_rec))


print('Adversarial leadership: denial vs capture (true leaders +x at rho=%.2f vs saboteurs -x)' % RHO_TRUE)
print('  N=%d  pure-flock  w_lead=%.1f  %d seeds\n' % (BASE['N'], W_LEAD, N_SEEDS))

RHO_SAB = [0.0, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30]
results = {}
for rs in RHO_SAB:
    accs = []; phis = []
    for s in range(N_SEEDS):
        a, _, ph = run(rs, s)
        accs.append(a); phis.append(ph)
    results[rs] = (np.mean(accs), np.std(accs), np.mean(phis))
    # classify outcome
    m = np.mean(accs)
    tag = 'GOAL'   if m > 0.5 else ('CAPTURED(trap)' if m < -0.3 else 'DENIED/deadlock')
    print('   rho_sab=%.3f (%2d sab vs %2d true)  acc_to_goal=%+.3f +/- %.3f  Phi=%.3f  [%s]'
          % (rs, int(round(rs*BASE['N'])), int(round(RHO_TRUE*BASE['N'])),
             np.mean(accs), np.std(accs), np.mean(phis), tag))

# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5.5))
fig.suptitle('Adversarial leadership: denial vs capture (rho_true=%.2f, N=%d, %d seeds)'
             % (RHO_TRUE, BASE['N'], N_SEEDS), fontsize=11)
acc = [results[r][0] for r in RHO_SAB]; err = [results[r][1] for r in RHO_SAB]
ax.axhline(0, ls=':', color='gray')
ax.axvline(RHO_TRUE, ls='--', color='red', alpha=0.6, label='pull parity (rho_sab=rho_true)')
ax.axhspan(-0.3, 0.5, color='gold', alpha=0.12, label='denial/deadlock band')
ax.errorbar(RHO_SAB, acc, yerr=err, marker='o', capsize=4, lw=2, color='navy')
ax.set_xlabel('saboteur fraction rho_sab (toward trap, -x)')
ax.set_ylabel('accuracy toward TRUE goal (+x)')
ax.set_ylim(-1.05, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('figures/adversarial_leaders_1.png', dpi=120)
plt.close()
print('\n  --> figures/adversarial_leaders_1.png')
print('\nAdversarial-leadership analysis complete.')
