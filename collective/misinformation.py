# misinformation.py -- how robust is crowd navigation to misinformed members,
# and does it matter whether their error is RANDOM or COORDINATED?
#
# The many-wrongs results (F82-F84) show alignment is a directional averager whose
# accuracy is set by the correlation structure of the estimates. A key practical
# corollary: each agent contributes a UNIT vector g_hat_i, so any one agent's
# influence is BOUNDED no matter how wrong its angle -- directional averaging is
# intrinsically robust to outliers. This experiment quantifies that robustness and
# contrasts two kinds of misinformation in a fraction f of the flock:
#   'lost'        : misinformed agents point in UNIFORM-RANDOM directions
#                   (uncoordinated -- their unit votes should cancel, ~1/sqrt(f N)).
#   'adversarial' : misinformed agents all point at a FALSE goal (-x), coordinated
#                   (a shared false cue -- competes with the true informed vote-for-vote).
# The rest (fraction 1-f) are well-informed: tight Gaussian (sigma_in) about +x.
#
# Prediction: random misinformation averages out, so accuracy stays high until f is
# large and then collapses when the net informed signal (1-f)*w*exp(-sigma_in^2/2)
# drops below the spontaneous-heading threshold (the F82 noise ceiling). Coordinated
# misinformation is far more damaging: net pull ~ (1-f) - f = 1-2f, so accuracy
# crosses zero at PARITY f=0.5 (the F80 product law). Uncoordinated noise << coordinated falsehood.
#
# Exp1: sweep f at fixed N for both modes. Exp2: lost mode, sweep N at fixed f
# (does the crowd average out a fixed fraction of lost members regardless of size?).

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
N_ITER   = 3000
W_BIAS   = 0.5
SIGMA_IN = 0.3     # rad, tight error of the well-informed majority


def run(N, f_mis, mode, seed):
    rng = np.random.RandomState(seed)
    p = BASE; dt = p['dt']
    r0, eps, rf = p['r0'], p['eps'], p['rf']
    v0, mu, alpha = p['v0'], p['mu'], p['alpha']

    n_mis = int(round(f_mis * N))
    mis = np.zeros(N, dtype=bool)
    if n_mis > 0:
        mis[rng.choice(N, size=n_mis, replace=False)] = True

    phi = np.empty(N)
    # well-informed majority: tight Gaussian about the true goal (+x, angle 0)
    phi[~mis] = rng.normal(0.0, SIGMA_IN, (~mis).sum())
    # misinformed minority
    if n_mis > 0:
        if mode == 'lost':
            phi[mis] = rng.uniform(-np.pi, np.pi, n_mis)          # uncoordinated
        else:  # 'adversarial' -- coordinated toward false goal -x (angle pi)
            phi[mis] = np.pi + rng.normal(0.0, SIGMA_IN, n_mis)
    gx = np.cos(phi); gy = np.sin(phi)

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

        fx = flockx + repx + fpropx + frandx + W_BIAS * gx
        fy = flocky + repy + fpropy + frandy + W_BIAS * gy

        vx += fx * dt; vy += fy * dt
        x[:N] = (x[:N] + vx*dt) % 1.; x[N:] = (x[N:] + vy*dt) % 1.

        if i >= N_WARMUP:
            mvx = vx.mean(); mvy = vy.mean(); mn = np.hypot(mvx, mvy)
            acc_rec.append(mvx/mn if mn > 1e-9 else 0.0)   # cos to true goal +x
            phi_rec.append(order_parameter(vx, vy))
    return float(np.mean(acc_rec)), float(np.std(acc_rec)), float(np.mean(phi_rec))


def sweep_seeds(N, f_mis, mode):
    accs = []; phis = []
    for s in range(N_SEEDS):
        a, _, ph = run(N, f_mis, mode, s)
        accs.append(a); phis.append(ph)
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(phis))


print('Misinformation robustness: random (lost) vs coordinated (adversarial) misinformed members')
print('  N=250  w_bias=%.1f  sigma_in=%.2f rad  %d seeds\n' % (W_BIAS, SIGMA_IN, N_SEEDS))

# --- Exp1: sweep f for both modes at N=250 ---
N_FIX = 250
F_VALS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
res1 = {}
for mode in ['lost', 'adversarial']:
    print('== mode=%s ==' % mode)
    for f in F_VALS:
        acc, sd, phi = sweep_seeds(N_FIX, f, mode)
        res1[(mode, f)] = (acc, sd, phi)
        print('   f_mis=%.2f  accuracy=%+.3f +/- %.3f  Phi=%.3f' % (f, acc, sd, phi))
    print()

print('  Reading: lost (uncoordinated) votes cancel -> robust to large f; adversarial')
print('  (coordinated) competes vote-for-vote -> accuracy crosses zero near parity f=0.5.\n')

# --- Exp2: lost mode, sweep N at fixed f ---
F_FIX = 0.4
N_VALS = [60, 125, 250, 500]
print('== Exp2: mode=lost, f_mis=%.2f, sweep N ==' % F_FIX)
res2 = {}
for N in N_VALS:
    acc, sd, phi = sweep_seeds(N, F_FIX, 'lost')
    res2[N] = (acc, sd, phi)
    print('   N=%4d  accuracy=%+.3f +/- %.3f  Phi=%.3f' % (N, acc, sd, phi))
print('   (if the crowd averages out lost members, accuracy ~ N-independent at fixed f)\n')

# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Robustness of crowd navigation to misinformation '
             '(N=%d, w_bias=%.1f, %d seeds)' % (N_FIX, W_BIAS, N_SEEDS), fontsize=11)
ax = axes[0]
for mode, col in zip(['lost', 'adversarial'], ['seagreen', 'crimson']):
    acc = [res1[(mode, f)][0] for f in F_VALS]
    err = [res1[(mode, f)][1] for f in F_VALS]
    ax.errorbar(F_VALS, acc, yerr=err, marker='o', capsize=4, lw=2, color=col, label=mode)
ax.axhline(0, ls=':', color='gray'); ax.axvline(0.5, ls=':', color='crimson', alpha=0.5)
ax.set_xlabel('misinformed fraction f'); ax.set_ylabel('accuracy toward true goal')
ax.set_title('Random misinformation averages out; coordinated flips at parity')
ax.set_ylim(-1.05, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=9)

ax = axes[1]
acc2 = [res2[N][0] for N in N_VALS]; err2 = [res2[N][1] for N in N_VALS]
ax.errorbar(N_VALS, acc2, yerr=err2, marker='o', capsize=4, lw=2, color='seagreen')
ax.set_xscale('log'); ax.set_xlabel('flock size N'); ax.set_ylabel('accuracy toward true goal')
ax.set_title('Lost mode at f=%.1f: accuracy ~ N-independent (votes cancel at any size)' % F_FIX)
ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/misinformation_1.png', dpi=120)
plt.close()
print('  --> figures/misinformation_1.png')
print('\nMisinformation-robustness analysis complete.')
