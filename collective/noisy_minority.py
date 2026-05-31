# noisy_minority.py -- closes the F81/F82/F83 trilogy: does a NOISY MINORITY
# behave like Couzin's informed minority (number fails, F81) or like the
# many-wrongs crowd (number suffices via averaging, F82)?
#
# F81: an EXACT-vector minority of fixed NUMBER fails as N grows -- accuracy is set
#      by the FRACTION, because steering is per-capita pull n_lead*w/N.
# F82: when EVERY agent holds an independent noisy estimate, averaging gives
#      1/sqrt(N) amplification -- number helps.
# The literature treats "informed minority" (Couzin 2005) and "many wrongs"
# (Simons 2004, Codling 2007) as distinct mechanisms. This experiment puts them in
# one setup: a fixed NUMBER n_lead of informed agents, each with a PRIVATE NOISY
# estimate (angle ~ N(0, sigma_pref)), the rest naive followers with no bias.
#
# Two scales pull opposite ways as N grows:
#   - the DIRECTION of the injected pull = the pooled estimate of n_lead agents,
#     with error ~ sigma/sqrt(n_lead) -- FIXED (independent of N);
#   - the MAGNITUDE of the pull per capita = n_lead*w/N -- DILUTES with N (F81).
# Prediction: dilution dominates, so accuracy still FALLS with N (number does NOT
# suffice for a minority); the minority's internal averaging only sets a ceiling on
# how accurate the injected direction can be. Many-wrongs amplification needs the
# informed FRACTION to grow (F82), not a fixed minority. Noisy minority <= exact
# minority (F81) always, equal at sigma=0.
#
# Exp1: sweep N at fixed n_lead, exact (sigma=0, reproduces F81) vs noisy (sigma=1).
# Exp2: sweep n_lead at fixed N for the noisy minority.

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
W_LEAD   = 1.0
G_HAT    = np.array([1.0, 0.0])   # true goal (+x)


def run(N, n_lead, sigma_pref, seed):
    rng = np.random.RandomState(seed)
    p = BASE; dt = p['dt']
    r0, eps, rf = p['r0'], p['eps'], p['rf']
    v0, mu, alpha = p['v0'], p['mu'], p['alpha']

    informed = np.zeros(N, dtype=bool)
    gx = np.zeros(N); gy = np.zeros(N)
    if n_lead > 0:
        sel = rng.choice(N, size=min(n_lead, N), replace=False)
        informed[sel] = True
        phi = rng.normal(0.0, sigma_pref, sel.size)   # each informed agent's private estimate
        gx[sel] = np.cos(phi); gy[sel] = np.sin(phi)

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

        fx = flockx + repx + fpropx + frandx + W_LEAD * gx   # gx,gy zero on naive agents
        fy = flocky + repy + fpropy + frandy + W_LEAD * gy

        vx += fx * dt; vy += fy * dt
        x[:N] = (x[:N] + vx*dt) % 1.; x[N:] = (x[N:] + vy*dt) % 1.

        if i >= N_WARMUP:
            mvx = vx.mean(); mvy = vy.mean(); mn = np.hypot(mvx, mvy)
            acc_rec.append((mvx*G_HAT[0] + mvy*G_HAT[1])/mn if mn > 1e-9 else 0.0)
            phi_rec.append(order_parameter(vx, vy))
    return float(np.mean(acc_rec)), float(np.std(acc_rec)), float(np.mean(phi_rec))


def sweep_seeds(N, n_lead, sigma_pref):
    accs = []; phis = []
    for s in range(N_SEEDS):
        a, _, ph = run(N, n_lead, sigma_pref, s)
        accs.append(a); phis.append(ph)
    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(phis))


print('Noisy minority: does a fixed NUMBER of noisy-informed agents suffice as N grows?')
print('  w_lead=%.1f  %d seeds  pure-flock\n' % (W_LEAD, N_SEEDS))

# --- Exp1: sweep N at fixed n_lead, exact vs noisy minority ---
N_VALS = [100, 250, 500]
NLEAD_VALS = [10, 20]
SIGMAS = [0.0, 1.0]   # 0 = exact (reproduces F81); 1.0 rad = noisy
print('== Exp1: accuracy vs N at fixed n_lead (exact sigma=0 vs noisy sigma=1.0 rad) ==')
res1 = {}
for sg in SIGMAS:
    tag = 'EXACT (=F81)' if sg == 0.0 else 'NOISY sigma=1.0'
    print('  -- %s --' % tag)
    for nl in NLEAD_VALS:
        line = []
        for N in N_VALS:
            acc, sd, phi = sweep_seeds(N, nl, sg)
            res1[(sg, nl, N)] = (acc, sd, phi)
            line.append('N=%d: %+.3f' % (N, acc))
        print('     n_lead=%2d   ' % nl + '   '.join(line)
              + '   (frac %.3f->%.3f)' % (nl/N_VALS[0], nl/N_VALS[-1]))
    print()

print('  Reading: if accuracy FALLS with N at fixed n_lead, the NUMBER does not suffice')
print('  (per-capita pull dilution dominates, F81). Noisy should sit at or below exact.\n')

# --- Exp2: sweep n_lead at fixed N, noisy minority ---
N_FIX = 250
NLEAD_SWEEP = [5, 10, 20, 40, 80]
print('== Exp2: noisy minority (sigma=1.0 rad), sweep n_lead at N=%d ==' % N_FIX)
res2 = {}
for nl in NLEAD_SWEEP:
    acc, sd, phi = sweep_seeds(N_FIX, nl, 1.0)
    res2[nl] = (acc, sd, phi)
    print('   n_lead=%2d (frac=%.3f)  accuracy=%+.3f +/- %.3f  Phi=%.3f'
          % (nl, nl/N_FIX, acc, sd, phi))
print('   (more leaders = more per-capita pull AND a better-pooled direction; both help)\n')

# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Noisy minority: a fixed NUMBER of noisy leaders still fails as N grows '
             '(w_lead=%.1f, %d seeds)' % (W_LEAD, N_SEEDS), fontsize=11)
ax = axes[0]
styles = {0.0: ('-', 'exact'), 1.0: ('--', 'noisy s=1.0')}
cols = {10: 'steelblue', 20: 'crimson'}
for sg in SIGMAS:
    ls, lab = styles[sg]
    for nl in NLEAD_VALS:
        acc = [res1[(sg, nl, N)][0] for N in N_VALS]
        err = [res1[(sg, nl, N)][1] for N in N_VALS]
        ax.errorbar(N_VALS, acc, yerr=err, marker='o', ls=ls, capsize=3, lw=2,
                    color=cols[nl], label='n_lead=%d %s' % (nl, lab))
ax.set_xlabel('flock size N'); ax.set_ylabel('accuracy toward true goal')
ax.set_title('Accuracy at FIXED leader NUMBER falls with N (exact AND noisy)')
ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=8)

ax = axes[1]
acc2 = [res2[nl][0] for nl in NLEAD_SWEEP]; err2 = [res2[nl][1] for nl in NLEAD_SWEEP]
ax.errorbar(NLEAD_SWEEP, acc2, yerr=err2, marker='o', capsize=4, lw=2, color='darkorange')
ax.set_xlabel('number of noisy leaders n_lead'); ax.set_ylabel('accuracy toward true goal')
ax.set_title('Noisy minority at N=%d: accuracy rises with n_lead' % N_FIX)
ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/noisy_minority_1.png', dpi=120)
plt.close()
print('  --> figures/noisy_minority_1.png')
print('\nNoisy-minority analysis complete.')
