# rare_reservoir_vaccination.py -- slow-recoverer vaccination when the slow
# class is rare (Finding 61)
#
# F56-F59 established slow-recoverer targeting with the slow class fixed at
# 50% of the population.  In realistic settings the reservoir-prone minority
# may be much smaller (e.g. immunocompromised individuals, ~5-15% of the
# population).  How does the slow-targeting advantage behave when the slow
# class is rare?
#
# Two effects compete.  (1) As the slow fraction shrinks, RANDOM is more
# likely to miss the reservoir entirely -- the slow-targeting advantage should
# grow.  (2) But when the slow fraction is below p_immune, slow-targeting
# eventually exhausts the slow class and starts immunising fast agents that
# don't matter -- the advantage should plateau.
#
# Setup.  Per-agent gamma_i bimodal with f_slow fraction at gamma_slow=0.1
# (deep reservoirs); fast class at gamma_fast set so the arithmetic mean
# stays 1.0.  Sweep f_slow from 0.05 to 0.50.  Compare random and slow
# vaccination at p_imm equal to f_slow ("target exactly the slow class") and
# at p_imm = 0.30 (fixed budget, matches F56).

import os
import numpy as np
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from flocking import buffer as buf_fn, order_parameter

os.makedirs('figures', exist_ok=True)

N_SEEDS = 4
BASE = dict(
    N=350, r0=0.005, eps=0.1, rf=0.1,
    alpha=1.0, v0=1.0, mu=10.0, ramp=0.5, dt=0.01,
    n_iter=4000,
)
PANIC_ALPHA = 0.1
PANIC_RAMP  = 10.0
R_CONT      = 0.05
GAMMA_MEAN  = 1.0
GAMMA_SLOW  = 0.1
BETA        = 0.30


def make_gamma(N, f_slow, rng):
    """Bimodal with f_slow of population at gamma_slow=0.1; remainder at
    gamma_fast set so arithmetic mean = GAMMA_MEAN."""
    n_slow = max(1, int(round(f_slow * N)))
    gamma_fast = (GAMMA_MEAN - f_slow * GAMMA_SLOW) / max(1e-6, (1 - f_slow))
    gamma = np.full(N, gamma_fast, dtype=float)
    is_slow = np.zeros(N, dtype=bool)
    idx = rng.choice(N, size=n_slow, replace=False)
    is_slow[idx] = True
    gamma[is_slow] = GAMMA_SLOW
    return gamma, is_slow


def run_sis(beta, gamma_arr, immune, seed):
    rng = np.random.RandomState(seed)
    p = BASE.copy()
    N = p['N']; dt = p['dt']; n_iter = p['n_iter']
    r0, eps, rf = p['r0'], p['eps'], p['rf']
    v0, mu = p['v0'], p['mu']
    frame_every = max(1, n_iter // 200)
    p_recover_per_step = 1. - np.exp(-gamma_arr * dt)
    n0 = max(1, round(0.05 * N))
    is_panicked = np.zeros(N, dtype=bool)
    nonimm = np.where(~immune)[0]
    if nonimm.size:
        n0 = min(n0, nonimm.size)
        is_panicked[rng.choice(nonimm, size=n0, replace=False)] = True
    x  = np.zeros(2*N)
    x[:N] = rng.uniform(0., 1., N); x[N:] = rng.uniform(0., 1., N)
    vx = rng.uniform(-1., 1., N) * v0
    vy = rng.uniform(-1., 1., N) * v0
    rb = max(r0, rf, R_CONT)
    f_hist = []
    for i in range(n_iter):
        nb, xb, yb, vxb, vyb = buf_fn(rb, x, vx, vy, N)
        dx = xb[:nb][np.newaxis, :] - x[:N][:, np.newaxis]
        dy = yb[:nb][np.newaxis, :] - x[N:][:, np.newaxis]
        d2 = dx**2 + dy**2
        not_self = np.ones((N, nb), dtype=bool)
        idx0 = np.arange(min(N, nb)); not_self[idx0, idx0] = False
        alpha_arr = np.where(is_panicked, PANIC_ALPHA, p['alpha'])
        ramp_arr  = np.where(is_panicked, PANIC_RAMP,  p['ramp'])
        flock_mask = (d2 <= rf**2) & not_self
        flx = np.where(flock_mask, vxb[:nb], 0.).sum(axis=1)
        fly = np.where(flock_mask, vyb[:nb], 0.).sum(axis=1)
        nfl = flock_mask.sum(axis=1)
        nrm = np.sqrt(flx**2 + fly**2); nrm[nfl == 0] = 1.
        flockx = alpha_arr * flx / nrm; flocky = alpha_arr * fly / nrm
        rep_mask = (d2 <= (2*r0)**2) & not_self & (d2 > 0)
        d_safe = np.where(rep_mask, np.sqrt(d2), 1.)
        base = np.where(rep_mask, 1. - d_safe/(2.*r0), 0.)
        strength = np.where(rep_mask, eps * base**1.5 / d_safe, 0.)
        repx = (-strength * dx).sum(axis=1); repy = (-strength * dy).sum(axis=1)
        vnorm = np.sqrt(vx**2 + vy**2)
        vnorms = np.where(vnorm == 0, 1., vnorm)
        fpropx = mu * (v0 - vnorm) * vx / vnorms
        fpropy = mu * (v0 - vnorm) * vy / vnorms
        frandx = ramp_arr * rng.uniform(-1., 1., N)
        frandy = ramp_arr * rng.uniform(-1., 1., N)
        fx = flockx + repx + fpropx + frandx
        fy = flocky + repy + fpropy + frandy
        vx += fx * dt; vy += fy * dt
        x[:N] = (x[:N] + vx*dt) % 1.; x[N:] = (x[N:] + vy*dt) % 1.
        if is_panicked.any() and (~is_panicked & ~immune).any():
            rdx = x[:N][np.newaxis, :] - x[:N][:, np.newaxis]
            rdy = x[N:][np.newaxis, :] - x[N:][:, np.newaxis]
            rdx -= np.round(rdx); rdy -= np.round(rdy)
            rd2 = rdx**2 + rdy**2
            within = (rd2 <= R_CONT**2) & (rd2 > 0)
            k = within @ is_panicked.astype(np.int32)
            calm_idx = np.where(~is_panicked & ~immune)[0]
            if calm_idx.size:
                p_trans = 1. - np.exp(-beta * k[calm_idx] * dt)
                r = rng.uniform(0., 1., calm_idx.size)
                flipped = calm_idx[r < p_trans]
                if flipped.size:
                    is_panicked[flipped] = True
        if is_panicked.any():
            pidx = np.where(is_panicked)[0]
            r = rng.uniform(0., 1., pidx.size)
            recovered = pidx[r < p_recover_per_step[pidx]]
            if recovered.size:
                is_panicked[recovered] = False
        if i % frame_every == 0:
            f_hist.append(is_panicked.mean())
    return np.array(f_hist)


print('F61: rare-reservoir slow-recoverer vaccination')
print('  beta=%.2f, mean gamma=%.2f, gamma_slow=%.2f, %d seeds' % (
    BETA, GAMMA_MEAN, GAMMA_SLOW, N_SEEDS))

f_slow_vals = [0.05, 0.10, 0.20, 0.30, 0.50]

# Exp 1: p_imm = f_slow (target exactly the reservoir class)
print('\nExp 1: p_imm = f_slow (target exactly the reservoir size)')
res1 = {}
for f_slow in f_slow_vals:
    p_imm = f_slow
    fs_r, fs_s = [], []
    for seed in range(N_SEEDS):
        rng_g = np.random.RandomState(seed)
        gamma_arr, is_slow = make_gamma(BASE['N'], f_slow, rng_g)
        n_imm = int(round(p_imm * BASE['N']))
        rng_pick = np.random.RandomState(3000 + seed)
        im_r = np.zeros(BASE['N'], dtype=bool)
        im_r[rng_pick.choice(BASE['N'], size=n_imm, replace=False)] = True
        order = np.argsort(gamma_arr + rng_pick.uniform(0., 1e-9, BASE['N']))
        im_s = np.zeros(BASE['N'], dtype=bool); im_s[order[:n_imm]] = True
        fs_r.append(run_sis(BETA, gamma_arr, im_r, seed)[-30:].mean())
        fs_s.append(run_sis(BETA, gamma_arr, im_s, seed)[-30:].mean())
    res1[f_slow] = (np.mean(fs_r), np.std(fs_r), np.mean(fs_s), np.std(fs_s))
    print('  f_slow=%.2f gamma_fast=%.3f p_imm=%.2f  random=%.3f+/-%.3f  '
          'slow=%.3f+/-%.3f' % (
        f_slow, (GAMMA_MEAN - f_slow*GAMMA_SLOW)/(1-f_slow), p_imm,
        res1[f_slow][0], res1[f_slow][1],
        res1[f_slow][2], res1[f_slow][3]))

# Exp 2: fixed p_imm = 0.30 (F56 budget) for all f_slow
print('\nExp 2: p_imm = 0.30 fixed (F56-budget across rarity sweep)')
P_IMM_FIX = 0.30
res2 = {}
for f_slow in f_slow_vals:
    fs_r, fs_s = [], []
    for seed in range(N_SEEDS):
        rng_g = np.random.RandomState(seed)
        gamma_arr, is_slow = make_gamma(BASE['N'], f_slow, rng_g)
        n_imm = int(round(P_IMM_FIX * BASE['N']))
        rng_pick = np.random.RandomState(5000 + seed)
        im_r = np.zeros(BASE['N'], dtype=bool)
        im_r[rng_pick.choice(BASE['N'], size=n_imm, replace=False)] = True
        order = np.argsort(gamma_arr + rng_pick.uniform(0., 1e-9, BASE['N']))
        im_s = np.zeros(BASE['N'], dtype=bool); im_s[order[:n_imm]] = True
        fs_r.append(run_sis(BETA, gamma_arr, im_r, seed)[-30:].mean())
        fs_s.append(run_sis(BETA, gamma_arr, im_s, seed)[-30:].mean())
    res2[f_slow] = (np.mean(fs_r), np.std(fs_r), np.mean(fs_s), np.std(fs_s))
    print('  f_slow=%.2f  random=%.3f+/-%.3f  slow=%.3f+/-%.3f' % (
        f_slow,
        res2[f_slow][0], res2[f_slow][1],
        res2[f_slow][2], res2[f_slow][3]))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Rare-reservoir vaccination (gamma_slow=0.1, beta=%.2f, %d seeds)'
             % (BETA, N_SEEDS), fontsize=11)
ax = axes[0]
fr_v = [res1[f][0] for f in f_slow_vals]; fr_s = [res1[f][1] for f in f_slow_vals]
fs_v = [res1[f][2] for f in f_slow_vals]; fs_s = [res1[f][3] for f in f_slow_vals]
ax.errorbar(f_slow_vals, fr_v, yerr=fr_s, fmt='o-', color='gray', lw=2,
            capsize=3, label='random')
ax.errorbar(f_slow_vals, fs_v, yerr=fs_s, fmt='o-', color='crimson', lw=2,
            capsize=3, label='slow (p_imm=f_slow)')
ax.set_xlabel('reservoir fraction f_slow (= p_imm)')
ax.set_ylabel('f_ss')
ax.set_title('Targeting exactly the reservoir class')
ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3); ax.legend()

ax = axes[1]
fr_v = [res2[f][0] for f in f_slow_vals]; fr_s = [res2[f][1] for f in f_slow_vals]
fs_v = [res2[f][2] for f in f_slow_vals]; fs_s = [res2[f][3] for f in f_slow_vals]
ax.errorbar(f_slow_vals, fr_v, yerr=fr_s, fmt='o-', color='gray', lw=2,
            capsize=3, label='random')
ax.errorbar(f_slow_vals, fs_v, yerr=fs_s, fmt='o-', color='crimson', lw=2,
            capsize=3, label='slow')
ax.set_xlabel('reservoir fraction f_slow')
ax.set_ylabel('f_ss')
ax.set_title('Fixed budget p_imm=0.30')
ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3); ax.legend()

plt.tight_layout()
plt.savefig('figures/rare_reservoir_vaccination_1.png', dpi=120)
plt.close()
print('\n  --> figures/rare_reservoir_vaccination_1.png')
print('\nF61 complete.')
