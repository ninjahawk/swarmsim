# het_recovery_spatial.py -- does spatial vaccination work when the population
# is heterogeneous in recovery rate?
#
# F37 ruled out spatial (farthest-point) vaccination because kinematic mixing
# scrambles the geometric separation faster than the epidemic resolves.  That
# experiment used homogeneous gamma.  This script asks whether the F37 null
# survives heterogeneous recovery, where the epidemic now persists on a
# subset of agents (the slow recoverers, per F54).
#
# Two hypotheses compete:
#   H1 (null transfers) -- mixing is dimension/state-independent; the spatial
#       coverage erodes equally fast whether or not gamma is heterogeneous,
#       so spatial vaccination remains no better than random.
#   H2 (heterogeneity helps spatial) -- because the epidemic localises on
#       slow recoverers, blocking their spatial neighbourhoods (random
#       targeting is enough since slow agents are spatially uniform) may
#       provide larger gains... actually this also predicts no advantage for
#       SPATIAL specifically, since slow agents are randomly placed.
#
# So the prediction from F37 + F54 is that spatial-vs-random parity holds.
# A failure would indicate the kinematic-erosion explanation is incomplete.
#
# We compare random vs spatial (3D-style farthest-point sampling on initial
# positions) vs slow-targeted at matched p_immune.

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
SPREAD      = 0.8
BETA        = 0.30
P_IMMUNE_VALS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def make_gamma(N, spread, rng):
    gamma = np.full(N, GAMMA_MEAN, dtype=float)
    is_slow = np.zeros(N, dtype=bool)
    if spread > 0:
        idx = rng.choice(N, size=N // 2, replace=False)
        is_slow[idx] = True
        gamma[is_slow]  = GAMMA_MEAN - spread
        gamma[~is_slow] = GAMMA_MEAN + spread
    return gamma, is_slow


def farthest_point_sample(x, y, n_imm, rng):
    """Greedy farthest-point sampling with periodic distance."""
    N = x.size
    if n_imm <= 0: return np.zeros(N, dtype=bool)
    if n_imm >= N: return np.ones(N, dtype=bool)
    chosen = [int(rng.randint(N))]
    dx = x - x[chosen[0]]; dy = y - y[chosen[0]]
    dx -= np.round(dx); dy -= np.round(dy)
    dmin = dx**2 + dy**2
    for _ in range(n_imm - 1):
        nxt = int(np.argmax(dmin))
        chosen.append(nxt)
        dx = x - x[nxt]; dy = y - y[nxt]
        dx -= np.round(dx); dy -= np.round(dy)
        d2 = dx**2 + dy**2
        dmin = np.minimum(dmin, d2)
    immune = np.zeros(N, dtype=bool)
    immune[chosen] = True
    return immune


def pick_immune(strategy, gamma_arr, is_slow, x_init, y_init, p_immune, seed):
    rng = np.random.RandomState(1000 + seed)
    N = gamma_arr.size
    n_imm = int(round(p_immune * N))
    immune = np.zeros(N, dtype=bool)
    if n_imm == 0: return immune
    if strategy == 'random':
        idx = rng.choice(N, size=n_imm, replace=False)
        immune[idx] = True
    elif strategy == 'spatial':
        immune = farthest_point_sample(x_init, y_init, n_imm, rng)
    elif strategy == 'slow':
        jitter = rng.uniform(0., 1e-9, N)
        order = np.argsort(gamma_arr + jitter)
        immune[order[:n_imm]] = True
    else:
        raise ValueError(strategy)
    return immune


def run_sis(strategy, p_immune, seed=None):
    rng = np.random.RandomState(seed)
    p = BASE.copy()
    N = p['N']; dt = p['dt']; n_iter = p['n_iter']
    r0, eps, rf = p['r0'], p['eps'], p['rf']
    v0, mu = p['v0'], p['mu']
    frame_every = max(1, n_iter // 200)

    gamma_arr, is_slow = make_gamma(N, SPREAD, rng)
    p_recover_per_step = 1. - np.exp(-gamma_arr * dt)

    x  = np.zeros(2*N)
    x[:N] = rng.uniform(0., 1., N); x[N:] = rng.uniform(0., 1., N)
    vx = rng.uniform(-1., 1., N) * v0
    vy = rng.uniform(-1., 1., N) * v0

    immune = pick_immune(strategy, gamma_arr, is_slow,
                          x[:N].copy(), x[N:].copy(), p_immune, seed)

    f0 = 0.05
    n0 = max(1, round(f0 * N))
    is_panicked = np.zeros(N, dtype=bool)
    nonimm = np.where(~immune)[0]
    if nonimm.size:
        n0 = min(n0, nonimm.size)
        is_panicked[rng.choice(nonimm, size=n0, replace=False)] = True

    rb = max(r0, rf, R_CONT)
    f_history = []
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
                p_trans = 1. - np.exp(-BETA * k[calm_idx] * dt)
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
            f_history.append(is_panicked.mean())
    return np.array(f_history)


STRATS = ['random', 'spatial', 'slow']

print('F57: het-recovery + spatial vaccination interaction')
print('  beta=%.2f spread=%.2f %d seeds' % (BETA, SPREAD, N_SEEDS))
results = {s: {} for s in STRATS}
for strat in STRATS:
    print('  -- %s --' % strat)
    for p_imm in P_IMMUNE_VALS:
        fs = []
        for seed in range(N_SEEDS):
            hist = run_sis(strat, p_imm, seed=seed)
            f_ss = hist[-30:].mean()
            fs.append(f_ss)
        results[strat][p_imm] = (np.mean(fs), np.std(fs))
        print('     p_imm=%.2f  f_ss=%.3f +/- %.3f' % (p_imm, np.mean(fs), np.std(fs)))

fig, ax = plt.subplots(1, 1, figsize=(7, 5))
colors = dict(random='gray', spatial='steelblue', slow='crimson')
for strat in STRATS:
    fv  = [results[strat][p][0] for p in P_IMMUNE_VALS]
    fvs = [results[strat][p][1] for p in P_IMMUNE_VALS]
    ax.errorbar(P_IMMUNE_VALS, fv, yerr=fvs, fmt='o-', color=colors[strat],
                lw=2, capsize=3, label=strat)
ax.set_xlabel('p_immune'); ax.set_ylabel('steady-state panic frac')
ax.set_title('Vaccination in heterogeneous-recovery flock (beta=%.2f)' % BETA)
ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3); ax.legend()
plt.tight_layout()
plt.savefig('figures/het_recovery_spatial_1.png', dpi=120)
plt.close()
print('\n  --> figures/het_recovery_spatial_1.png')
print('\nF57 analysis complete.')
