# herd_immunity.py -- Can immune sub-population suppress SIS contagion?
#
# Standard SIR/SIS result: a fraction p_immune of agents that never become
# panicked reduces the effective <k> by factor (1 - p_immune), shifting the
# epidemic threshold up.  In particular, p_immune = 1 - 1/R0 should bring R0
# back to 1 and quench the outbreak.
#
# We're in the supercritical SIS regime (beta=2.5, gamma=2.0; beta/gamma=1.25
# without immunity).  Sweep p_immune from 0 to 0.6 and find the herd-immunity
# threshold p_c above which f_ss collapses.

import os
import numpy as np
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from flocking import buffer as buf_fn, order_parameter

os.makedirs('figures', exist_ok=True)
N_SEEDS = 5

BASE = dict(
    N=350, r0=0.005, eps=0.1, rf=0.1,
    alpha=1.0, v0=0.02, mu=10.0, ramp=0.1, dt=0.01,
    n_iter=5000,
)
PANIC_ALPHA = 0.1
PANIC_RAMP  = 10.0
R_CONT      = 0.05
BETA        = 2.5
GAMMA       = 2.0


def run_immunity(p_immune, seed):
    np.random.seed(seed)
    p = BASE.copy()
    N = p['N']; dt = p['dt']; n_iter = p['n_iter']
    r0, eps, rf = p['r0'], p['eps'], p['rf']
    v0, mu = p['v0'], p['mu']

    x  = np.zeros(2*N)
    x[:N] = np.random.uniform(0., 1., N)
    x[N:] = np.random.uniform(0., 1., N)
    vx = np.random.uniform(-1., 1., N) * v0
    vy = np.random.uniform(-1., 1., N) * v0

    # immune mask: these agents are flagged calm and never transition
    n_immune = round(p_immune * N)
    is_immune = np.zeros(N, dtype=bool)
    if n_immune > 0:
        immune_idx = np.random.choice(N, size=n_immune, replace=False)
        is_immune[immune_idx] = True

    # initial panic seed: from non-immune subset only
    is_panicked = np.zeros(N, dtype=bool)
    non_immune = np.where(~is_immune)[0]
    n_seed = max(1, round(0.05 * non_immune.size))
    seed_idx = np.random.choice(non_immune, size=n_seed, replace=False)
    is_panicked[seed_idx] = True

    p_recover = 1. - np.exp(-GAMMA * dt)
    rb = max(r0, rf, R_CONT)
    f_t = []

    for i in range(n_iter):
        nb, xb, yb, vxb, vyb = buf_fn(rb, x, vx, vy, N)
        dx = xb[:nb][np.newaxis, :] - x[:N][:, np.newaxis]
        dy = yb[:nb][np.newaxis, :] - x[N:][:, np.newaxis]
        d2 = dx**2 + dy**2

        not_self = np.ones((N, nb), dtype=bool)
        idx = np.arange(min(N, nb))
        not_self[idx, idx] = False

        alpha_arr = np.where(is_panicked, PANIC_ALPHA, p['alpha'])
        ramp_arr  = np.where(is_panicked, PANIC_RAMP,  p['ramp'])

        flock_mask = (d2 <= rf**2) & not_self
        flx = np.where(flock_mask, vxb[:nb], 0.).sum(axis=1)
        fly = np.where(flock_mask, vyb[:nb], 0.).sum(axis=1)
        nfl = flock_mask.sum(axis=1)
        nrm = np.sqrt(flx**2 + fly**2)
        nrm[nfl == 0] = 1.
        flockx = alpha_arr * flx / nrm
        flocky = alpha_arr * fly / nrm

        rep_mask = (d2 <= (2*r0)**2) & not_self & (d2 > 0)
        d_safe   = np.where(rep_mask, np.sqrt(d2), 1.)
        base_r   = np.where(rep_mask, 1. - d_safe/(2.*r0), 0.)
        strength = np.where(rep_mask, eps * base_r**1.5 / d_safe, 0.)
        repx = (-strength * dx).sum(axis=1)
        repy = (-strength * dy).sum(axis=1)

        vnorm  = np.sqrt(vx**2 + vy**2)
        vnorms = np.where(vnorm == 0, 1., vnorm)
        fpropx = mu * (v0 - vnorm) * vx / vnorms
        fpropy = mu * (v0 - vnorm) * vy / vnorms

        frandx = ramp_arr * np.random.uniform(-1., 1., N)
        frandy = ramp_arr * np.random.uniform(-1., 1., N)

        vx += (flockx + repx + fpropx + frandx) * dt
        vy += (flocky + repy + fpropy + frandy) * dt
        x[:N] = (x[:N] + vx*dt) % 1.
        x[N:] = (x[N:] + vy*dt) % 1.

        # contagion + recovery (immune agents skip both)
        susceptible = ~is_panicked & ~is_immune
        if is_panicked.any() and susceptible.any():
            real_dx = x[:N][np.newaxis, :] - x[:N][:, np.newaxis]
            real_dy = x[N:][np.newaxis, :] - x[N:][:, np.newaxis]
            real_dx -= np.round(real_dx); real_dy -= np.round(real_dy)
            rd2 = real_dx**2 + real_dy**2
            within = (rd2 <= R_CONT**2) & (rd2 > 0)
            k_arr = within @ is_panicked.astype(np.int32)
            sus_idx = np.where(susceptible)[0]
            k_s = k_arr[sus_idx]
            p_trans = 1. - np.exp(-BETA * k_s * dt)
            r = np.random.uniform(0., 1., sus_idx.size)
            flipped = sus_idx[r < p_trans]
            if flipped.size:
                is_panicked[flipped] = True
        if is_panicked.any():
            panic_idx = np.where(is_panicked)[0]
            r = np.random.uniform(0., 1., panic_idx.size)
            recovered = panic_idx[r < p_recover]
            if recovered.size:
                is_panicked[recovered] = False

        if i % 50 == 0:
            f_t.append(is_panicked.mean())

    f_ss = np.mean(f_t[-len(f_t)//5:])
    return f_ss


print('Herd-immunity sweep: beta=%.2f gamma=%.2f, %d seeds' % (BETA, GAMMA, N_SEEDS))
p_vals = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]
results = {}
for p in p_vals:
    fs = []
    for s in range(N_SEEDS):
        fs.append(run_immunity(p, s))
    results[p] = (np.mean(fs), np.std(fs))
    print('  p_immune=%.2f  f_ss=%.3f +/- %.3f' % (p, results[p][0], results[p][1]))


# Apparent herd threshold
def threshold(vals):
    for i in range(len(p_vals)-1):
        if vals[i] >= 0.1 > vals[i+1]:
            f0, f1 = vals[i], vals[i+1]
            return p_vals[i] + (vals[i] - 0.1)/(vals[i]-vals[i+1]) * (p_vals[i+1]-p_vals[i])
    return None
fs_arr = [results[p][0] for p in p_vals]
p_c = threshold(fs_arr)
print('\nHerd-immunity threshold (f_ss drops below 0.1): p_c = %s' % p_c)
# theoretical: p_c = 1 - 1/R0 with R0 = beta/gamma = 1.25 -> p_c = 0.2
print('Mean-field prediction (1 - gamma/beta): %.2f' % (1 - GAMMA/BETA))


# Plot
fig, ax = plt.subplots(figsize=(8, 5))
means = [results[p][0] for p in p_vals]
stds  = [results[p][1] for p in p_vals]
ax.errorbar(p_vals, means, yerr=stds, fmt='o-', color='crimson', lw=2, capsize=4,
            label='measured f_ss')
ax.axvline(1 - GAMMA/BETA, ls='--', color='gray', alpha=0.6,
           label='mean-field p_c = 1 - gamma/beta')
ax.set_xlabel('Immune fraction p_immune')
ax.set_ylabel('Steady-state panic fraction f_ss')
ax.set_title('Herd immunity (beta=%.2f, gamma=%.2f; %d seeds)' % (BETA, GAMMA, N_SEEDS))
ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/herd_immunity_1.png', dpi=120)
plt.close()
print('  --> figures/herd_immunity_1.png')
