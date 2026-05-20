# continuous_gamma_vaccination.py -- slow-recoverer vaccination with a
# continuous (lognormal) per-agent gamma distribution (Finding 59)
#
# F56 established slow-recoverer targeting in a BIMODAL gamma population
# {0.2, 1.8} -- a sharp two-class structure where "slow" is unambiguous.  Real
# populations don't have a discrete slow/fast split; their recovery rates are
# continuously distributed.  Does the F56 advantage survive when slowness is
# a quantile rather than a class?
#
# Two hypotheses compete:
#   H1 (continuous works) -- the F56 mechanism is about the lower TAIL of the
#       gamma distribution.  Targeting the lowest-gamma p_immune fraction
#       removes the deepest reservoirs, which sustain the epidemic
#       disproportionately to their share of population.  Should work.
#   H2 (continuous fails) -- in a continuous distribution there is no sharp
#       boundary between slow and fast; the targeted agents are only slightly
#       slower than the next-slowest, and the reservoir effect is diluted.
#       Targeting becomes nearly random.
#
# Setup.  Per-agent gamma_i drawn from a lognormal distribution with arithmetic
# mean 1.0.  Sweep the width (sigma_log) to vary heterogeneity.  At each width,
# compare random vs slow-targeting vs an idealised "knowable" upper bound
# strategy (top-X% by exact gamma_i) at matched p_immune.
#
# Outputs:
#   - random vs slow advantage as a function of sigma_log
#   - threshold beta for random vs slow at sigma_log = 0.6 (moderate spread)

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


def make_gamma_lognormal(N, sigma_log, rng):
    """Lognormal with E[gamma] = GAMMA_MEAN. mu = log(mean) - sigma^2/2."""
    if sigma_log <= 0:
        return np.full(N, GAMMA_MEAN, dtype=float)
    mu_log = np.log(GAMMA_MEAN) - 0.5 * sigma_log**2
    g = rng.lognormal(mean=mu_log, sigma=sigma_log, size=N)
    # renormalise arithmetic mean exactly so cross-sigma comparisons hold
    g *= GAMMA_MEAN / g.mean()
    return g


def run_sis(beta, gamma_arr, immune, f0=0.05, seed=None):
    rng = np.random.RandomState(seed)
    p = BASE.copy()
    N = p['N']; dt = p['dt']; n_iter = p['n_iter']
    r0, eps, rf = p['r0'], p['eps'], p['rf']
    v0, mu = p['v0'], p['mu']
    frame_every = max(1, n_iter // 200)
    p_recover_per_step = 1. - np.exp(-gamma_arr * dt)
    n0 = max(1, round(f0 * N))
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


def pick_immune(strategy, gamma_arr, p_immune, seed):
    rng = np.random.RandomState(1000 + seed)
    N = gamma_arr.size
    n_imm = int(round(p_immune * N))
    immune = np.zeros(N, dtype=bool)
    if n_imm == 0: return immune
    if strategy == 'random':
        immune[rng.choice(N, size=n_imm, replace=False)] = True
    elif strategy == 'slow':
        order = np.argsort(gamma_arr + rng.uniform(0., 1e-9, N))
        immune[order[:n_imm]] = True
    return immune


# =============================================================================
# EXP 1: width sweep -- slow vs random advantage as sigma_log grows
# =============================================================================
print('F59: continuous lognormal gamma vaccination')
print('  E[gamma]=1.0, beta=0.35 (above homogeneous threshold), %d seeds' % N_SEEDS)
BETA1 = 0.35
P_IMM = 0.20
sigma_vals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
res1 = {}
print('\nExp 1: advantage vs spread, p_imm=%.2f, beta=%.2f' % (P_IMM, BETA1))
for sigma in sigma_vals:
    f_random, f_slow = [], []
    for seed in range(N_SEEDS):
        rng_g = np.random.RandomState(seed)
        gamma_arr = make_gamma_lognormal(BASE['N'], sigma, rng_g)
        im_r = pick_immune('random', gamma_arr, P_IMM, seed)
        im_s = pick_immune('slow',   gamma_arr, P_IMM, seed)
        f_random.append(run_sis(BETA1, gamma_arr, im_r, seed=seed)[-30:].mean())
        f_slow.append(  run_sis(BETA1, gamma_arr, im_s, seed=seed)[-30:].mean())
    res1[sigma] = (np.mean(f_random), np.std(f_random),
                    np.mean(f_slow),   np.std(f_slow))
    print('  sigma=%.2f  random=%.3f+/-%.3f  slow=%.3f+/-%.3f  advantage=%.3f' % (
        sigma, res1[sigma][0], res1[sigma][1], res1[sigma][2], res1[sigma][3],
        res1[sigma][0] - res1[sigma][2]))

# =============================================================================
# EXP 2: p_immune sweep at fixed sigma_log = 0.6
# =============================================================================
print('\nExp 2: p_immune sweep at sigma_log=0.6, beta=%.2f' % BETA1)
sigma_fix = 0.6
P_IMM_VALS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
res2 = {p: {'random': (None, None), 'slow': (None, None)} for p in P_IMM_VALS}
for p_imm in P_IMM_VALS:
    f_random, f_slow = [], []
    for seed in range(N_SEEDS):
        rng_g = np.random.RandomState(seed)
        gamma_arr = make_gamma_lognormal(BASE['N'], sigma_fix, rng_g)
        im_r = pick_immune('random', gamma_arr, p_imm, seed)
        im_s = pick_immune('slow',   gamma_arr, p_imm, seed)
        f_random.append(run_sis(BETA1, gamma_arr, im_r, seed=seed)[-30:].mean())
        f_slow.append(  run_sis(BETA1, gamma_arr, im_s, seed=seed)[-30:].mean())
    res2[p_imm]['random'] = (np.mean(f_random), np.std(f_random))
    res2[p_imm]['slow']   = (np.mean(f_slow),   np.std(f_slow))
    print('  p_imm=%.2f  random=%.3f+/-%.3f  slow=%.3f+/-%.3f' % (
        p_imm, res2[p_imm]['random'][0], res2[p_imm]['random'][1],
        res2[p_imm]['slow'][0], res2[p_imm]['slow'][1]))

# =============================================================================
# FIGURE
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Continuous (lognormal) gamma -- vaccination (N=%d, beta=%.2f, %d seeds)'
             % (BASE['N'], BETA1, N_SEEDS), fontsize=11)

ax = axes[0]
ax.errorbar(sigma_vals, [res1[s][0] for s in sigma_vals],
            yerr=[res1[s][1] for s in sigma_vals],
            fmt='o-', color='gray', lw=2, capsize=3, label='random')
ax.errorbar(sigma_vals, [res1[s][2] for s in sigma_vals],
            yerr=[res1[s][3] for s in sigma_vals],
            fmt='o-', color='crimson', lw=2, capsize=3, label='slow-targeted')
ax.set_xlabel('lognormal width sigma_log'); ax.set_ylabel('f_ss')
ax.set_title('Strategy by heterogeneity (p_imm=%.2f)' % P_IMM)
ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3); ax.legend()

ax = axes[1]
ax.errorbar(P_IMM_VALS, [res2[p]['random'][0] for p in P_IMM_VALS],
            yerr=[res2[p]['random'][1] for p in P_IMM_VALS],
            fmt='o-', color='gray', lw=2, capsize=3, label='random')
ax.errorbar(P_IMM_VALS, [res2[p]['slow'][0] for p in P_IMM_VALS],
            yerr=[res2[p]['slow'][1] for p in P_IMM_VALS],
            fmt='o-', color='crimson', lw=2, capsize=3, label='slow-targeted')
ax.set_xlabel('p_immune'); ax.set_ylabel('f_ss')
ax.set_title('Sweep at sigma_log=%.1f' % sigma_fix)
ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3); ax.legend()

plt.tight_layout()
plt.savefig('figures/continuous_gamma_vaccination_1.png', dpi=120)
plt.close()
print('\n  --> figures/continuous_gamma_vaccination_1.png')
print('\nF59 complete.')
