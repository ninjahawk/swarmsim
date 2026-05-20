# slow_recoverer_vaccination.py -- vaccination strategies in a heterogeneous-
# recovery flock.
#
# F54 established that with bimodal per-agent gamma (slow recoverers vs fast),
# the SIS endemic state localises on the slow agents (panic ratio slow/fast
# 1.45-1.97).  Slow recoverers act as reservoirs.  F36 and F48 ruled out
# DEGREE-based targeting because the contact graph has no fat tail; F37 ruled
# out SPATIAL targeting because kinematic mixing scrambles immune positions.
# But F54 adds a THIRD candidate target class: internal-state hubs (slow
# recoverers).  Their "hub-ness" travels with the agent across kinematic
# mixing, so the F36/F37/F48 erosion mechanisms do not apply.
#
# Prediction (from F54): targeting slow recoverers should beat random
# vaccination, because removing reservoirs cuts the effective harmonic-mean
# 1/gamma that sets the threshold.
#
# Strategies compared (matched p_immune):
#   none      -- baseline
#   random    -- p_immune fraction chosen uniformly
#   slow      -- protect the agents with the smallest gamma_i first
#   fast      -- protect the agents with the largest gamma_i first (control)
#   degree    -- protect highest mean-contact-degree agents (re-check vs F36)
#
# Immune agents never become panicked (calm absorbing for that subset).
# Heterogeneity: bimodal {0.2, 1.8} (F54 "strong"), mean gamma = 1.0.
# beta is set to ~F54's "strong" beta_c.
#
# Outputs:
#   - f_ss vs p_immune for each strategy
#   - relative improvement vs random
#   - sanity check: panic fraction among slow/fast at one operating point

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
SPREAD      = 0.8   # bimodal {0.2, 1.8} -- F54 "strong"
BETA        = 0.30  # near F54 strong-condition threshold (~0.318)
P_IMMUNE_VALS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def make_gamma(N, spread, rng):
    gamma = np.full(N, GAMMA_MEAN, dtype=float)
    is_slow = np.zeros(N, dtype=bool)
    if spread > 0:
        idx = rng.choice(N, size=N // 2, replace=False)
        is_slow[idx] = True
        gamma[is_slow]  = GAMMA_MEAN - spread
        gamma[~is_slow] = GAMMA_MEAN + spread
    return gamma, is_slow


def estimate_degree(seed):
    """Quick warm-up sim to estimate per-agent mean contact-graph degree
    (the standard 'who has the most neighbours over time' proxy used by F36)."""
    rng = np.random.RandomState(seed)
    p = BASE.copy()
    N = p['N']; dt = p['dt']
    r0, eps, rf = p['r0'], p['eps'], p['rf']
    v0, mu = p['v0'], p['mu']
    x  = np.zeros(2*N)
    x[:N] = rng.uniform(0., 1., N); x[N:] = rng.uniform(0., 1., N)
    vx = rng.uniform(-1., 1., N) * v0
    vy = rng.uniform(-1., 1., N) * v0
    rb = max(r0, rf, R_CONT)
    deg_sum = np.zeros(N)
    n_steps = 800
    n_count = 0
    for i in range(n_steps):
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
        d_safe = np.where(rep_mask, np.sqrt(d2), 1.)
        base = np.where(rep_mask, 1. - d_safe/(2.*r0), 0.)
        strength = np.where(rep_mask, eps * base**1.5 / d_safe, 0.)
        repx = (-strength * dx).sum(axis=1); repy = (-strength * dy).sum(axis=1)
        vnorm = np.sqrt(vx**2 + vy**2)
        vnorms = np.where(vnorm == 0, 1., vnorm)
        fpropx = mu * (v0 - vnorm) * vx / vnorms
        fpropy = mu * (v0 - vnorm) * vy / vnorms
        frandx = p['ramp'] * rng.uniform(-1., 1., N)
        frandy = p['ramp'] * rng.uniform(-1., 1., N)
        fx = flockx + repx + fpropx + frandx
        fy = flocky + repy + fpropy + frandy
        vx += fx * dt; vy += fy * dt
        x[:N] = (x[:N] + vx*dt) % 1.; x[N:] = (x[N:] + vy*dt) % 1.
        if i >= 200:
            # measure contact degree at R_CONT
            rdx = x[:N][np.newaxis, :] - x[:N][:, np.newaxis]
            rdy = x[N:][np.newaxis, :] - x[N:][:, np.newaxis]
            rdx -= np.round(rdx); rdy -= np.round(rdy)
            rd2 = rdx**2 + rdy**2
            within = (rd2 <= R_CONT**2) & (rd2 > 0)
            deg_sum += within.sum(axis=1)
            n_count += 1
    return deg_sum / max(1, n_count)


def pick_immune(strategy, gamma_arr, is_slow, p_immune, seed):
    """Return boolean immune-mask of size N."""
    rng = np.random.RandomState(1000 + seed)
    N = gamma_arr.size
    n_imm = int(round(p_immune * N))
    immune = np.zeros(N, dtype=bool)
    if n_imm == 0:
        return immune
    if strategy == 'random':
        idx = rng.choice(N, size=n_imm, replace=False)
    elif strategy == 'slow':
        # ascending gamma -- but with bimodal, ties: break by rng so order
        # within the slow group is uniform
        jitter = rng.uniform(0., 1e-9, N)
        order = np.argsort(gamma_arr + jitter)
        idx = order[:n_imm]
    elif strategy == 'fast':
        jitter = rng.uniform(0., 1e-9, N)
        order = np.argsort(-(gamma_arr + jitter))
        idx = order[:n_imm]
    elif strategy == 'degree':
        deg = estimate_degree(seed)
        jitter = rng.uniform(0., 1e-9, N)
        order = np.argsort(-(deg + jitter))
        idx = order[:n_imm]
    else:
        raise ValueError(strategy)
    immune[idx] = True
    return immune


def run_sis_vacc(strategy, p_immune, beta=BETA, spread=SPREAD,
                  f0=0.05, n_frames=200, seed=None):
    rng = np.random.RandomState(seed)
    p = BASE.copy()
    N = p['N']; dt = p['dt']; n_iter = p['n_iter']
    r0, eps, rf = p['r0'], p['eps'], p['rf']
    v0, mu = p['v0'], p['mu']
    frame_every = max(1, n_iter // n_frames)

    gamma_arr, is_slow = make_gamma(N, spread, rng)
    p_recover_per_step = 1. - np.exp(-gamma_arr * dt)

    immune = pick_immune(strategy, gamma_arr, is_slow, p_immune, seed)

    n0 = max(1, round(f0 * N)) if f0 > 0 else 0
    is_panicked = np.zeros(N, dtype=bool)
    if n0 > 0:
        nonimm = np.where(~immune)[0]
        if nonimm.size < n0: n0 = nonimm.size
        is_panicked[rng.choice(nonimm, size=n0, replace=False)] = True

    x  = np.zeros(2*N)
    x[:N] = rng.uniform(0., 1., N); x[N:] = rng.uniform(0., 1., N)
    vx = rng.uniform(-1., 1., N) * v0
    vy = rng.uniform(-1., 1., N) * v0

    rb = max(r0, rf, R_CONT)
    frames = []
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
            if calm_idx.size and beta > 0:
                p_trans = 1. - np.exp(-beta * k[calm_idx] * dt)
                r = rng.uniform(0., 1., calm_idx.size)
                flipped = calm_idx[r < p_trans]
                if flipped.size:
                    is_panicked[flipped] = True

        if is_panicked.any():
            panic_idx = np.where(is_panicked)[0]
            r = rng.uniform(0., 1., panic_idx.size)
            recovered = panic_idx[r < p_recover_per_step[panic_idx]]
            if recovered.size:
                is_panicked[recovered] = False

        if i % frame_every == 0:
            frames.append((is_panicked.copy(), is_slow.copy(), immune.copy(),
                           vx.copy(), vy.copy()))
    return frames


def summarize(frames, ss_tail=30):
    last = frames[-ss_tail:]
    f_ss = np.mean([m.mean() for m, _, _, _, _ in last])
    phi_ss = np.mean([order_parameter(vx, vy) for _, _, _, vx, vy in last])
    slow_f, fast_f = [], []
    for m, slow, imm, _, _ in last:
        avail_slow = slow & ~imm
        avail_fast = (~slow) & ~imm
        if avail_slow.any():
            slow_f.append(m[avail_slow].mean())
        if avail_fast.any():
            fast_f.append(m[avail_fast].mean())
    slow_f = np.mean(slow_f) if slow_f else float('nan')
    fast_f = np.mean(fast_f) if fast_f else float('nan')
    return f_ss, phi_ss, slow_f, fast_f


STRATS = ['random', 'slow', 'fast', 'degree']

# =============================================================================
# EXP 1: f_ss vs p_immune by strategy
# =============================================================================
print('Exp 1: vaccination sweep with heterogeneous gamma (bimodal {0.2, 1.8})')
print('  beta=%.2f, %d seeds' % (BETA, N_SEEDS))
results = {s: {} for s in STRATS}
for strat in STRATS:
    print('  -- %s --' % strat)
    for p_imm in P_IMMUNE_VALS:
        fs = []
        for seed in range(N_SEEDS):
            frames = run_sis_vacc(strat, p_imm, seed=seed)
            f_ss, _, _, _ = summarize(frames)
            fs.append(f_ss)
        results[strat][p_imm] = (np.mean(fs), np.std(fs))
        print('     p_imm=%.2f  f_ss=%.3f +/- %.3f' % (p_imm, np.mean(fs), np.std(fs)))

# =============================================================================
# EXP 2: localisation check at p_imm=0.20
# =============================================================================
print('\nExp 2: localisation among non-immune slow/fast at p_imm=0.20')
loc = {}
for strat in STRATS:
    sl, fa = [], []
    for seed in range(N_SEEDS):
        frames = run_sis_vacc(strat, 0.20, seed=seed)
        _, _, s_f, f_f = summarize(frames)
        sl.append(s_f); fa.append(f_f)
    loc[strat] = (np.nanmean(sl), np.nanmean(fa))
    print('  %-8s  slow_f=%.3f  fast_f=%.3f' % (strat, loc[strat][0], loc[strat][1]))

# =============================================================================
# FIGURES
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Vaccination strategies in heterogeneous-recovery flock '
             '(beta=%.2f, %d seeds)' % (BETA, N_SEEDS), fontsize=11)

ax = axes[0]
colors = dict(random='gray', slow='crimson', fast='steelblue', degree='goldenrod')
for strat in STRATS:
    fv  = [results[strat][p][0] for p in P_IMMUNE_VALS]
    fvs = [results[strat][p][1] for p in P_IMMUNE_VALS]
    ax.errorbar(P_IMMUNE_VALS, fv, yerr=fvs, fmt='o-', color=colors[strat],
                lw=2, capsize=3, label=strat)
ax.set_xlabel('p_immune'); ax.set_ylabel('steady-state panic frac')
ax.set_title('SIS endemic level by vaccination strategy')
ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=9)

ax = axes[1]
xp = np.arange(len(STRATS))
sl_v = [loc[s][0] for s in STRATS]
fa_v = [loc[s][1] for s in STRATS]
ax.bar(xp - 0.2, sl_v, 0.4, color='crimson', label='slow (non-immune) f')
ax.bar(xp + 0.2, fa_v, 0.4, color='steelblue', label='fast (non-immune) f')
ax.set_xticks(xp); ax.set_xticklabels(STRATS)
ax.set_ylabel('panic fraction'); ax.set_title('Localisation at p_imm=0.20')
ax.set_ylim(0, 1.0); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figures/slow_recoverer_vaccination_1.png', dpi=120)
plt.close()
print('\n  --> figures/slow_recoverer_vaccination_1.png')
print('\nSlow-recoverer vaccination analysis complete.')
