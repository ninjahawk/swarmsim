# drifting_gamma_vaccination.py -- does slow-recoverer vaccination survive a
# TIME-EVOLVING recovery rate?
#
# F56-F61 established that targeting slow recoverers (smallest gamma_i) beats
# random vaccination, and the synthesis explains why: the "hub" label lives in
# per-agent gamma_i, a FIXED property of the individual that kinematic mixing
# cannot scramble (unlike degree, F36/F48, or spatial position, F37).  Every
# prior variant held gamma_i STATIONARY.  That stationarity is the single
# load-bearing assumption of the whole positive result, and it has never been
# tested.
#
# This experiment makes gamma_i a fluctuating STATE rather than a trait.  We
# vaccinate the slow agents ONCE at t=0 from a snapshot of gamma, then let each
# agent's slow/fast identity decorrelate by symmetric two-state resampling at
# rate r_drift (autocorrelation time ~ 1/r_drift).  The bimodal marginal and the
# ~50/50 split are preserved at all times; only WHICH individuals are slow drifts.
#
# Prediction (from the per-agent-invariance argument): the one-shot slow-target
# advantage should DECAY toward random as r_drift rises, because agents that were
# "fast" at t=0 (and therefore not vaccinated) drift into the slow class and
# replenish the reservoir behind the fixed immune set.  The boundary should sit
# where the drift timescale 1/r_drift drops below the reservoir-memory timescale
# (~1/gamma_slow).  If the advantage instead PERSISTS at high drift, the
# mechanism is subtler than "fixed label" and the synthesis needs revising.
#
# Strategies (matched p_immune, chosen once at t=0):
#   none    -- baseline, no vaccination (drift-independent reference)
#   random  -- p_immune fraction uniform
#   slow    -- protect the agents with smallest gamma_i at t=0
#
# Outputs:
#   - f_ss vs r_drift for slow vs random at p_imm=0.20 and 0.40
#   - advantage (random - slow) vs r_drift
#   - instantaneous localisation on the (drifting) slow set under slow-targeting

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
SPREAD      = 0.8            # bimodal {0.2, 1.8} -- F54/F56 "strong"
BETA        = 0.30          # F56 operating point
R_DRIFT_VALS = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]   # identity resample rate (per time unit)
P_IMM_VALS   = [0.20, 0.40]


def make_gamma(N, spread, rng):
    gamma = np.full(N, GAMMA_MEAN, dtype=float)
    is_slow = np.zeros(N, dtype=bool)
    if spread > 0:
        idx = rng.choice(N, size=N // 2, replace=False)
        is_slow[idx] = True
        gamma[is_slow]  = GAMMA_MEAN - spread
        gamma[~is_slow] = GAMMA_MEAN + spread
    return gamma, is_slow


def pick_immune(strategy, gamma_arr, p_immune, seed):
    """Immune mask of size N, chosen ONCE from the t=0 gamma snapshot."""
    rng = np.random.RandomState(1000 + seed)
    N = gamma_arr.size
    n_imm = int(round(p_immune * N))
    immune = np.zeros(N, dtype=bool)
    if n_imm == 0 or strategy == 'none':
        return immune
    if strategy == 'random':
        idx = rng.choice(N, size=n_imm, replace=False)
    elif strategy == 'slow':
        jitter = rng.uniform(0., 1e-9, N)
        order = np.argsort(gamma_arr + jitter)
        idx = order[:n_imm]
    else:
        raise ValueError(strategy)
    immune[idx] = True
    return immune


def run_sis_drift(strategy, p_immune, r_drift, beta=BETA, spread=SPREAD,
                  f0=0.05, n_frames=200, seed=None):
    rng = np.random.RandomState(seed)
    p = BASE.copy()
    N = p['N']; dt = p['dt']; n_iter = p['n_iter']
    r0, eps, rf = p['r0'], p['eps'], p['rf']
    v0, mu = p['v0'], p['mu']
    frame_every = max(1, n_iter // n_frames)

    gamma_arr, is_slow = make_gamma(N, spread, rng)
    p_recover_per_step = 1. - np.exp(-gamma_arr * dt)

    # vaccination decided ONCE from the t=0 snapshot
    immune = pick_immune(strategy, gamma_arr, p_immune, seed)

    # symmetric two-state resample: each agent, at rate r_drift, redraws its
    # slow/fast label uniformly. Preserves the bimodal marginal and ~50/50 split.
    p_flip = 1. - np.exp(-r_drift * dt) if r_drift > 0 else 0.0

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

        # transmission (immune agents never catch it)
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

        # recovery (uses CURRENT, possibly drifted, gamma)
        if is_panicked.any():
            panic_idx = np.where(is_panicked)[0]
            r = rng.uniform(0., 1., panic_idx.size)
            recovered = panic_idx[r < p_recover_per_step[panic_idx]]
            if recovered.size:
                is_panicked[recovered] = False

        # gamma identity drift (after the dynamics for this step)
        if p_flip > 0:
            flip = rng.uniform(0., 1., N) < p_flip
            if flip.any():
                new_slow = rng.uniform(0., 1., N) < 0.5
                is_slow[flip] = new_slow[flip]
                gamma_arr[flip] = np.where(is_slow[flip],
                                           GAMMA_MEAN - spread, GAMMA_MEAN + spread)
                p_recover_per_step[flip] = 1. - np.exp(-gamma_arr[flip] * dt)

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


# =============================================================================
# baseline: no vaccination (drift-independent reference)
# =============================================================================
print('Drifting-gamma vaccination -- testing the stationarity assumption of F56')
print('  bimodal {%.1f, %.1f}, beta=%.2f, %d seeds' %
      (GAMMA_MEAN - SPREAD, GAMMA_MEAN + SPREAD, BETA, N_SEEDS))
print('\nBaseline (no vaccination):')
base_fs = []
for seed in range(N_SEEDS):
    frames = run_sis_drift('none', 0.0, 0.0, seed=seed)
    f_ss, _, _, _ = summarize(frames)
    base_fs.append(f_ss)
print('  f_ss = %.3f +/- %.3f' % (np.mean(base_fs), np.std(base_fs)))

# =============================================================================
# EXP: f_ss vs r_drift for slow vs random, at each p_immune
# =============================================================================
results = {}   # results[(p_imm, strat)][r_drift] = (mean, std)
loc = {}        # loc[(p_imm, r_drift)] = (slow_f, fast_f) under slow-targeting
for p_imm in P_IMM_VALS:
    print('\n=== p_immune = %.2f ===' % p_imm)
    for strat in ('random', 'slow'):
        results[(p_imm, strat)] = {}
        print('  -- %s --' % strat)
        for rd in R_DRIFT_VALS:
            fs = []
            sl_loc, fa_loc = [], []
            for seed in range(N_SEEDS):
                frames = run_sis_drift(strat, p_imm, rd, seed=seed)
                f_ss, _, s_f, f_f = summarize(frames)
                fs.append(f_ss)
                sl_loc.append(s_f); fa_loc.append(f_f)
            results[(p_imm, strat)][rd] = (np.mean(fs), np.std(fs))
            if strat == 'slow':
                loc[(p_imm, rd)] = (np.nanmean(sl_loc), np.nanmean(fa_loc))
            print('     r_drift=%5.2f  f_ss=%.3f +/- %.3f' %
                  (rd, np.mean(fs), np.std(fs)))

# =============================================================================
# summary table: advantage (random - slow)
# =============================================================================
print('\nAdvantage of slow over random (random_f_ss - slow_f_ss):')
for p_imm in P_IMM_VALS:
    print('  p_immune=%.2f:' % p_imm)
    for rd in R_DRIFT_VALS:
        radv = results[(p_imm, 'random')][rd][0] - results[(p_imm, 'slow')][rd][0]
        print('     r_drift=%5.2f  random=%.3f  slow=%.3f  advantage=%+.3f' %
              (rd, results[(p_imm, 'random')][rd][0],
               results[(p_imm, 'slow')][rd][0], radv))

# =============================================================================
# FIGURES
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle('Slow-recoverer vaccination under a drifting recovery rate '
             '(bimodal {%.1f,%.1f}, beta=%.2f, %d seeds)'
             % (GAMMA_MEAN - SPREAD, GAMMA_MEAN + SPREAD, BETA, N_SEEDS),
             fontsize=11)

xp = np.arange(len(R_DRIFT_VALS))
xlabels = [('%.1f' % rd) for rd in R_DRIFT_VALS]
base_mean = np.mean(base_fs)

# panel 1: f_ss vs drift, p_imm=0.20
ax = axes[0]
for strat, col in (('random', 'gray'), ('slow', 'crimson')):
    fv  = [results[(0.20, strat)][rd][0] for rd in R_DRIFT_VALS]
    fvs = [results[(0.20, strat)][rd][1] for rd in R_DRIFT_VALS]
    ax.errorbar(xp, fv, yerr=fvs, fmt='o-', color=col, lw=2, capsize=3, label=strat)
ax.axhline(base_mean, ls='--', color='black', alpha=0.5, label='no vaccination')
ax.set_xticks(xp); ax.set_xticklabels(xlabels)
ax.set_xlabel('gamma identity drift rate (per tu)')
ax.set_ylabel('steady-state panic frac')
ax.set_title('p_immune = 0.20')
ax.set_ylim(-0.05, max(0.4, base_mean + 0.1)); ax.grid(alpha=0.3); ax.legend(fontsize=9)

# panel 2: f_ss vs drift, p_imm=0.40
ax = axes[1]
for strat, col in (('random', 'gray'), ('slow', 'crimson')):
    fv  = [results[(0.40, strat)][rd][0] for rd in R_DRIFT_VALS]
    fvs = [results[(0.40, strat)][rd][1] for rd in R_DRIFT_VALS]
    ax.errorbar(xp, fv, yerr=fvs, fmt='o-', color=col, lw=2, capsize=3, label=strat)
ax.axhline(base_mean, ls='--', color='black', alpha=0.5, label='no vaccination')
ax.set_xticks(xp); ax.set_xticklabels(xlabels)
ax.set_xlabel('gamma identity drift rate (per tu)')
ax.set_ylabel('steady-state panic frac')
ax.set_title('p_immune = 0.40')
ax.set_ylim(-0.05, max(0.4, base_mean + 0.1)); ax.grid(alpha=0.3); ax.legend(fontsize=9)

# panel 3: advantage vs drift
ax = axes[2]
for p_imm, col in ((0.20, 'darkgreen'), (0.40, 'purple')):
    adv = [results[(p_imm, 'random')][rd][0] - results[(p_imm, 'slow')][rd][0]
           for rd in R_DRIFT_VALS]
    ax.plot(xp, adv, 'o-', color=col, lw=2, label='p_imm=%.2f' % p_imm)
ax.axhline(0, ls='--', color='black', alpha=0.5)
ax.set_xticks(xp); ax.set_xticklabels(xlabels)
ax.set_xlabel('gamma identity drift rate (per tu)')
ax.set_ylabel('advantage (random - slow)')
ax.set_title('Slow-targeting advantage vs drift')
ax.grid(alpha=0.3); ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('figures/drifting_gamma_vaccination_1.png', dpi=120)
plt.close()
print('\n  --> figures/drifting_gamma_vaccination_1.png')
print('\nDrifting-gamma vaccination analysis complete.')
