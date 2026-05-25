# het_beta_gamma_vaccination.py -- does slow-recoverer targeting survive when a
# SECOND heterogeneity axis (infectiousness beta_i) is layered on, possibly
# adversarially correlated with recovery rate?
#
# F55: heterogeneous infectiousness (per-agent beta_i, source-side) does NOT
#      shift the SIS threshold -- super-spreaders source most transmission
#      EVENTS but produce no stock asymmetry, because recovery (gamma) was
#      homogeneous.  Conclusion: target gamma_i, not beta_i.
# F56-F61: targeting slow recoverers (smallest gamma_i) beats random; the hub
#      label lives in per-agent gamma_i.
#
# Here BOTH heterogeneities are present.  gamma_i is bimodal {0.2,1.8} (slow/
# fast, 50/50); beta_i is bimodal {0.15, 0.90} (normal/super-spreader, 80/20,
# arithmetic mean 0.30 == F56 operating point).  We vary the CORRELATION
# between the two labels:
#   independent -- super-spreaders assigned at random
#   pos         -- super-spreaders are preferentially SLOW recoverers
#                  (worst reservoirs; the dangerous agents ARE the targeted ones)
#   neg         -- super-spreaders are preferentially FAST recoverers
#                  (ADVERSARIAL: the biggest spreaders escape a gamma-based vaccine)
#
# Transmission is source-weighted (F55 convention): a calm agent's hazard is
# sum of beta_j over its panicked contacts j.  Recovery uses the agent's gamma_i.
#
# Strategies (matched p_immune):
#   random  -- uniform
#   slow    -- smallest gamma_i first   (F56 policy)
#   super   -- largest beta_i first     (infectiousness policy; F55 predicts weak)
#   combo   -- half budget slow + half budget super
#
# Prediction (from F54/F55/F56): the reservoir is set by gamma_i, so slow-
# targeting should remain the best single strategy in EVERY correlation regime.
# The decisive case is 'neg': if anti-correlated super-spreaders leak enough to
# let beta- or combo-targeting overtake slow, then "target gamma_i not beta_i"
# breaks under combined heterogeneity; if slow still wins, F55/F56 compose cleanly.

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
SPREAD      = 0.8            # gamma bimodal {0.2, 1.8}
BETA_NORMAL = 0.15
BETA_SUPER  = 0.90          # super source 6x normal
F_SUPER     = 0.20          # 20% super-spreaders -> mean beta = 0.30
CORRELATIONS = ['independent', 'pos', 'neg']
P_IMM_SWEEP  = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
P_IMM_FIXED  = 0.30


def make_labels(N, corr, rng):
    """Return gamma_arr, beta_arr, is_slow, is_super for a correlation regime."""
    gamma = np.full(N, GAMMA_MEAN, dtype=float)
    is_slow = np.zeros(N, dtype=bool)
    slow_idx = rng.choice(N, size=N // 2, replace=False)
    is_slow[slow_idx] = True
    gamma[is_slow]  = GAMMA_MEAN - SPREAD
    gamma[~is_slow] = GAMMA_MEAN + SPREAD

    n_super = int(round(F_SUPER * N))
    is_super = np.zeros(N, dtype=bool)
    slow_members = np.where(is_slow)[0]
    fast_members = np.where(~is_slow)[0]
    if corr == 'independent':
        sup = rng.choice(N, size=n_super, replace=False)
    elif corr == 'pos':       # super-spreaders preferentially slow
        pool = np.concatenate([rng.permutation(slow_members),
                               rng.permutation(fast_members)])
        sup = pool[:n_super]
    elif corr == 'neg':       # super-spreaders preferentially fast
        pool = np.concatenate([rng.permutation(fast_members),
                               rng.permutation(slow_members)])
        sup = pool[:n_super]
    else:
        raise ValueError(corr)
    is_super[sup] = True
    beta = np.where(is_super, BETA_SUPER, BETA_NORMAL)
    return gamma, beta, is_slow, is_super


def pick_immune(strategy, gamma_arr, beta_arr, p_immune, seed):
    rng = np.random.RandomState(1000 + seed)
    N = gamma_arr.size
    n_imm = int(round(p_immune * N))
    immune = np.zeros(N, dtype=bool)
    if n_imm == 0 or strategy == 'none':
        return immune
    jit = rng.uniform(0., 1e-9, N)
    if strategy == 'random':
        idx = rng.choice(N, size=n_imm, replace=False)
    elif strategy == 'slow':
        idx = np.argsort(gamma_arr + jit)[:n_imm]
    elif strategy == 'super':
        idx = np.argsort(-(beta_arr + jit))[:n_imm]
    elif strategy == 'combo':
        n_slow = n_imm // 2
        n_sup  = n_imm - n_slow
        slow_order = np.argsort(gamma_arr + jit)
        sup_order  = np.argsort(-(beta_arr + jit))
        chosen = []
        si = ti = 0
        chosen_set = set()
        # interleave: take slowest and most-infectious until budget filled
        while len(chosen) < n_imm:
            if len(chosen) < n_slow + (n_imm - n_slow - n_sup) + 1 and si < N:
                c = slow_order[si]; si += 1
                if c not in chosen_set:
                    chosen.append(c); chosen_set.add(c)
            if len(chosen) < n_imm and ti < N:
                c = sup_order[ti]; ti += 1
                if c not in chosen_set:
                    chosen.append(c); chosen_set.add(c)
        idx = np.array(chosen[:n_imm])
    else:
        raise ValueError(strategy)
    immune[idx] = True
    return immune


def run(strategy, p_immune, corr, f0=0.05, n_frames=200, seed=None):
    rng = np.random.RandomState(seed)
    p = BASE.copy()
    N = p['N']; dt = p['dt']; n_iter = p['n_iter']
    r0, eps, rf = p['r0'], p['eps'], p['rf']
    v0, mu = p['v0'], p['mu']
    frame_every = max(1, n_iter // n_frames)

    gamma_arr, beta_arr, is_slow, is_super = make_labels(N, corr, rng)
    p_recover_per_step = 1. - np.exp(-gamma_arr * dt)
    immune = pick_immune(strategy, gamma_arr, beta_arr, p_immune, seed)

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
        baserep = np.where(rep_mask, 1. - d_safe/(2.*r0), 0.)
        strength = np.where(rep_mask, eps * baserep**1.5 / d_safe, 0.)
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

        # source-weighted transmission: hazard_i = sum_j beta_j over panicked contacts j
        if is_panicked.any() and (~is_panicked & ~immune).any():
            rdx = x[:N][np.newaxis, :] - x[:N][:, np.newaxis]
            rdy = x[N:][np.newaxis, :] - x[N:][:, np.newaxis]
            rdx -= np.round(rdx); rdy -= np.round(rdy)
            rd2 = rdx**2 + rdy**2
            within = (rd2 <= R_CONT**2) & (rd2 > 0)
            hazard = within @ (is_panicked.astype(float) * beta_arr)
            calm_idx = np.where(~is_panicked & ~immune)[0]
            if calm_idx.size:
                p_trans = 1. - np.exp(-hazard[calm_idx] * dt)
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
            frames.append((is_panicked.copy(), vx.copy(), vy.copy()))
    return frames


def f_ss(frames, ss_tail=30):
    last = frames[-ss_tail:]
    return np.mean([m.mean() for m, _, _ in last])


STRATS = ['random', 'slow', 'super', 'combo']
COLORS = dict(random='gray', slow='crimson', super='steelblue', combo='goldenrod')

print('Combined beta_i + gamma_i heterogeneity with vaccination targeting')
print('  gamma {%.1f,%.1f} 50/50 ; beta {%.2f,%.2f} %.0f%% super (mean %.2f) ; %d seeds'
      % (GAMMA_MEAN - SPREAD, GAMMA_MEAN + SPREAD, BETA_NORMAL, BETA_SUPER,
         100*F_SUPER, BETA_NORMAL*(1-F_SUPER) + BETA_SUPER*F_SUPER, N_SEEDS))

# Exp 1: p_imm sweep under independent correlation
print('\nExp 1: p_immune sweep (independent correlation)')
sweep = {s: {} for s in STRATS}
for strat in STRATS:
    print('  -- %s --' % strat)
    for p_imm in P_IMM_SWEEP:
        vals = [f_ss(run(strat, p_imm, 'independent', seed=s)) for s in range(N_SEEDS)]
        sweep[strat][p_imm] = (np.mean(vals), np.std(vals))
        print('     p_imm=%.2f  f_ss=%.3f +/- %.3f' % (p_imm, np.mean(vals), np.std(vals)))

# Exp 2: fixed p_imm, all strategies x all correlations
print('\nExp 2: fixed p_immune=%.2f, strategy x correlation' % P_IMM_FIXED)
grid = {}
for corr in CORRELATIONS:
    print('  == correlation: %s ==' % corr)
    for strat in STRATS:
        vals = [f_ss(run(strat, P_IMM_FIXED, corr, seed=s)) for s in range(N_SEEDS)]
        grid[(corr, strat)] = (np.mean(vals), np.std(vals))
        print('     %-7s  f_ss=%.3f +/- %.3f' % (strat, np.mean(vals), np.std(vals)))
    # baseline (no vaccination) for this correlation
    vals = [f_ss(run('none', 0.0, corr, seed=s)) for s in range(N_SEEDS)]
    grid[(corr, 'none')] = (np.mean(vals), np.std(vals))
    print('     %-7s  f_ss=%.3f +/- %.3f' % ('none', np.mean(vals), np.std(vals)))

# =============================================================================
# FIGURES
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Combined beta_i + gamma_i heterogeneity: vaccination targeting (%d seeds)'
             % N_SEEDS, fontsize=11)

ax = axes[0]
for strat in STRATS:
    fv  = [sweep[strat][p][0] for p in P_IMM_SWEEP]
    fvs = [sweep[strat][p][1] for p in P_IMM_SWEEP]
    ax.errorbar(P_IMM_SWEEP, fv, yerr=fvs, fmt='o-', color=COLORS[strat],
                lw=2, capsize=3, label=strat)
ax.set_xlabel('p_immune'); ax.set_ylabel('steady-state panic frac')
ax.set_title('p_immune sweep (independent corr)')
ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=9)

ax = axes[1]
xc = np.arange(len(CORRELATIONS))
w = 0.2
for j, strat in enumerate(STRATS):
    vals = [grid[(c, strat)][0] for c in CORRELATIONS]
    errs = [grid[(c, strat)][1] for c in CORRELATIONS]
    ax.bar(xc + (j - 1.5)*w, vals, w, yerr=errs, capsize=2,
           color=COLORS[strat], label=strat)
ax.set_xticks(xc); ax.set_xticklabels(CORRELATIONS)
ax.set_xlabel('beta-gamma correlation')
ax.set_ylabel('steady-state panic frac')
ax.set_title('Strategy x correlation at p_immune=%.2f' % P_IMM_FIXED)
ax.grid(alpha=0.3, axis='y'); ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('figures/het_beta_gamma_vaccination_1.png', dpi=120)
plt.close()
print('\n  --> figures/het_beta_gamma_vaccination_1.png')
print('\nCombined beta_i + gamma_i vaccination analysis complete.')
