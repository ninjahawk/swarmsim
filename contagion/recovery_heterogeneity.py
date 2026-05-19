# recovery_heterogeneity.py -- SIS contagion with heterogeneous recovery rates
#
# Finding 17 established a clean SIS epidemic threshold at beta/gamma ~ 1 for a
# flock in which every agent recovers at the SAME rate gamma.  Real populations
# are heterogeneous: some individuals shed panic quickly, others stay agitated
# far longer.  This script asks whether a SPREAD in per-agent recovery rate
# gamma_i -- holding the population MEAN gamma fixed -- changes the outbreak.
#
# Mean-field expectation.  In well-mixed SIS the endemic state is governed by
# the per-agent ratio beta*<k>/gamma_i.  Agents with small gamma_i sit panicked
# far longer and act as RESERVOIRS that keep reseeding their neighbours.  The
# outbreak threshold for a heterogeneous-gamma population is set not by the
# arithmetic mean of gamma but by something closer to its HARMONIC mean
# (slow recoverers dominate).  Since harmonic mean <= arithmetic mean, spreading
# gamma at fixed arithmetic mean should LOWER the effective threshold and RAISE
# the steady-state panic fraction.  Equivalently: heterogeneity helps the
# epidemic.  This experiment tests that and measures where the panic localises.
#
# Conditions (all share arithmetic-mean gamma = 1.0):
#   homog   : every agent gamma_i = 1.0
#   mild    : bimodal, gamma_i in {0.5, 1.5}  (50/50)
#   strong  : bimodal, gamma_i in {0.2, 1.8}  (50/50)
#   extreme : bimodal, gamma_i in {0.05, 1.95} (50/50)
#
# Experiments:
#   1. Beta sweep for each condition -- threshold shift + f_ss elevation
#   2. Reservoir check: at fixed beta, panic fraction among slow vs fast agents
#   3. Heterogeneity sweep at fixed near-threshold beta -- f_ss vs gamma spread

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


def make_gamma(N, spread, rng):
    """Per-agent recovery rates: bimodal {mean-spread, mean+spread}, 50/50.
    spread=0 reproduces the homogeneous case.  Returns (gamma_arr, is_slow)."""
    gamma = np.full(N, GAMMA_MEAN, dtype=float)
    is_slow = np.zeros(N, dtype=bool)
    if spread > 0:
        slow_idx = rng.choice(N, size=N // 2, replace=False)
        is_slow[slow_idx] = True
        gamma[is_slow]  = GAMMA_MEAN - spread
        gamma[~is_slow] = GAMMA_MEAN + spread
    return gamma, is_slow


def run_sis_het(beta=2.0, spread=0.0, f0=0.05, n_frames=200, seed=None):
    """SIS contagion with per-agent recovery rate. Arithmetic-mean gamma fixed
    at GAMMA_MEAN; `spread` sets the bimodal half-width.  Returns frames where
    each frame carries (x, y, vx, vy, is_panicked, is_slow)."""
    rng = np.random.RandomState(seed)

    p = BASE.copy()
    N      = p['N']
    dt     = p['dt']
    n_iter = p['n_iter']
    r0, eps, rf = p['r0'], p['eps'], p['rf']
    v0, mu       = p['v0'], p['mu']
    frame_every  = max(1, n_iter // n_frames)

    gamma_arr, is_slow = make_gamma(N, spread, rng)
    p_recover_per_step = 1. - np.exp(-gamma_arr * dt)   # per-agent

    n0 = max(1, round(f0 * N)) if f0 > 0 else 0
    is_panicked = np.zeros(N, dtype=bool)
    if n0 > 0:
        seed_idx = rng.choice(N, size=n0, replace=False)
        is_panicked[seed_idx] = True

    x  = np.zeros(2*N)
    x[:N] = rng.uniform(0., 1., N)
    x[N:] = rng.uniform(0., 1., N)
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
        base     = np.where(rep_mask, 1. - d_safe/(2.*r0), 0.)
        strength = np.where(rep_mask, eps * base**1.5 / d_safe, 0.)
        repx = (-strength * dx).sum(axis=1)
        repy = (-strength * dy).sum(axis=1)

        vnorm  = np.sqrt(vx**2 + vy**2)
        vnorms = np.where(vnorm == 0, 1., vnorm)
        fpropx = mu * (v0 - vnorm) * vx / vnorms
        fpropy = mu * (v0 - vnorm) * vy / vnorms

        frandx = ramp_arr * rng.uniform(-1., 1., N)
        frandy = ramp_arr * rng.uniform(-1., 1., N)

        fx = flockx + repx + fpropx + frandx
        fy = flocky + repy + fpropy + frandy

        vx += fx * dt
        vy += fy * dt
        x[:N] = (x[:N] + vx*dt) % 1.
        x[N:] = (x[N:] + vy*dt) % 1.

        # contagion: calm -> panic at rate beta*k
        if is_panicked.any() and (~is_panicked).any():
            real_dx = x[:N][np.newaxis, :] - x[:N][:, np.newaxis]
            real_dy = x[N:][np.newaxis, :] - x[N:][:, np.newaxis]
            real_dx -= np.round(real_dx); real_dy -= np.round(real_dy)
            rd2 = real_dx**2 + real_dy**2
            within = (rd2 <= R_CONT**2) & (rd2 > 0)
            k = within @ is_panicked.astype(np.int32)
            calm_idx = np.where(~is_panicked)[0]
            if calm_idx.size and beta > 0:
                k_calm = k[calm_idx]
                p_trans = 1. - np.exp(-beta * k_calm * dt)
                r = rng.uniform(0., 1., calm_idx.size)
                flipped = calm_idx[r < p_trans]
                if flipped.size:
                    is_panicked[flipped] = True

        # recovery: panic -> calm at per-agent rate gamma_i
        if is_panicked.any():
            panic_idx = np.where(is_panicked)[0]
            r = rng.uniform(0., 1., panic_idx.size)
            recovered = panic_idx[r < p_recover_per_step[panic_idx]]
            if recovered.size:
                is_panicked[recovered] = False

        if i % frame_every == 0:
            frames.append((x[:N].copy(), x[N:].copy(),
                           vx.copy(), vy.copy(),
                           is_panicked.copy(), is_slow.copy()))

    return frames


def summarize(frames, ss_tail=30):
    last = frames[-ss_tail:]
    f_ss = np.mean([m.mean() for _, _, _, _, m, _ in last])
    phi_ss = np.mean([order_parameter(vx, vy) for _, _, vx, vy, _, _ in last])
    # panic fraction split by recovery class
    slow_f, fast_f = [], []
    for _, _, _, _, m, slow in last:
        if slow.any():
            slow_f.append(m[slow].mean())
            fast_f.append(m[~slow].mean())
    slow_f = np.mean(slow_f) if slow_f else float('nan')
    fast_f = np.mean(fast_f) if fast_f else float('nan')
    return f_ss, phi_ss, slow_f, fast_f


CONDITIONS = [
    ('homog',   0.00),
    ('mild',    0.50),
    ('strong',  0.80),
    ('extreme', 0.95),
]

# =============================================================================
# EXP 1: BETA SWEEP for each heterogeneity condition
# =============================================================================
print('Exp 1: beta sweep, mean gamma=1.0, %d seeds' % N_SEEDS)
print('  (bimodal {1-spread, 1+spread}; spread=0 is homogeneous)')
beta_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
res1 = {}
for name, spread in CONDITIONS:
    res1[name] = {}
    print('  -- %s (spread=%.2f) --' % (name, spread))
    for beta in beta_vals:
        fs = []
        for s in range(N_SEEDS):
            frames = run_sis_het(beta=beta, spread=spread, f0=0.05, seed=s)
            f_ss, _, _, _ = summarize(frames)
            fs.append(f_ss)
        res1[name][beta] = (np.mean(fs), np.std(fs))
        print('     beta=%4.1f  f_ss=%.3f +/- %.3f' % (beta, np.mean(fs), np.std(fs)))


def threshold_cross(betas, fvals, level=0.15):
    """Linear-interpolated beta at which f_ss first crosses `level`."""
    for j in range(1, len(betas)):
        if fvals[j-1] < level <= fvals[j]:
            t = (level - fvals[j-1]) / (fvals[j] - fvals[j-1])
            return betas[j-1] + t * (betas[j] - betas[j-1])
    return float('nan')

print('\n  Threshold beta (f_ss crosses 0.15):')
for name, _ in CONDITIONS:
    betas = beta_vals
    fvals = [res1[name][b][0] for b in betas]
    bc = threshold_cross(betas, fvals)
    print('     %-8s beta_c = %.3f' % (name, bc))

# =============================================================================
# EXP 2: RESERVOIR CHECK -- panic localisation by recovery class
# =============================================================================
print('\nExp 2: reservoir check -- panic fraction among slow vs fast agents')
print('  (fixed beta=1.0, near threshold)')
beta_res = 1.0
res2 = {}
for name, spread in CONDITIONS:
    if spread == 0:
        continue
    ss, fs_ = [], []
    for s in range(N_SEEDS):
        frames = run_sis_het(beta=beta_res, spread=spread, f0=0.05, seed=s)
        _, _, slow_f, fast_f = summarize(frames)
        ss.append(slow_f); fs_.append(fast_f)
    res2[name] = (np.mean(ss), np.mean(fs_))
    ratio = np.mean(ss) / max(np.mean(fs_), 1e-6)
    print('  %-8s slow-agent f=%.3f  fast-agent f=%.3f  ratio=%.2f' % (
        name, np.mean(ss), np.mean(fs_), ratio))

# =============================================================================
# EXP 3: HETEROGENEITY SWEEP at fixed near-threshold beta
# =============================================================================
print('\nExp 3: heterogeneity sweep at fixed beta=0.7 (sub-threshold for homog)')
beta_het = 0.7
spread_vals = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95]
res3 = {}
for spread in spread_vals:
    fs = []
    for s in range(N_SEEDS):
        frames = run_sis_het(beta=beta_het, spread=spread, f0=0.05, seed=s)
        f_ss, _, _, _ = summarize(frames)
        fs.append(f_ss)
    res3[spread] = (np.mean(fs), np.std(fs))
    print('  spread=%.2f  f_ss=%.3f +/- %.3f' % (spread, np.mean(fs), np.std(fs)))

# =============================================================================
# FIGURES
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('SIS with heterogeneous recovery (N=350, mean gamma=1.0, %d seeds)'
             % N_SEEDS, fontsize=11)

ax = axes[0]
colors = {'homog': 'black', 'mild': 'goldenrod',
          'strong': 'darkorange', 'extreme': 'crimson'}
for name, _ in CONDITIONS:
    fvals = [res1[name][b][0] for b in beta_vals]
    fstd  = [res1[name][b][1] for b in beta_vals]
    ax.errorbar(beta_vals, fvals, yerr=fstd, fmt='o-', lw=2, capsize=3,
                color=colors[name], label=name)
ax.axhline(0.15, ls=':', color='gray', lw=1)
ax.set_xlabel('Contagion rate beta'); ax.set_ylabel('steady-state panic frac')
ax.set_title('Beta sweep by heterogeneity')
ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1]
names = list(res2.keys())
slow_v = [res2[n][0] for n in names]
fast_v = [res2[n][1] for n in names]
xp = np.arange(len(names))
ax.bar(xp - 0.2, slow_v, 0.4, color='crimson', label='slow recoverers')
ax.bar(xp + 0.2, fast_v, 0.4, color='steelblue', label='fast recoverers')
ax.set_xticks(xp); ax.set_xticklabels(names)
ax.set_ylabel('panic fraction'); ax.set_title('Reservoir check (beta=1.0)')
ax.set_ylim(0, 1.0); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

ax = axes[2]
fv  = [res3[s][0] for s in spread_vals]
fvs = [res3[s][1] for s in spread_vals]
ax.errorbar(spread_vals, fv, yerr=fvs, fmt='o-', color='purple', lw=2, capsize=3)
ax.set_xlabel('Recovery-rate spread (bimodal half-width)')
ax.set_ylabel('steady-state panic frac')
ax.set_title('Heterogeneity sweep (beta=0.7, sub-threshold for homog)')
ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/recovery_heterogeneity_1.png', dpi=120)
plt.close()
print('\n  --> figures/recovery_heterogeneity_1.png')
print('\nHeterogeneous-recovery SIS analysis complete.')
