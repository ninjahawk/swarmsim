# noisy_gamma_vaccination.py -- robustness of slow-recoverer vaccination to
# noisy estimates of per-agent gamma_i (Finding 60)
#
# F56-F59 established slow-recoverer targeting as the only vaccination strategy
# in this study that beats random.  All of those experiments used the EXACT
# true gamma_i to rank agents.  In practice, the recovery rate of an individual
# cannot be measured exactly -- the best one could do is observe how long past
# panic episodes lasted and estimate gamma_i with noise.  If the slow-targeting
# advantage requires perfect knowledge, it is impractical.  If it tolerates
# substantial noise, it remains an actionable policy.
#
# Setup.  Per-agent true gamma_i drawn from the F56 bimodal {0.2, 1.8}.  The
# vaccinator does NOT see true gamma_i; they see an observation
#     gamma_hat_i = true_gamma_i + N(0, sigma_obs).
# Then they immunise the bottom p_imm fraction by gamma_hat_i.  Sweep sigma_obs
# to find the noise tolerance threshold beyond which slow-targeting degrades
# to random.
#
# Key tunable: sigma_obs (observation noise).  At sigma_obs ~ 0 the policy is
# exact F56.  At sigma_obs >> (gamma_fast - gamma_slow)/2 = 0.8 the observation
# is uninformative and the policy should collapse to random.

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
P_IMM       = 0.30


def make_gamma(N, rng):
    gamma = np.full(N, GAMMA_MEAN, dtype=float)
    is_slow = np.zeros(N, dtype=bool)
    idx = rng.choice(N, size=N // 2, replace=False)
    is_slow[idx] = True
    gamma[is_slow]  = GAMMA_MEAN - SPREAD
    gamma[~is_slow] = GAMMA_MEAN + SPREAD
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


print('F60: noisy gamma vaccination -- robustness of slow-targeting to obs noise')
print('  beta=%.2f, true gamma bimodal {%.2f, %.2f}, %d seeds' % (
    BETA, GAMMA_MEAN-SPREAD, GAMMA_MEAN+SPREAD, N_SEEDS))
print('  p_immune fixed at %.2f' % P_IMM)

sigma_obs_vals = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 100.0]
res = {}
for sigma_obs in sigma_obs_vals:
    fs = []
    correct_frac_list = []
    for seed in range(N_SEEDS):
        rng_g = np.random.RandomState(seed)
        gamma_arr, is_slow = make_gamma(BASE['N'], rng_g)
        # observation noise
        rng_obs = np.random.RandomState(2000 + seed)
        gamma_hat = gamma_arr + rng_obs.normal(0., sigma_obs, BASE['N'])
        rng_pick = np.random.RandomState(3000 + seed)
        n_imm = int(round(P_IMM * BASE['N']))
        order = np.argsort(gamma_hat + rng_pick.uniform(0., 1e-9, BASE['N']))
        immune = np.zeros(BASE['N'], dtype=bool)
        immune[order[:n_imm]] = True
        # fraction of immune that are truly slow
        true_slow_count = is_slow[immune].sum()
        n_slow_avail = is_slow.sum()
        correct_frac_list.append(true_slow_count / max(1, n_imm))
        hist = run_sis(BETA, gamma_arr, immune, seed)
        fs.append(hist[-30:].mean())
    res[sigma_obs] = (np.mean(fs), np.std(fs), np.mean(correct_frac_list))
    print('  sigma_obs=%5.2f  f_ss=%.3f +/- %.3f  slow_hit_rate=%.3f' % (
        sigma_obs, np.mean(fs), np.std(fs), np.mean(correct_frac_list)))

# random baseline at the same p_imm and beta for direct comparison
print('\nRandom baseline (same p_imm=%.2f, same gamma seeds):' % P_IMM)
random_fs = []
for seed in range(N_SEEDS):
    rng_g = np.random.RandomState(seed)
    gamma_arr, is_slow = make_gamma(BASE['N'], rng_g)
    rng_pick = np.random.RandomState(4000 + seed)
    n_imm = int(round(P_IMM * BASE['N']))
    immune = np.zeros(BASE['N'], dtype=bool)
    immune[rng_pick.choice(BASE['N'], size=n_imm, replace=False)] = True
    hist = run_sis(BETA, gamma_arr, immune, seed)
    random_fs.append(hist[-30:].mean())
f_random = np.mean(random_fs)
f_random_std = np.std(random_fs)
print('  random f_ss=%.3f +/- %.3f' % (f_random, f_random_std))

# FIGURE
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Slow-targeting robustness to noisy gamma observation '
             '(p_imm=%.2f, beta=%.2f)' % (P_IMM, BETA), fontsize=11)

ax = axes[0]
sig_plot = [s if s < 50 else 5.0 for s in sigma_obs_vals]
fs_plot  = [res[s][0] for s in sigma_obs_vals]
fserr    = [res[s][1] for s in sigma_obs_vals]
ax.errorbar(sig_plot, fs_plot, yerr=fserr, fmt='o-', color='crimson',
            lw=2, capsize=3, label='slow-targeted (noisy)')
ax.axhline(f_random, ls='--', color='gray', lw=2, label='random baseline')
ax.fill_between(sig_plot, f_random - f_random_std, f_random + f_random_std,
                color='gray', alpha=0.2)
ax.set_xlabel('observation noise sigma_obs'); ax.set_ylabel('f_ss')
ax.set_title('Endemic level vs observation noise')
ax.set_xscale('symlog', linthresh=0.1); ax.set_ylim(-0.05, 0.4)
ax.grid(alpha=0.3); ax.legend()

ax = axes[1]
hit_plot = [res[s][2] for s in sigma_obs_vals]
ax.plot(sig_plot, hit_plot, 'o-', color='purple', lw=2)
ax.axhline(0.5, ls=':', color='gray', label='chance (true slow share)')
ax.set_xlabel('observation noise sigma_obs')
ax.set_ylabel('fraction of immunised that are truly slow')
ax.set_title('Targeting accuracy')
ax.set_xscale('symlog', linthresh=0.1)
ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend()

plt.tight_layout()
plt.savefig('figures/noisy_gamma_vaccination_1.png', dpi=120)
plt.close()
print('\n  --> figures/noisy_gamma_vaccination_1.png')
print('\nF60 complete.')
