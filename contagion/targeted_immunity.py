# targeted_immunity.py -- Targeted vs random herd immunity in the flock
#
# Finding 30 measured the random-vaccination herd-immunity threshold at p_c ~ 0.46
# (2x the mean-field prediction) due to spatial clustering.  Classical epidemic theory
# on networks predicts that targeting high-degree (hub) nodes can dramatically lower
# the required immune fraction.
#
# This experiment compares two vaccination strategies at supercritical SIS
# (beta=2.5, gamma=2.0, R0=1.25 -- same as Finding 30):
#   "random"   -- p_immune agents chosen uniformly at random
#   "targeted" -- p_immune agents chosen as the top-degree neighbors within r_cont
#                 (degree measured once from the settled flock configuration before
#                  the first panicked agent is seeded)
#
# Sweep p_immune = 0, 0.10, 0.20, 0.30, 0.40, 0.46, 0.50, 0.60, 0.70
# Measure f_ss = mean panic fraction over last 20 time units of a 100 tu run.
#
# Expected: if the flock contact network has a fat-tailed degree distribution,
# targeted vaccination should outperform random -- lower f_ss at the same p_immune,
# or cross the quench threshold (f_ss < 0.1) at a lower p_immune.

import os
import numpy as np
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from flocking import buffer as buf_fn, order_parameter

os.makedirs('figures', exist_ok=True)

N_SEEDS   = 6
N_WARMUP  = 1500
N_ITER    = 10000   # 100 time units of SIS dynamics
RECORD_EVERY = 50

BASE = dict(
    N=350, r0=0.005, eps=0.1, rf=0.1,
    alpha=1.0, v0=0.02, mu=10.0, ramp=0.1, dt=0.01,
)
PANIC_ALPHA = 0.1
PANIC_RAMP  = 10.0
R_CONT      = 0.05
BETA        = 2.5
GAMMA       = 2.0

P_IMMUNE_LIST = [0.00, 0.10, 0.20, 0.30, 0.40, 0.46, 0.50, 0.60, 0.70]
F0_FRAC       = 0.05  # 5% seed panic fraction (same as Finding 30)


def warmup(seed, n_steps):
    """Return settled flock (x, vx, vy)."""
    np.random.seed(seed)
    p = BASE.copy()
    N = p['N']; dt = p['dt']
    r0, eps, rf = p['r0'], p['eps'], p['rf']
    v0, mu = p['v0'], p['mu']
    rb = max(r0, rf)

    x  = np.zeros(2*N)
    x[:N] = np.random.uniform(0., 1., N)
    x[N:] = np.random.uniform(0., 1., N)
    vx = np.random.uniform(-1., 1., N) * v0
    vy = np.random.uniform(-1., 1., N) * v0

    for _ in range(n_steps):
        nb, xb, yb, vxb, vyb = buf_fn(rb, x, vx, vy, N)
        dx = xb[:nb][np.newaxis, :] - x[:N][:, np.newaxis]
        dy = yb[:nb][np.newaxis, :] - x[N:][:, np.newaxis]
        d2 = dx**2 + dy**2

        not_self = np.ones((N, nb), dtype=bool)
        idx = np.arange(min(N, nb))
        not_self[idx, idx] = False

        flock_mask = (d2 <= rf**2) & not_self
        flx = np.where(flock_mask, vxb[:nb], 0.).sum(axis=1)
        fly = np.where(flock_mask, vyb[:nb], 0.).sum(axis=1)
        nfl = flock_mask.sum(axis=1)
        nrm = np.sqrt(flx**2 + fly**2)
        nrm[nfl == 0] = 1.
        flockx = p['alpha'] * flx / nrm
        flocky = p['alpha'] * fly / nrm

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

        frandx = p['ramp'] * np.random.uniform(-1., 1., N)
        frandy = p['ramp'] * np.random.uniform(-1., 1., N)

        vx += (flockx + repx + fpropx + frandx) * dt
        vy += (flocky + repy + fpropy + frandy) * dt
        x[:N] = (x[:N] + vx*dt) % 1.
        x[N:] = (x[N:] + vy*dt) % 1.

    return x.copy(), vx.copy(), vy.copy()


def measure_degree(x, N):
    """Return contact degree of each agent (# within r_cont)."""
    real_dx = x[:N][np.newaxis, :] - x[:N][:, np.newaxis]
    real_dy = x[N:][np.newaxis, :] - x[N:][:, np.newaxis]
    real_dx -= np.round(real_dx); real_dy -= np.round(real_dy)
    rd2 = real_dx**2 + real_dy**2
    within = (rd2 <= R_CONT**2) & (rd2 > 0)
    return within.sum(axis=1)


def run_sis(x0, vx0, vy0, seed_rng, is_immune):
    """Run SIS dynamics from settled flock. Return f_ss (mean last 20 tu)."""
    N = BASE['N']; dt = BASE['dt']
    r0, eps, rf = BASE['r0'], BASE['eps'], BASE['rf']
    v0, mu = BASE['v0'], BASE['mu']
    rb = max(r0, rf, R_CONT)

    x  = x0.copy(); vx = vx0.copy(); vy = vy0.copy()
    is_panicked = np.zeros(N, dtype=bool)

    n0 = max(1, round(F0_FRAC * N))
    susceptible = np.where(~is_immune)[0]
    if susceptible.size >= n0:
        idx0 = seed_rng.choice(susceptible, size=n0, replace=False)
    else:
        idx0 = susceptible
    is_panicked[idx0] = True

    p_recover = 1. - np.exp(-GAMMA * dt)

    f_series = []
    last_window_start = N_ITER - int(20.0 / dt)   # last 20 time units

    for i in range(N_ITER):
        alpha_arr = np.where(is_panicked, PANIC_ALPHA, BASE['alpha'])
        ramp_arr  = np.where(is_panicked, PANIC_RAMP,  BASE['ramp'])

        nb, xb, yb, vxb, vyb = buf_fn(rb, x, vx, vy, N)
        dx = xb[:nb][np.newaxis, :] - x[:N][:, np.newaxis]
        dy = yb[:nb][np.newaxis, :] - x[N:][:, np.newaxis]
        d2 = dx**2 + dy**2

        not_self = np.ones((N, nb), dtype=bool)
        idx = np.arange(min(N, nb))
        not_self[idx, idx] = False

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

        frandx = ramp_arr * seed_rng.uniform(-1., 1., N)
        frandy = ramp_arr * seed_rng.uniform(-1., 1., N)

        vx += (flockx + repx + fpropx + frandx) * dt
        vy += (flocky + repy + fpropy + frandy) * dt
        x[:N] = (x[:N] + vx*dt) % 1.
        x[N:] = (x[N:] + vy*dt) % 1.

        # SIS contagion
        if is_panicked.any() and (~is_panicked & ~is_immune).any():
            real_dx = x[:N][np.newaxis, :] - x[:N][:, np.newaxis]
            real_dy = x[N:][np.newaxis, :] - x[N:][:, np.newaxis]
            real_dx -= np.round(real_dx); real_dy -= np.round(real_dy)
            rd2 = real_dx**2 + real_dy**2
            within = (rd2 <= R_CONT**2) & (rd2 > 0)
            k_arr = within @ is_panicked.astype(np.int32)
            calm_sus = np.where(~is_panicked & ~is_immune)[0]
            if calm_sus.size:
                k_cs = k_arr[calm_sus]
                p_trans = 1. - np.exp(-BETA * k_cs * dt)
                r = seed_rng.uniform(0., 1., calm_sus.size)
                flipped = calm_sus[r < p_trans]
                if flipped.size:
                    is_panicked[flipped] = True

        if is_panicked.any():
            panic_idx = np.where(is_panicked)[0]
            r = seed_rng.uniform(0., 1., panic_idx.size)
            recovered = panic_idx[r < p_recover]
            if recovered.size:
                is_panicked[recovered] = False

        if i >= last_window_start and i % RECORD_EVERY == 0:
            f_series.append(is_panicked.mean())

    return np.mean(f_series) if f_series else 0.0


print('Targeted vs random herd immunity')
print('beta=%.2f gamma=%.2f R0=%.2f, %d seeds' % (BETA, GAMMA, BETA/GAMMA, N_SEEDS))
print('p_immune list: %s' % P_IMMUNE_LIST)

results = {p: {'random': [], 'targeted': []} for p in P_IMMUNE_LIST}
degree_dist_all = []

for s in range(N_SEEDS):
    print('  seed %d: warmup...' % s, flush=True)
    x0, vx0, vy0 = warmup(s, N_WARMUP)
    phi_w = order_parameter(vx0, vy0)
    deg = measure_degree(x0, BASE['N'])
    degree_dist_all.append(deg)
    print('    warmed up: Phi=%.3f  mean_k=%.2f  max_k=%d' % (phi_w, deg.mean(), deg.max()), flush=True)

    sorted_idx = np.argsort(-deg)   # descending degree

    for p_im in P_IMMUNE_LIST:
        N = BASE['N']
        n_im = int(round(p_im * N))

        # random
        rng_r = np.random.default_rng(seed=s * 1000 + int(p_im * 1000))
        is_immune_r = np.zeros(N, dtype=bool)
        if n_im > 0:
            is_immune_r[rng_r.choice(N, size=n_im, replace=False)] = True

        # targeted (top degree)
        is_immune_t = np.zeros(N, dtype=bool)
        if n_im > 0:
            is_immune_t[sorted_idx[:n_im]] = True

        rng_run_r = np.random.default_rng(seed=s * 2000 + int(p_im * 1000))
        rng_run_t = np.random.default_rng(seed=s * 3000 + int(p_im * 1000))

        f_r = run_sis(x0, vx0, vy0, rng_run_r, is_immune_r)
        f_t = run_sis(x0, vx0, vy0, rng_run_t, is_immune_t)
        results[p_im]['random'].append(f_r)
        results[p_im]['targeted'].append(f_t)

print('\n=== Results ===')
print('%8s  %12s  %12s  %12s' % ('p_immune', 'f_ss (random)', 'f_ss (target)', 'diff'))
r_means = []; t_means = []
for p in P_IMMUNE_LIST:
    rm = np.mean(results[p]['random'])
    tm = np.mean(results[p]['targeted'])
    rs = np.std(results[p]['random'])
    ts = np.std(results[p]['targeted'])
    r_means.append(rm); t_means.append(tm)
    print('%8.2f  %5.3f+/-%.3f  %5.3f+/-%.3f  %+.3f' % (p, rm, rs, tm, ts, tm - rm))

# Degree distribution across seeds
all_deg = np.concatenate(degree_dist_all)
print('\nDegree distribution across all seeds:')
print('  mean=%.2f  median=%.0f  std=%.2f  max=%d' % (
    all_deg.mean(), np.median(all_deg), all_deg.std(), all_deg.max()))

# Figures
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
p_arr = np.array(P_IMMUNE_LIST)
r_arr = np.array(r_means); t_arr = np.array(t_means)
r_stds = np.array([np.std(results[p]['random']) for p in P_IMMUNE_LIST])
t_stds = np.array([np.std(results[p]['targeted']) for p in P_IMMUNE_LIST])
ax.errorbar(p_arr, r_arr, yerr=r_stds, marker='o', label='random', color='steelblue', capsize=4)
ax.errorbar(p_arr, t_arr, yerr=t_stds, marker='s', label='targeted (top degree)', color='crimson', capsize=4)
ax.axhline(0.1, ls='--', color='gray', lw=1, label='quench threshold (f=0.1)')
ax.axvline(0.46, ls=':', color='gray', lw=1, label='F30 random threshold (0.46)')
ax.set_xlabel('Immune fraction p_immune')
ax.set_ylabel('Steady-state panic fraction f_ss')
ax.set_title('Targeted vs random herd immunity\nbeta=2.5 gamma=2.0 R0=1.25')
ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_ylim(-0.02, 0.65)

ax = axes[1]
deg_vals, deg_counts = np.unique(all_deg, return_counts=True)
ax.bar(deg_vals, deg_counts / deg_counts.sum(), color='slategray', edgecolor='white', width=0.8)
ax.set_xlabel('Contact degree k (neighbors within r_cont=0.05)')
ax.set_ylabel('Fraction of agents')
ax.set_title('Contact degree distribution in settled flock')
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figures/targeted_immunity_1.png', dpi=120)
plt.close()
print('\n  --> figures/targeted_immunity_1.png')
