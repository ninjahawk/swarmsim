# spatial_vaccination.py -- Spatial vs random vs degree-targeted herd immunity
#
# Finding 36 showed degree-targeted vaccination provides no advantage over random.
# The identified reason: the flock contact network lacks the fat-tailed degree
# heterogeneity required for hub-targeting to matter.  However, F36 also
# identified the real mechanism behind the 2x mean-field inflation of the herd-
# immunity threshold: SPATIAL CLUSTERING of panicked sub-groups.  Agents in the
# flock move in proximity and infect mostly their local neighbors, creating spatial
# clusters of panicked agents that sustain the epidemic even at lower beta/gamma.
#
# F36 concluded: "geographically targeted vaccination -- vaccinating agents
# uniformly spread across the spatial extent of the flock -- may work better than
# degree-targeted vaccination, because it directly targets the spatial-clustering
# mechanism."  This experiment tests that hypothesis directly.
#
# Three strategies:
#   "random"   -- immune agents chosen uniformly at random (F30 baseline)
#   "spatial"  -- immune agents chosen by farthest-point (maxmin) spatial sampling:
#                 each successive immune agent is the one farthest from all already-
#                 selected immune agents.  This maximises spatial coverage.
#   "targeted" -- immune agents chosen as top contact-degree (F36 reference)
#
# Parameters: beta=2.5, gamma=2.0, R0=1.25 (same as F30 and F36).
# Sweep p_immune = 0, 0.10, 0.20, 0.30, 0.40, 0.46, 0.50, 0.60.
# Metric: f_ss = mean panic fraction over last 20 time units of a 100 tu SIS run.

import os
import numpy as np
import matplotlib.pyplot as plt
from flocking import buffer as buf_fn, order_parameter

os.makedirs('figures', exist_ok=True)

N_SEEDS          = 5
N_WARMUP         = 1500
N_ITER           = 10000   # 100 time units of SIS dynamics
RECORD_EVERY     = 50

BASE = dict(
    N=350, r0=0.005, eps=0.1, rf=0.1,
    alpha=1.0, v0=0.02, mu=10.0, ramp=0.1, dt=0.01,
)
PANIC_ALPHA = 0.1
PANIC_RAMP  = 10.0
R_CONT      = 0.05
BETA        = 2.5
GAMMA       = 2.0

P_IMMUNE_LIST = [0.00, 0.10, 0.20, 0.30, 0.40, 0.46, 0.50, 0.60]
F0_FRAC       = 0.05   # 5% seed panic fraction (same as F30 and F36)


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
    """Contact degree of each agent (# within r_cont on torus)."""
    real_dx = x[:N][np.newaxis, :] - x[:N][:, np.newaxis]
    real_dy = x[N:][np.newaxis, :] - x[N:][:, np.newaxis]
    real_dx -= np.round(real_dx); real_dy -= np.round(real_dy)
    rd2 = real_dx**2 + real_dy**2
    within = (rd2 <= R_CONT**2) & (rd2 > 0)
    return within.sum(axis=1)


def spatial_select(x, N, n_im, rng):
    """Farthest-point (maxmin) spatial sampling on the torus.

    Returns indices of n_im agents maximally spread across the flock.
    Each successive agent is the one with the largest minimum distance to
    all already-selected agents.  O(N * n_im) on torus distance.
    """
    if n_im == 0:
        return np.array([], dtype=int)

    pos_x = x[:N]
    pos_y = x[N:]

    # Start with a fixed first agent (centre-of-mass nearest agent)
    cx = np.mean(pos_x)
    cy = np.mean(pos_y)
    dx0 = pos_x - cx; dy0 = pos_y - cy
    dx0 -= np.round(dx0); dy0 -= np.round(dy0)
    first = int(np.argmin(dx0**2 + dy0**2))

    selected = [first]
    # min_dist[i] = distance from agent i to the nearest selected agent so far
    dx = pos_x - pos_x[first]; dy = pos_y - pos_y[first]
    dx -= np.round(dx); dy -= np.round(dy)
    min_dist = np.sqrt(dx**2 + dy**2)
    min_dist[first] = -1.   # mark as selected

    while len(selected) < n_im:
        next_agent = int(np.argmax(min_dist))
        selected.append(next_agent)
        min_dist[next_agent] = -1.
        # Update min distances with new selection
        dx = pos_x - pos_x[next_agent]; dy = pos_y - pos_y[next_agent]
        dx -= np.round(dx); dy -= np.round(dy)
        new_dist = np.sqrt(dx**2 + dy**2)
        min_dist = np.where(min_dist >= 0,
                            np.minimum(min_dist, new_dist),
                            min_dist)

    return np.array(selected, dtype=int)


def run_sis(x0, vx0, vy0, seed_rng, is_immune):
    """Run SIS dynamics. Return f_ss (mean panic fraction, last 20 tu)."""
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
    last_window_start = N_ITER - int(20.0 / dt)   # last 20 time units

    f_series = []

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


# -----------------------------------------------------------------------
print('Spatial vs random vs degree-targeted herd immunity')
print('beta=%.2f  gamma=%.2f  R0=%.2f  %d seeds' % (BETA, GAMMA, BETA/GAMMA, N_SEEDS))
print('p_immune list: %s' % P_IMMUNE_LIST)
print()

results = {p: {'random': [], 'spatial': [], 'targeted': []} for p in P_IMMUNE_LIST}
spatial_spread_all = []    # mean min-distance between selected agents (measure of coverage)
degree_dist_all   = []

for s in range(N_SEEDS):
    print('  seed %d: warmup...' % s, flush=True)
    x0, vx0, vy0 = warmup(s, N_WARMUP)
    phi_w = order_parameter(vx0, vy0)
    deg = measure_degree(x0, BASE['N'])
    degree_dist_all.append(deg)
    sorted_deg_idx = np.argsort(-deg)   # descending degree
    print('    Phi=%.3f  mean_k=%.2f  max_k=%d' % (phi_w, deg.mean(), deg.max()), flush=True)

    rng_sel = np.random.default_rng(seed=s * 7777)

    for p_im in P_IMMUNE_LIST:
        N  = BASE['N']
        n_im = int(round(p_im * N))

        # Random
        rng_r = np.random.default_rng(seed=s * 1000 + int(p_im * 1000))
        is_immune_r = np.zeros(N, dtype=bool)
        if n_im > 0:
            is_immune_r[rng_r.choice(N, size=n_im, replace=False)] = True

        # Spatial (farthest-point sampling)
        rng_sp = np.random.default_rng(seed=s * 4000 + int(p_im * 1000))
        sp_idx = spatial_select(x0, N, n_im, rng_sp)
        is_immune_s = np.zeros(N, dtype=bool)
        if sp_idx.size > 0:
            is_immune_s[sp_idx] = True

        # Targeted (top degree)
        is_immune_t = np.zeros(N, dtype=bool)
        if n_im > 0:
            is_immune_t[sorted_deg_idx[:n_im]] = True

        rng_run_r = np.random.default_rng(seed=s * 2000 + int(p_im * 1000))
        rng_run_s = np.random.default_rng(seed=s * 5000 + int(p_im * 1000))
        rng_run_t = np.random.default_rng(seed=s * 3000 + int(p_im * 1000))

        f_r = run_sis(x0, vx0, vy0, rng_run_r, is_immune_r)
        f_s = run_sis(x0, vx0, vy0, rng_run_s, is_immune_s)
        f_t = run_sis(x0, vx0, vy0, rng_run_t, is_immune_t)
        results[p_im]['random'].append(f_r)
        results[p_im]['spatial'].append(f_s)
        results[p_im]['targeted'].append(f_t)
        print('    p=%.2f: random=%.3f  spatial=%.3f  targeted=%.3f' % (p_im, f_r, f_s, f_t),
              flush=True)

# -----------------------------------------------------------------------
print()
print('=== Results ===')
print('%8s  %14s  %14s  %14s' % ('p_immune', 'f_ss (random)', 'f_ss (spatial)', 'f_ss (target)'))
r_means = []; s_means = []; t_means = []
r_stds  = []; s_stds  = []; t_stds  = []
for p in P_IMMUNE_LIST:
    rm = np.mean(results[p]['random']);  rs = np.std(results[p]['random'])
    sm = np.mean(results[p]['spatial']); ss = np.std(results[p]['spatial'])
    tm = np.mean(results[p]['targeted']);ts = np.std(results[p]['targeted'])
    r_means.append(rm); s_means.append(sm); t_means.append(tm)
    r_stds.append(rs);  s_stds.append(ss);  t_stds.append(ts)
    print('%8.2f  %5.3f+/-%.3f  %5.3f+/-%.3f  %5.3f+/-%.3f' % (
        p, rm, rs, sm, ss, tm, ts))

# Degree distribution summary
all_deg = np.concatenate(degree_dist_all)
print()
print('Degree distribution: mean=%.2f  median=%.0f  std=%.2f  max=%d' % (
    all_deg.mean(), np.median(all_deg), all_deg.std(), all_deg.max()))

# -----------------------------------------------------------------------
# Figures
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
p_arr = np.array(P_IMMUNE_LIST)
r_arr = np.array(r_means); s_arr = np.array(s_means); t_arr = np.array(t_means)
r_e = np.array(r_stds);   s_e = np.array(s_stds);   t_e = np.array(t_stds)

ax.errorbar(p_arr, r_arr, yerr=r_e, marker='o', label='random',
            color='steelblue', capsize=4)
ax.errorbar(p_arr, s_arr, yerr=s_e, marker='^', label='spatial (farthest-point)',
            color='seagreen', capsize=4)
ax.errorbar(p_arr, t_arr, yerr=t_e, marker='s', label='degree-targeted (F36 ref)',
            color='crimson', capsize=4, alpha=0.6, ls='--')
ax.axhline(0.1, ls='--', color='gray', lw=1, label='quench threshold (f=0.1)')
ax.axvline(0.46, ls=':', color='gray', lw=1, label='F30/F36 threshold (0.46)')
ax.set_xlabel('Immune fraction p_immune')
ax.set_ylabel('Steady-state panic fraction f_ss')
ax.set_title('Spatial vs random vs targeted herd immunity\nbeta=2.5  gamma=2.0  R0=1.25')
ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(-0.03, 0.65)

# Panel 2: spatial coverage diagnostic -- histogram of min inter-immune distances
# for random vs spatial at p=0.30 (about 105 agents)
ax2 = axes[1]
p_diag = 0.30
n_diag = int(round(p_diag * BASE['N']))
rand_min_dists = []
spat_min_dists = []
rng_diag = np.random.default_rng(999)

for s in range(N_SEEDS):
    x0, _, _ = warmup(s, N_WARMUP)
    N = BASE['N']

    # Random selection
    rng_r = np.random.default_rng(seed=s * 1000 + int(p_diag * 1000))
    rand_idx = rng_r.choice(N, size=n_diag, replace=False)
    # Spatial selection
    rng_sp = np.random.default_rng(seed=s * 4000 + int(p_diag * 1000))
    spat_idx = spatial_select(x0, N, n_diag, rng_sp)

    pos_x = x0[:N]; pos_y = x0[N:]
    for grp, idx in [('rand', rand_idx), ('spat', spat_idx)]:
        px = pos_x[idx]; py = pos_y[idx]
        for i in range(len(idx)):
            dx = px - px[i]; dy = py - py[i]
            dx -= np.round(dx); dy -= np.round(dy)
            d = np.sqrt(dx**2 + dy**2)
            d[i] = np.inf
            if grp == 'rand':
                rand_min_dists.append(d.min())
            else:
                spat_min_dists.append(d.min())

bins = np.linspace(0, 0.12, 30)
ax2.hist(rand_min_dists, bins=bins, alpha=0.5, label='random', color='steelblue', density=True)
ax2.hist(spat_min_dists, bins=bins, alpha=0.5, label='spatial', color='seagreen', density=True)
ax2.set_xlabel('Min distance to nearest immune neighbor')
ax2.set_ylabel('Density')
ax2.set_title('Immune-agent spatial coverage (p=0.30)\nSpatial = fewer isolated clusters')
ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/spatial_vaccination_1.png', dpi=120)
plt.close()
print()
print('  --> figures/spatial_vaccination_1.png')
