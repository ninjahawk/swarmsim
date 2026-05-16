# segregation.py -- Active / passive agent segregation (Charbonneau Sec 10.4)
#
# Section 10.4 of the textbook describes a mixed population where one group
# is more "active" (higher speed) than another, and asks whether they segregate.
# In this implementation we vary v0 between two sub-populations:
#   Active  agents: v0 = v0_active   (default 1.0)
#   Passive agents: v0 = v0_passive  (default 0.3)
# Everything else (alpha, ramp, mu, etc.) is identical between groups.
#
# Question: do the two groups spatially segregate (e.g., active agents
# at the leading edge of the flock, passive agents trailing), or do they
# remain well mixed?
#
# Metric: Pearson cross-group separation in the flock-aligned frame.  We
# transform coordinates so the mean flock heading is +x, then compare the
# mean x-coordinate of each group.  Positive segregation index s = (mean_x_active
# - mean_x_passive) / Rg means active leads passive by s * Rg.
#
# Experiments:
#   1. v0 contrast sweep: hold v0_active=1, vary v0_passive=1.0 (no contrast)
#      down to 0.1 (large contrast).  Plot segregation index vs contrast.
#   2. Fraction sweep at fixed contrast (v0_passive=0.3): vary f_active 0.1..0.9
#      to see if minority/majority status matters.

import os
import numpy as np
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from flocking import buffer as buf_fn, order_parameter
from model import _periodic_com

os.makedirs('figures', exist_ok=True)

N_SEEDS = 5

BASE = dict(
    N=350, r0=0.005, eps=0.1, rf=0.1,
    alpha=1.0, mu=10.0, ramp=0.5, dt=0.01,
    n_iter=4000,
)


def run_mixed(v0_active=1.0, v0_passive=0.3, f_active=0.5,
              seed=None, n_frames=200, overrides=None):
    if seed is not None:
        np.random.seed(seed)

    p = BASE.copy()
    if overrides:
        p.update(overrides)

    N      = p['N']
    dt     = p['dt']
    n_iter = p['n_iter']
    r0, eps, rf = p['r0'], p['eps'], p['rf']
    mu, alpha   = p['mu'], p['alpha']
    ramp        = p['ramp']
    frame_every = max(1, n_iter // n_frames)

    n_active = round(f_active * N)
    is_active = np.zeros(N, dtype=bool)
    active_idx = np.random.choice(N, size=n_active, replace=False)
    is_active[active_idx] = True
    v0_arr = np.where(is_active, v0_active, v0_passive)

    x  = np.zeros(2*N)
    x[:N] = np.random.uniform(0., 1., N)
    x[N:] = np.random.uniform(0., 1., N)
    vx = np.random.uniform(-1., 1., N) * v0_active
    vy = np.random.uniform(-1., 1., N) * v0_active

    rb = max(r0, rf)
    frames = []

    for i in range(n_iter):
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
        flockx = alpha * flx / nrm
        flocky = alpha * fly / nrm

        rep_mask = (d2 <= (2*r0)**2) & not_self & (d2 > 0)
        d_safe   = np.where(rep_mask, np.sqrt(d2), 1.)
        base_r   = np.where(rep_mask, 1. - d_safe/(2.*r0), 0.)
        strength = np.where(rep_mask, eps * base_r**1.5 / d_safe, 0.)
        repx = (-strength * dx).sum(axis=1)
        repy = (-strength * dy).sum(axis=1)

        vnorm  = np.sqrt(vx**2 + vy**2)
        vnorms = np.where(vnorm == 0, 1., vnorm)
        fpropx = mu * (v0_arr - vnorm) * vx / vnorms
        fpropy = mu * (v0_arr - vnorm) * vy / vnorms

        frandx = ramp * np.random.uniform(-1., 1., N)
        frandy = ramp * np.random.uniform(-1., 1., N)

        vx += (flockx + repx + fpropx + frandx) * dt
        vy += (flocky + repy + fpropy + frandy) * dt
        x[:N] = (x[:N] + vx*dt) % 1.
        x[N:] = (x[N:] + vy*dt) % 1.

        if i % frame_every == 0:
            frames.append((x[:N].copy(), x[N:].copy(),
                           vx.copy(), vy.copy(),
                           is_active.copy()))

    return frames


def segregation_index(px, py, vx, vy, is_active):
    """Return signed segregation s = (mean_x_active - mean_x_passive) / Rg
       in the flock-aligned frame (heading = +x)."""
    cx, cy = _periodic_com(px), _periodic_com(py)
    rel_x = (px - cx + 0.5) % 1.0 - 0.5
    rel_y = (py - cy + 0.5) % 1.0 - 0.5
    head_x = vx.mean(); head_y = vy.mean()
    h = np.sqrt(head_x**2 + head_y**2)
    if h < 1e-9:
        return 0.0
    head_x /= h; head_y /= h
    # rotate so heading is +x
    rotx =  head_x * rel_x + head_y * rel_y
    roty = -head_y * rel_x + head_x * rel_y
    Rg = np.sqrt((rotx**2 + roty**2).mean())
    if Rg < 1e-9:
        return 0.0
    return (rotx[is_active].mean() - rotx[~is_active].mean()) / Rg


# Exp 1: v0 contrast sweep at f_active=0.5
print('Exp 1: contrast sweep (f_active=0.5)')
v0p_vals = [1.0, 0.7, 0.5, 0.3, 0.2, 0.1]
res1 = {}
for v0p in v0p_vals:
    s_runs=[]; phi_runs=[]
    for s in range(N_SEEDS):
        frames = run_mixed(v0_passive=v0p, f_active=0.5, seed=s)
        last = frames[-30:]
        s_vals=[]
        for px, py, vx, vy, is_a in last:
            s_vals.append(segregation_index(px, py, vx, vy, is_a))
        s_runs.append(np.mean(s_vals))
        phi_runs.append(np.mean([order_parameter(vx, vy) for _, _, vx, vy, _ in last]))
    res1[v0p] = dict(s=np.mean(s_runs), s_std=np.std(s_runs),
                     phi=np.mean(phi_runs))
    print('  v0_passive=%.2f  segregation=%.3f +/- %.3f  Phi=%.3f' % (
        v0p, res1[v0p]['s'], res1[v0p]['s_std'], res1[v0p]['phi']))


# Exp 2: fraction sweep at v0_passive=0.3
print('\nExp 2: fraction sweep (v0_passive=0.3)')
fa_vals = [0.1, 0.25, 0.5, 0.75, 0.9]
res2 = {}
for fa in fa_vals:
    s_runs=[]; phi_runs=[]
    for s in range(N_SEEDS):
        frames = run_mixed(v0_passive=0.3, f_active=fa, seed=s)
        last = frames[-30:]
        s_vals=[]
        for px, py, vx, vy, is_a in last:
            s_vals.append(segregation_index(px, py, vx, vy, is_a))
        s_runs.append(np.mean(s_vals))
        phi_runs.append(np.mean([order_parameter(vx, vy) for _, _, vx, vy, _ in last]))
    res2[fa] = dict(s=np.mean(s_runs), s_std=np.std(s_runs),
                    phi=np.mean(phi_runs))
    print('  f_active=%.2f  segregation=%.3f +/- %.3f  Phi=%.3f' % (
        fa, res2[fa]['s'], res2[fa]['s_std'], res2[fa]['phi']))


# Snapshot for v0_passive=0.2
print('\nExp 3: snapshot gallery (v0_passive=0.2)')
frames_snap = run_mixed(v0_passive=0.2, f_active=0.5, seed=1,
                        overrides={'N': 200, 'n_iter': 5000}, n_frames=200)

# Figures
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Active/passive segregation (%d seeds)' % N_SEEDS, fontsize=11)

ax = axes[0]
contrast = [1.0 - v0p for v0p in v0p_vals]
s_means  = [res1[v0p]['s']     for v0p in v0p_vals]
s_stds   = [res1[v0p]['s_std'] for v0p in v0p_vals]
ax.errorbar(contrast, s_means, yerr=s_stds, fmt='o-', color='crimson', lw=2, capsize=4)
ax.set_xlabel('Speed contrast (1 - v0_passive)')
ax.set_ylabel('Segregation index (active lead in Rg units)')
ax.set_title('Contrast sweep at f_active=0.5')
ax.axhline(0, ls=':', color='k', alpha=0.4)
ax.grid(alpha=0.3)

ax = axes[1]
s_means2 = [res2[fa]['s']     for fa in fa_vals]
s_stds2  = [res2[fa]['s_std'] for fa in fa_vals]
ax.errorbar(fa_vals, s_means2, yerr=s_stds2, fmt='o-', color='steelblue', lw=2, capsize=4)
ax.set_xlabel('Active fraction f_active')
ax.set_ylabel('Segregation index')
ax.set_title('Fraction sweep at v0_passive=0.3')
ax.axhline(0, ls=':', color='k', alpha=0.4)
ax.grid(alpha=0.3)

ax = axes[2]
# show snapshot at final time
px, py, vx, vy, is_a = frames_snap[-1]
ax.scatter(px[is_a], py[is_a], s=8, color='crimson', label='active (v0=1.0)')
ax.scatter(px[~is_a], py[~is_a], s=8, color='steelblue', label='passive (v0=0.2)')
sp = np.sqrt(vx**2 + vy**2); sp[sp==0] = 1.
ax.quiver(px, py, vx/sp, vy/sp, scale=80, width=0.003, color='gray', alpha=0.4)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
ax.set_xticks([]); ax.set_yticks([])
ax.set_title('Final snapshot (v0_passive=0.2)')
ax.legend(fontsize=8, loc='upper right')

plt.tight_layout()
plt.savefig('figures/segregation_1_summary.png', dpi=120)
plt.close()
print('  --> figures/segregation_1_summary.png')

print('\nSegregation analysis complete.')
