# segregation_alpha.py -- Active/passive segregation via alpha contrast
#
# Finding 24 showed that mixed-v0 populations do not spatially segregate -- the
# alignment force homogenises group speed.  But the alignment force MAGNITUDE
# is set by alpha.  What if the two groups have different alpha values?
# An agent with smaller alpha is less tightly coupled to its neighbors and may
# fall behind, while a high-alpha agent is more tightly coupled.
#
# Experiments:
#   1. Alpha contrast sweep: hold alpha_active=1.0, vary alpha_passive 0..1.0.
#      Measure along-heading segregation index + a perpendicular component.
#   2. Local-purity diagnostic: for each agent, what fraction of its rf-neighbors
#      are same-type?  A perfectly mixed population gives purity = f_type.
#      Spatial segregation produces purity > f_type.

import os
import numpy as np
import matplotlib.pyplot as plt
from flocking import buffer as buf_fn, order_parameter
from model import _periodic_com

os.makedirs('figures', exist_ok=True)

N_SEEDS = 5

BASE = dict(
    N=350, r0=0.005, eps=0.1, rf=0.1,
    v0=1.0, mu=10.0, ramp=0.5, dt=0.01,
    n_iter=4000,
)


def run_mixed(alpha_active=1.0, alpha_passive=0.3, f_active=0.5,
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
    v0, mu       = p['v0'], p['mu']
    ramp         = p['ramp']
    frame_every  = max(1, n_iter // n_frames)

    n_active = round(f_active * N)
    is_active = np.zeros(N, dtype=bool)
    active_idx = np.random.choice(N, size=n_active, replace=False)
    is_active[active_idx] = True
    alpha_arr = np.where(is_active, alpha_active, alpha_passive)

    x  = np.zeros(2*N)
    x[:N] = np.random.uniform(0., 1., N)
    x[N:] = np.random.uniform(0., 1., N)
    vx = np.random.uniform(-1., 1., N) * v0
    vy = np.random.uniform(-1., 1., N) * v0

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


def segregation_metrics(px, py, vx, vy, is_active, rf=0.1):
    """
    Return (seg_along, seg_perp, purity_active, purity_passive).
    seg_along : (mean rotx_active - mean rotx_passive) / Rg
    seg_perp  : (mean roty_active - mean roty_passive) / Rg
    purity_*  : fraction of rf-neighbors of same type
    """
    N = len(px)
    cx, cy = _periodic_com(px), _periodic_com(py)
    rel_x = (px - cx + 0.5) % 1.0 - 0.5
    rel_y = (py - cy + 0.5) % 1.0 - 0.5
    head_x = vx.mean(); head_y = vy.mean()
    h = np.sqrt(head_x**2 + head_y**2)
    if h < 1e-9:
        seg_a = seg_p = 0.0
    else:
        head_x /= h; head_y /= h
        rotx =  head_x * rel_x + head_y * rel_y
        roty = -head_y * rel_x + head_x * rel_y
        Rg = np.sqrt((rotx**2 + roty**2).mean())
        if Rg < 1e-9:
            seg_a = seg_p = 0.0
        else:
            seg_a = (rotx[is_active].mean() - rotx[~is_active].mean()) / Rg
            seg_p = (roty[is_active].mean() - roty[~is_active].mean()) / Rg

    dx = px[np.newaxis, :] - px[:, np.newaxis]
    dy = py[np.newaxis, :] - py[:, np.newaxis]
    dx -= np.round(dx); dy -= np.round(dy)
    d2 = dx**2 + dy**2
    adj = (d2 <= rf**2) & (d2 > 0)
    deg = adj.sum(axis=1)
    deg_safe = np.where(deg == 0, 1, deg)
    same_active = adj @ is_active.astype(np.int32)
    purity_active = (same_active[is_active] / deg_safe[is_active]).mean()
    same_passive = adj @ (~is_active).astype(np.int32)
    purity_passive = (same_passive[~is_active] / deg_safe[~is_active]).mean()
    return seg_a, seg_p, purity_active, purity_passive


# Exp 1: alpha contrast sweep at f_active=0.5
print('Exp 1: alpha contrast sweep (alpha_active=1.0, f_active=0.5)')
alpha_p_vals = [1.0, 0.7, 0.5, 0.3, 0.1, 0.0]
res1 = {}
for ap in alpha_p_vals:
    rs=[]
    for s in range(N_SEEDS):
        frames = run_mixed(alpha_active=1.0, alpha_passive=ap,
                           f_active=0.5, seed=s)
        last = frames[-30:]
        metrics = [segregation_metrics(px, py, vx, vy, is_a)
                   for px, py, vx, vy, is_a in last]
        m = np.mean(metrics, axis=0)
        phi = np.mean([order_parameter(vx, vy) for _, _, vx, vy, _ in last])
        rs.append((m[0], m[1], m[2], m[3], phi))
    arr = np.array(rs)
    res1[ap] = dict(seg_a=arr[:,0].mean(), seg_a_std=arr[:,0].std(),
                    seg_p=arr[:,1].mean(),
                    purity_a=arr[:,2].mean(), purity_p=arr[:,3].mean(),
                    phi=arr[:,4].mean())
    print('  alpha_p=%.2f  seg_along=%.3f +/- %.3f  seg_perp=%.3f  '
          'purity_a=%.3f purity_p=%.3f  Phi=%.3f' % (
        ap, res1[ap]['seg_a'], res1[ap]['seg_a_std'],
        res1[ap]['seg_p'], res1[ap]['purity_a'], res1[ap]['purity_p'],
        res1[ap]['phi']))


# Exp 2: fraction sweep at alpha_passive=0.1
print('\nExp 2: fraction sweep (alpha_active=1.0, alpha_passive=0.1)')
fa_vals = [0.1, 0.25, 0.5, 0.75, 0.9]
res2 = {}
for fa in fa_vals:
    rs=[]
    for s in range(N_SEEDS):
        frames = run_mixed(alpha_active=1.0, alpha_passive=0.1,
                           f_active=fa, seed=s)
        last = frames[-30:]
        metrics = [segregation_metrics(px, py, vx, vy, is_a)
                   for px, py, vx, vy, is_a in last]
        m = np.mean(metrics, axis=0)
        phi = np.mean([order_parameter(vx, vy) for _, _, vx, vy, _ in last])
        rs.append((m[0], m[1], m[2], m[3], phi))
    arr = np.array(rs)
    res2[fa] = dict(seg_a=arr[:,0].mean(), seg_a_std=arr[:,0].std(),
                    purity_a=arr[:,2].mean(), purity_p=arr[:,3].mean(),
                    phi=arr[:,4].mean())
    print('  f_active=%.2f  seg_along=%.3f +/- %.3f  '
          'purity_a=%.3f purity_p=%.3f  Phi=%.3f' % (
        fa, res2[fa]['seg_a'], res2[fa]['seg_a_std'],
        res2[fa]['purity_a'], res2[fa]['purity_p'], res2[fa]['phi']))


# Snapshot
print('\nExp 3: snapshot gallery (alpha_passive=0.0)')
frames_snap = run_mixed(alpha_active=1.0, alpha_passive=0.0, f_active=0.5,
                        seed=1, overrides={'N': 200, 'n_iter': 5000})


# Figures
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Alpha-contrast segregation (%d seeds)' % N_SEEDS, fontsize=11)

ax = axes[0]
contrast = [1.0 - ap for ap in alpha_p_vals]
seg = [res1[ap]['seg_a'] for ap in alpha_p_vals]
std = [res1[ap]['seg_a_std'] for ap in alpha_p_vals]
ax.errorbar(contrast, seg, yerr=std, fmt='o-', color='crimson', lw=2, capsize=4,
            label='along-heading seg')
seg_p = [res1[ap]['seg_p'] for ap in alpha_p_vals]
ax.plot(contrast, seg_p, 's--', color='steelblue', lw=2, label='perp seg')
ax.set_xlabel('Alpha contrast (1 - alpha_passive)')
ax.set_ylabel('Segregation index (Rg units)')
ax.set_title('Contrast sweep at f_active=0.5')
ax.axhline(0, ls=':', color='k', alpha=0.4)
ax.grid(alpha=0.3); ax.legend(fontsize=8)

ax = axes[1]
purity_a = [res1[ap]['purity_a'] for ap in alpha_p_vals]
purity_p = [res1[ap]['purity_p'] for ap in alpha_p_vals]
ax.plot(contrast, purity_a, 'o-', color='crimson', lw=2, label='active purity')
ax.plot(contrast, purity_p, 's-', color='steelblue', lw=2, label='passive purity')
ax.axhline(0.5, ls=':', color='k', alpha=0.4, label='random (=f)')
ax.set_xlabel('Alpha contrast (1 - alpha_passive)')
ax.set_ylabel('Same-type fraction of rf-neighbors')
ax.set_title('Local purity (segregation diagnostic)')
ax.set_ylim(0.4, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=8)

ax = axes[2]
px, py, vx, vy, is_a = frames_snap[-1]
ax.scatter(px[is_a], py[is_a], s=8, color='crimson', label='active (alpha=1.0)')
ax.scatter(px[~is_a], py[~is_a], s=8, color='steelblue', label='passive (alpha=0.0)')
sp = np.sqrt(vx**2 + vy**2); sp[sp==0] = 1.
ax.quiver(px, py, vx/sp, vy/sp, scale=80, width=0.003, color='gray', alpha=0.4)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
ax.set_xticks([]); ax.set_yticks([])
ax.set_title('Final snapshot (alpha_passive=0)')
ax.legend(fontsize=8, loc='upper right')

plt.tight_layout()
plt.savefig('figures/segregation_alpha_1_summary.png', dpi=120)
plt.close()
print('  --> figures/segregation_alpha_1_summary.png')

print('\nAlpha-segregation analysis complete.')
