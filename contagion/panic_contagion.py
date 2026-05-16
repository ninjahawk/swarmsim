# panic_contagion.py -- Panic with social contagion (Charbonneau Ch.10 follow-up)
#
# Finding 18 showed that in the static-panic model panic does NOT propagate:
# the flock's alignment force dominates and calm agents stay coherent even at
# f=20% panic.  The book's "Why You Should Never Panic" result implicitly
# requires a CONTAGION mechanism (panic spreads through contact).  This script
# implements that mechanism and asks two questions:
#
#   Q1: Is there a critical contagion rate beta_c above which a small seed
#       (f0 ~ 1%) saturates the population?
#   Q2: Once contagion is on, does calm_Phi finally break -- i.e., does the
#       flock disrupt because panic propagates, even if static panic does not?
#
# Contagion rule (continuous-time SI process discretised at dt):
#   For each calm agent i, let k_i = number of panicked agents within r_cont.
#   Per timestep, transition to panicked with probability
#       p = 1 - exp(-beta * k_i * dt)
#   No recovery (SI, not SIR) -- panic is treated as absorbing here.
#
# Panicked agents behave as in panic.py (Finding 18):
#   alpha_p = 0.1, ramp_p = 10.0 (weak alignment, large noise).
#
# Experiments
# -----------
#   1. Beta sweep at fixed f0=1%, N=350 -- final panic fraction f_inf, calm_Phi,
#      global Phi at steady state.
#   2. Time series of f(t) at sub/supercritical beta.
#   3. Initial seed sensitivity: at fixed beta, vary f0 (0.5% .. 5%).
#   4. Snapshot gallery of contagion spreading.

import os
import numpy as np
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from flocking import buffer as buf_fn, order_parameter

os.makedirs('figures', exist_ok=True)

N_SEEDS = 6

BASE = dict(
    N=350, r0=0.005, eps=0.1, rf=0.1,
    alpha=1.0, v0=1.0, mu=10.0, ramp=0.5, dt=0.01,
    n_iter=4000,
)
PANIC_ALPHA = 0.1
PANIC_RAMP  = 10.0
R_CONT      = 0.05   # contagion radius (half of rf=0.1; near-neighbor only)


def run_contagion(beta=0.0, f0=0.01, n_frames=200, seed=None, overrides=None):
    """
    Mixed calm+panic flock with contagion.

    beta : contagion rate (per panicked-neighbor per unit time).
           beta=0 reproduces static panic (Finding 18).
    f0   : initial panicked fraction.

    Returns frames as list of (px, py, vx, vy, is_panicked).
    """
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
    frame_every  = max(1, n_iter // n_frames)

    n0 = max(1, round(f0 * N)) if f0 > 0 else 0
    is_panicked = np.zeros(N, dtype=bool)
    if n0 > 0:
        seed_idx = np.random.choice(N, size=n0, replace=False)
        is_panicked[seed_idx] = True

    x  = np.zeros(2*N)
    x[:N] = np.random.uniform(0., 1., N)
    x[N:] = np.random.uniform(0., 1., N)
    vx = np.random.uniform(-1., 1., N) * v0
    vy = np.random.uniform(-1., 1., N) * v0

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

        # flocking
        flock_mask = (d2 <= rf**2) & not_self
        flx = np.where(flock_mask, vxb[:nb], 0.).sum(axis=1)
        fly = np.where(flock_mask, vyb[:nb], 0.).sum(axis=1)
        nfl = flock_mask.sum(axis=1)
        nrm = np.sqrt(flx**2 + fly**2)
        nrm[nfl == 0] = 1.
        flockx = alpha_arr * flx / nrm
        flocky = alpha_arr * fly / nrm

        # repulsion
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

        frandx = ramp_arr * np.random.uniform(-1., 1., N)
        frandy = ramp_arr * np.random.uniform(-1., 1., N)

        fx = flockx + repx + fpropx + frandx
        fy = flocky + repy + fpropy + frandy

        vx += fx * dt
        vy += fy * dt
        x[:N] = (x[:N] + vx*dt) % 1.
        x[N:] = (x[N:] + vy*dt) % 1.

        # ---- contagion step --------------------------------------------------
        # For each calm agent, count panicked neighbors within R_CONT.
        # buffer's first N entries are the real agents, in order.
        if beta > 0 and is_panicked.any() and (~is_panicked).any():
            # is_panicked status of each buffer entry: real agents 0..N-1, ghosts copy
            # We need a mask over the buffer indicating "this ghost mirrors a panicked real".
            # Reconstruct buffer ownership: ghosts are added in order for each real k.
            # Simplest: ignore ghosts for contagion (only intra-real contact within R_CONT).
            # Since R_CONT=0.05 << 1 and flock is much smaller than domain, periodic
            # wrap-around contagion contributes negligibly.
            real_dx = x[:N][np.newaxis, :] - x[:N][:, np.newaxis]
            real_dy = x[N:][np.newaxis, :] - x[N:][:, np.newaxis]
            real_dx -= np.round(real_dx)  # periodic shortest-displacement
            real_dy -= np.round(real_dy)
            rd2 = real_dx**2 + real_dy**2
            within = (rd2 <= R_CONT**2) & (rd2 > 0)
            # k_i = number of panicked agents within R_CONT of i
            k = within @ is_panicked.astype(np.int32)
            calm_idx = np.where(~is_panicked)[0]
            if calm_idx.size:
                k_calm = k[calm_idx]
                p_trans = 1. - np.exp(-beta * k_calm * dt)
                r = np.random.uniform(0., 1., calm_idx.size)
                flipped = calm_idx[r < p_trans]
                if flipped.size:
                    is_panicked[flipped] = True

        if i % frame_every == 0:
            frames.append((x[:N].copy(), x[N:].copy(),
                           vx.copy(), vy.copy(),
                           is_panicked.copy()))

    return frames


# =============================================================================
# EXP 1: BETA SWEEP at f0=1%
# =============================================================================
print('Exp 1: Beta sweep at f0=1%% (%d seeds)' % N_SEEDS)
beta_vals = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

res1 = {}
for beta in beta_vals:
    f_inf_runs = []; phi_runs = []; calm_phi_runs = []; t_half_runs = []
    for s in range(N_SEEDS):
        frames = run_contagion(beta=beta, f0=0.01, n_frames=200, seed=s)
        last = frames[-30:]
        # final panic fraction
        f_inf = np.mean([m.mean() for _, _, _, _, m in last])
        # global Phi
        phi_ss = np.mean([order_parameter(vx, vy) for _, _, vx, vy, _ in last])
        # calm Phi
        calm_phi_t = []
        for _, _, vx, vy, m in last:
            calm = ~m
            if calm.sum() > 1:
                calm_phi_t.append(order_parameter(vx[calm], vy[calm]))
        calm_phi_ss = np.mean(calm_phi_t) if calm_phi_t else float('nan')

        # time to half-panicked across the run
        f_t = np.array([m.mean() for _, _, _, _, m in frames])
        idx = np.where(f_t >= 0.5)[0]
        t_half = idx[0] if idx.size else -1

        f_inf_runs.append(f_inf)
        phi_runs.append(phi_ss)
        calm_phi_runs.append(calm_phi_ss)
        t_half_runs.append(t_half)

    res1[beta] = dict(
        f_inf=np.mean(f_inf_runs), f_inf_std=np.std(f_inf_runs),
        phi=np.mean(phi_runs), phi_std=np.std(phi_runs),
        calm_phi=np.nanmean(calm_phi_runs),
        t_half=np.mean([t for t in t_half_runs if t >= 0]) if any(t>=0 for t in t_half_runs) else -1,
    )
    print('  beta=%5.1f  f_inf=%.3f +/- %.3f  Phi=%.3f  calm_Phi=%.3f  t_half_frames=%.0f' % (
        beta, res1[beta]['f_inf'], res1[beta]['f_inf_std'],
        res1[beta]['phi'], res1[beta]['calm_phi'], res1[beta]['t_half']))


# =============================================================================
# EXP 2: TIME SERIES OF f(t)
# =============================================================================
print('\nExp 2: Time series of panic fraction f(t)')
ts_results = {}
n_fr2 = 200
for beta in [0.5, 2.0, 5.0, 20.0]:
    f_runs = []; phi_runs = []; calm_runs = []
    for s in range(N_SEEDS):
        frames = run_contagion(beta=beta, f0=0.01, n_frames=n_fr2, seed=s)
        f_runs.append([m.mean() for _, _, _, _, m in frames])
        phi_runs.append([order_parameter(vx, vy) for _, _, vx, vy, _ in frames])
        calm_t = []
        for _, _, vx, vy, m in frames:
            calm = ~m
            calm_t.append(order_parameter(vx[calm], vy[calm]) if calm.sum() > 1 else 1.0)
        calm_runs.append(calm_t)
    ts_results[beta] = dict(
        f=np.array(f_runs), phi=np.array(phi_runs), calm=np.array(calm_runs))
    print('  beta=%4.1f  final f=%.3f  final calm_Phi=%.3f' % (
        beta, ts_results[beta]['f'][:,-1].mean(),
        ts_results[beta]['calm'][:,-1].mean()))

dt_val = BASE['dt']
fs2 = max(1, BASE['n_iter'] // n_fr2)
t_axis = np.arange(n_fr2) * fs2 * dt_val


# =============================================================================
# EXP 3: INITIAL SEED SENSITIVITY at beta=2.0
# =============================================================================
print('\nExp 3: Initial seed sensitivity at beta=2.0')
f0_vals = [0.005, 0.01, 0.02, 0.05, 0.10]
res3 = {}
for f0 in f0_vals:
    f_inf_runs = []
    for s in range(N_SEEDS):
        frames = run_contagion(beta=2.0, f0=f0, n_frames=150, seed=s)
        last = frames[-30:]
        f_inf_runs.append(np.mean([m.mean() for _, _, _, _, m in last]))
    res3[f0] = (np.mean(f_inf_runs), np.std(f_inf_runs))
    print('  f0=%.3f  f_inf=%.3f +/- %.3f' % (f0, res3[f0][0], res3[f0][1]))


# =============================================================================
# EXP 4: SNAPSHOT GALLERY at beta=5.0
# =============================================================================
print('\nExp 4: Snapshot gallery (beta=5.0, f0=1%%, N=200)')
frames_snap = run_contagion(beta=5.0, f0=0.01, n_frames=300,
                            seed=2, overrides={'N': 200, 'n_iter': 6000})
fs4 = max(1, 6000 // 300)


# =============================================================================
# FIGURES
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Panic contagion: beta sweep (f0=1%%, N=350, %d seeds)' % N_SEEDS, fontsize=11)

ax = axes[0]
ax.errorbar(beta_vals, [res1[b]['f_inf'] for b in beta_vals],
            yerr=[res1[b]['f_inf_std'] for b in beta_vals],
            fmt='o-', color='crimson', lw=2, capsize=4, label='final panic fraction')
ax.set_xlabel('Contagion rate beta')
ax.set_ylabel('Final panic fraction f_inf')
ax.set_title('Outbreak threshold')
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
ax.errorbar(beta_vals, [res1[b]['phi'] for b in beta_vals],
            yerr=[res1[b]['phi_std'] for b in beta_vals],
            fmt='o-', color='steelblue', lw=2, capsize=4, label='Global Phi')
ax.plot(beta_vals, [res1[b]['calm_phi'] for b in beta_vals],
        's--', color='seagreen', lw=2, label='Calm-agent Phi')
ax.set_xlabel('Contagion rate beta')
ax.set_ylabel('Order parameter Phi')
ax.set_title('Coherence vs contagion rate')
ax.set_ylim(0, 1.05)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[2]
colors = {0.5: 'steelblue', 2.0: 'darkorange', 5.0: 'crimson', 20.0: 'purple'}
for beta, d in ts_results.items():
    m = d['f'].mean(0); s = d['f'].std(0)
    ax.plot(t_axis, m, color=colors[beta], lw=2, label='beta=%.1f' % beta)
    ax.fill_between(t_axis, m-s, m+s, color=colors[beta], alpha=0.15)
ax.set_xlabel('Time')
ax.set_ylabel('Panicked fraction f(t)')
ax.set_title('Epidemic curves')
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/contagion_1_sweep.png', dpi=120)
plt.close()
print('  --> figures/contagion_1_sweep.png')


# Calm-Phi time series and seed sensitivity
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
for beta, d in ts_results.items():
    m = d['calm'].mean(0); s = d['calm'].std(0)
    ax.plot(t_axis, m, color=colors[beta], lw=2, label='beta=%.1f' % beta)
    ax.fill_between(t_axis, m-s, m+s, color=colors[beta], alpha=0.15)
ax.set_xlabel('Time')
ax.set_ylabel('Calm-agent Phi')
ax.set_title('Does the calm sub-flock survive the outbreak?')
ax.set_ylim(0, 1.05)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
f0_arr  = np.array(f0_vals)
mean_arr = np.array([res3[f][0] for f in f0_vals])
std_arr  = np.array([res3[f][1] for f in f0_vals])
ax.errorbar(f0_arr*100, mean_arr, yerr=std_arr,
            fmt='o-', color='darkorange', lw=2, capsize=4)
ax.set_xlabel('Initial panic fraction f0 (%)')
ax.set_ylabel('Final panic fraction f_inf')
ax.set_title('Seed-size sensitivity at beta=2.0')
ax.set_ylim(-0.05, 1.05)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/contagion_2_calm_seed.png', dpi=120)
plt.close()
print('  --> figures/contagion_2_calm_seed.png')


# Snapshot gallery
fig, axes = plt.subplots(2, 4, figsize=(15, 7))
fig.suptitle('Contagion snapshots  (N=200, f0=1%%, beta=5.0; calm=blue, panic=red)',
             fontsize=11)
snap_times = [0., 5., 10., 15., 25., 35., 45., 58.]
for ax, ts in zip(axes.flat, snap_times):
    fi = min(int(ts / (fs4 * BASE['dt'])), len(frames_snap)-1)
    px, py, vx, vy, m = frames_snap[fi]
    sp = np.sqrt(vx**2 + vy**2); sp[sp==0] = 1.
    calm = ~m
    ax.scatter(px[calm], py[calm], s=5, color='steelblue', zorder=3)
    ax.scatter(px[m], py[m], s=12, color='crimson', zorder=5)
    ax.quiver(px[calm], py[calm],
              vx[calm]/sp[calm], vy[calm]/sp[calm],
              scale=80, width=0.003, color='steelblue', alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    f_now = m.mean()
    ax.set_title('t=%.0f  f=%.2f' % (fi*fs4*BASE['dt'], f_now), fontsize=8)
plt.tight_layout()
plt.savefig('figures/contagion_3_snapshots.png', dpi=120)
plt.close()
print('  --> figures/contagion_3_snapshots.png')

print('\nPanic-contagion analysis complete.')
