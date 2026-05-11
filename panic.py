# panic.py -- Effect of panicked agents on a calm flocking crowd
#
# Based on Charbonneau Ch. 10 Section 10.5 (Why You Should Never Panic).
# Two agent types share the same space:
#   Calm agents:    strong flocking (alpha=1.0), low noise (ramp=0.5), v0=1.0
#   Panicked agents: weak flocking (alpha=0.1), high noise (ramp=10.0), v0=1.0
#
# Panicked agents don't chase a predator -- they just move erratically and
# collide with calm agents, carving holes and deflecting the flock.
#
# Experiments:
#   1. Panic fraction sweep (f=0,1,2,5,10,20%) -- flock deflection and coherence
#   2. Time series: how quickly does panic propagate? (f=2% and f=10%)
#   3. Snapshot gallery showing spatial "holes" carved by panicked agents
#   4. Comparison with predator: does a predator cause more or less disruption
#      than an equivalent fraction of panicked agents?

import os
import numpy as np
import matplotlib.pyplot as plt
from flocking import params, buffer, force, order_parameter
from predator import PREY_DEFAULT, PRED_DEFAULT

os.makedirs('figures', exist_ok=True)
N_SEEDS = 8

CALM_PARAMS = dict(
    N=350, r0=0.005, eps=0.1, rf=0.1,
    alpha=1.0, v0=1.0, mu=10.0, ramp=0.5, dt=0.01,
    n_iter=5000
)
PANIC_ALPHA = 0.1
PANIC_RAMP  = 10.0


def run_panic(f_panic=0.02, calm_overrides=None, n_frames=250, seed=None):
    """
    Run mixed calm+panicked population.
    f_panic = fraction of panicked agents (0..1).
    Returns frames of (px, py, vx, vy, is_panicked_mask).

    Efficient single-pass: geometry computed once per step, per-agent alpha
    and ramp applied directly without calling force() twice.
    """
    if seed is not None:
        np.random.seed(seed)

    cp = CALM_PARAMS.copy()
    if calm_overrides:
        cp.update(calm_overrides)

    N      = cp['N']
    dt     = cp['dt']
    n_iter = cp['n_iter']
    r0, eps, rf = cp['r0'], cp['eps'], cp['rf']
    v0, mu       = cp['v0'], cp['mu']
    frame_every  = max(1, n_iter // n_frames)

    N_panic     = max(0, round(f_panic * N))
    N_calm      = N - N_panic
    is_panicked = np.array([False]*N_calm + [True]*N_panic)
    alpha_arr   = np.where(is_panicked, PANIC_ALPHA, cp['alpha'])
    ramp_arr    = np.where(is_panicked, PANIC_RAMP,  cp['ramp'])

    x  = np.zeros(2*N)
    x[:N] = np.random.uniform(0., 1., N)
    x[N:] = np.random.uniform(0., 1., N)
    vx = np.random.uniform(-1., 1., N) * v0
    vy = np.random.uniform(-1., 1., N) * v0

    from flocking import buffer as buf_fn
    rb = max(r0, rf)
    frames = []

    for i in range(n_iter):
        nb, xb, yb, vxb, vyb = buf_fn(rb, x, vx, vy, N)

        # geometry: (N, nb) distance matrices
        dx = xb[:nb][np.newaxis, :] - x[:N][:, np.newaxis]
        dy = yb[:nb][np.newaxis, :] - x[N:][:, np.newaxis]
        d2 = dx**2 + dy**2
        not_self = np.ones((N, nb), dtype=bool)
        idx = np.arange(min(N, nb))
        not_self[idx, idx] = False

        # flocking: compute unit direction once, scale by per-agent alpha
        flock_mask = (d2 <= rf**2) & not_self
        flx = np.where(flock_mask, vxb[:nb], 0.).sum(axis=1)
        fly = np.where(flock_mask, vyb[:nb], 0.).sum(axis=1)
        nfl = flock_mask.sum(axis=1)
        nrm = np.sqrt(flx**2 + fly**2)
        nrm[nfl == 0] = 1.
        flockx = alpha_arr * flx / nrm
        flocky = alpha_arr * fly / nrm

        # repulsion: same for all agents
        rep_mask = (d2 <= (2*r0)**2) & not_self & (d2 > 0)
        d_safe   = np.where(rep_mask, np.sqrt(d2), 1.)
        base     = np.where(rep_mask, 1. - d_safe/(2.*r0), 0.)
        strength = np.where(rep_mask, eps * base**1.5 / d_safe, 0.)
        repx = (-strength * dx).sum(axis=1)
        repy = (-strength * dy).sum(axis=1)

        # self-propulsion: same v0, mu for all
        vnorm  = np.sqrt(vx**2 + vy**2)
        vnorms = np.where(vnorm == 0, 1., vnorm)
        fpropx = mu * (v0 - vnorm) * vx / vnorms
        fpropy = mu * (v0 - vnorm) * vy / vnorms

        # noise: per-agent ramp
        frandx = ramp_arr * np.random.uniform(-1., 1., N)
        frandy = ramp_arr * np.random.uniform(-1., 1., N)

        fx = flockx + repx + fpropx + frandx
        fy = flocky + repy + fpropy + frandy

        vx += fx * dt
        vy += fy * dt
        x[:N] = (x[:N] + vx*dt) % 1.
        x[N:] = (x[N:] + vy*dt) % 1.

        if i % frame_every == 0:
            frames.append((x[:N].copy(), x[N:].copy(),
                           vx.copy(), vy.copy(),
                           is_panicked.copy()))

    return frames


def flock_direction(vx, vy):
    """Mean heading angle in degrees."""
    mx = vx.mean(); my = vy.mean()
    return np.degrees(np.arctan2(my, mx))


# =============================================================================
# EXP 1: PANIC FRACTION SWEEP
# =============================================================================
print('Exp 1: Panic fraction sweep  (%d seeds)' % N_SEEDS)
f_vals = [0.00, 0.01, 0.02, 0.05, 0.10, 0.20]

res1 = {}
for f in f_vals:
    phi_ss=[]; dir_ss=[]; calm_phi_ss=[]
    for s in range(N_SEEDS):
        frames = run_panic(f_panic=f, n_frames=250, seed=s)
        phi_t = [order_parameter(vx, vy) for _, _, vx, vy, _ in frames]
        # steady-state: last 20% of frames
        phi_ss.append(np.mean(phi_t[-50:]))

        # mean flock direction at steady state
        dirs = []
        for _, _, vx, vy, _ in frames[-50:]:
            dirs.append(flock_direction(vx, vy))
        dir_ss.append(np.mean(dirs))

        # coherence of calm agents only
        calm_phi_t = []
        for _, _, vx, vy, panic_mask in frames[-50:]:
            calm_mask = ~panic_mask
            if calm_mask.sum() > 0:
                calm_phi_t.append(order_parameter(vx[calm_mask], vy[calm_mask]))
        calm_phi_ss.append(np.mean(calm_phi_t))

    res1[f] = dict(phi=np.mean(phi_ss), phi_std=np.std(phi_ss),
                   calm_phi=np.mean(calm_phi_ss),
                   direction=np.mean(dir_ss))
    print('  f=%.0f%%  Phi=%.3f +/- %.3f  calm_Phi=%.3f  direction=%.1f deg' % (
        f*100, res1[f]['phi'], res1[f]['phi_std'],
        res1[f]['calm_phi'], res1[f]['direction']))


# =============================================================================
# EXP 2: TIME SERIES FOR f=2% AND f=10%
# =============================================================================
print('\nExp 2: Time series  (f=2%% and f=10%%, %d seeds)' % N_SEEDS)
n_it2 = 5000; n_fr2 = 250
ts_results = {}

for f in [0.00, 0.02, 0.10]:
    phi_runs = []
    for s in range(N_SEEDS):
        frames = run_panic(f_panic=f,
                           calm_overrides={'n_iter': n_it2},
                           n_frames=n_fr2, seed=s)
        phi_runs.append([order_parameter(vx, vy) for _, _, vx, vy, _ in frames])
    ts_results[f] = np.array(phi_runs)

dt_val = CALM_PARAMS['dt']
fs2 = max(1, n_it2 // n_fr2)
t2 = np.arange(n_fr2) * fs2 * dt_val


# =============================================================================
# EXP 3: SNAPSHOT GALLERY (f=5%, seed=1)
# =============================================================================
print('\nExp 3: Snapshot gallery  (f=5%%)')
frames_snap = run_panic(f_panic=0.05,
                        calm_overrides={'n_iter': 6000, 'N': 200},
                        n_frames=300, seed=1)
fs3 = max(1, 6000 // 300)


# =============================================================================
# FIGURES
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Panic fraction sweep  (%d seeds)' % N_SEEDS, fontsize=11)

f_pct = [f*100 for f in f_vals]
axes[0].errorbar(f_pct, [res1[f]['phi'] for f in f_vals],
                 yerr=[res1[f]['phi_std'] for f in f_vals],
                 fmt='o-', color='steelblue', lw=2, capsize=5, label='Global Phi')
axes[0].plot(f_pct, [res1[f]['calm_phi'] for f in f_vals],
             's--', color='seagreen', lw=2, label='Calm agents only')
axes[0].set_xlabel('Panic fraction (%)'); axes[0].set_ylabel('Order parameter Phi')
axes[0].set_title('Flock coherence vs panic fraction'); axes[0].set_ylim(0,1)
axes[0].legend(fontsize=8)

# direction deflection relative to f=0 baseline
base_dir = res1[0.00]['direction']
deflections = [abs(res1[f]['direction'] - base_dir) for f in f_vals]
axes[1].plot(f_pct, deflections, 'o-', color='crimson', lw=2)
axes[1].set_xlabel('Panic fraction (%)'); axes[1].set_ylabel('Direction deflection (deg)')
axes[1].set_title('Flock deflection angle vs panic fraction')

# time series
colors_ts = {0.00: 'steelblue', 0.02: 'darkorange', 0.10: 'crimson'}
labels_ts = {0.00: '0% panic', 0.02: '2% panic', 0.10: '10% panic'}
for f, arr in ts_results.items():
    axes[2].plot(t2, arr.mean(0), color=colors_ts[f], lw=2, label=labels_ts[f])
    axes[2].fill_between(t2, arr.mean(0)-arr.std(0), arr.mean(0)+arr.std(0),
                         color=colors_ts[f], alpha=0.15)
axes[2].set_xlabel('Time'); axes[2].set_ylabel('Order parameter Phi')
axes[2].set_title('Coherence over time'); axes[2].set_ylim(0,1)
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig('figures/panic_1_sweep.png', dpi=120)
plt.close()
print('  --> figures/panic_1_sweep.png')

# Snapshot gallery
snap_times = [0., 5., 10., 20., 30., 40., 50., 59.]
fig, axes = plt.subplots(2, 4, figsize=(15, 7))
fig.suptitle('Panic snapshots  (N=200, f=5%%, calm=blue, panic=red)', fontsize=11)
for ax, ts in zip(axes.flat, snap_times):
    fi = min(int(ts / (fs3 * CALM_PARAMS['dt'])), len(frames_snap)-1)
    px, py, vx, vy, panic_mask = frames_snap[fi]
    calm_mask = ~panic_mask
    sp = np.sqrt(vx**2+vy**2); sp[sp==0]=1.
    ax.scatter(px[calm_mask], py[calm_mask], s=4, color='steelblue', zorder=3)
    ax.scatter(px[panic_mask], py[panic_mask], s=10, color='crimson', zorder=5)
    ax.quiver(px[calm_mask], py[calm_mask],
              vx[calm_mask]/sp[calm_mask], vy[calm_mask]/sp[calm_mask],
              scale=80, width=0.003, color='steelblue', alpha=0.3)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    phi_val = order_parameter(vx, vy)
    ax.set_title('t=%.0f  Phi=%.2f' % (fi*fs3*CALM_PARAMS['dt'], phi_val), fontsize=8)

plt.tight_layout()
plt.savefig('figures/panic_2_snapshots.png', dpi=120)
plt.close()
print('  --> figures/panic_2_snapshots.png')

print('\nPanic analysis complete.')
