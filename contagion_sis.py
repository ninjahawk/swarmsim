# contagion_sis.py -- Panic contagion with recovery (SIS dynamics)
#
# Finding 20 showed that in the no-recovery SI model, any contagion rate beta > 0
# leads to total saturation (f_inf=1).  The absorbing state of panic is the cause.
# This script adds a recovery rate gamma: panicked agents return to calm at rate
# gamma per timestep.  Now the question is whether there is a real epidemic
# threshold beta_c(gamma).
#
# Standard SIS result (mean-field, well-mixed):  outbreak iff beta * <k> / gamma > 1.
# Here <k> ~ density * pi * r_cont^2 * N = 1 * pi * 0.0025 * 350 ~ 2.7 neighbors.
# So threshold predicted at beta/gamma ~ 1/2.7 ~ 0.37.
#
# Experiments:
#   1. Sweep beta at fixed gamma=1.0 (one recovery per time unit, matches dt=0.01 -> p=0.01 per step)
#   2. Sweep gamma at fixed beta=2.0
#   3. 2D phase diagram beta x gamma at low resolution
#   4. Time series at sub/super-threshold to show stable endemic state vs die-out

import os
import numpy as np
import matplotlib.pyplot as plt
from flocking import buffer as buf_fn, order_parameter

os.makedirs('figures', exist_ok=True)

N_SEEDS = 5

BASE = dict(
    N=350, r0=0.005, eps=0.1, rf=0.1,
    alpha=1.0, v0=1.0, mu=10.0, ramp=0.5, dt=0.01,
    n_iter=4000,
)
PANIC_ALPHA = 0.1
PANIC_RAMP  = 10.0
R_CONT      = 0.05


def run_sis(beta=2.0, gamma=1.0, f0=0.05, n_frames=200, seed=None, overrides=None):
    """SIS contagion: calm -> panic at rate beta*k, panic -> calm at rate gamma."""
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

    p_recover_per_step = 1. - np.exp(-gamma * dt)

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

        frandx = ramp_arr * np.random.uniform(-1., 1., N)
        frandy = ramp_arr * np.random.uniform(-1., 1., N)

        fx = flockx + repx + fpropx + frandx
        fy = flocky + repy + fpropy + frandy

        vx += fx * dt
        vy += fy * dt
        x[:N] = (x[:N] + vx*dt) % 1.
        x[N:] = (x[N:] + vy*dt) % 1.

        # contagion + recovery
        if is_panicked.any() and (~is_panicked).any():
            real_dx = x[:N][np.newaxis, :] - x[:N][:, np.newaxis]
            real_dy = x[N:][np.newaxis, :] - x[N:][:, np.newaxis]
            real_dx -= np.round(real_dx); real_dy -= np.round(real_dy)
            rd2 = real_dx**2 + real_dy**2
            within = (rd2 <= R_CONT**2) & (rd2 > 0)
            k = within @ is_panicked.astype(np.int32)
            # infection
            calm_idx = np.where(~is_panicked)[0]
            if calm_idx.size and beta > 0:
                k_calm = k[calm_idx]
                p_trans = 1. - np.exp(-beta * k_calm * dt)
                r = np.random.uniform(0., 1., calm_idx.size)
                flipped = calm_idx[r < p_trans]
                if flipped.size:
                    is_panicked[flipped] = True

        # recovery (acts independent of neighborhood)
        if is_panicked.any() and gamma > 0:
            panic_idx = np.where(is_panicked)[0]
            r = np.random.uniform(0., 1., panic_idx.size)
            recovered = panic_idx[r < p_recover_per_step]
            if recovered.size:
                is_panicked[recovered] = False

        if i % frame_every == 0:
            frames.append((x[:N].copy(), x[N:].copy(),
                           vx.copy(), vy.copy(),
                           is_panicked.copy()))

    return frames


def summarize(frames, ss_tail=30):
    last = frames[-ss_tail:]
    f_ss = np.mean([m.mean() for _, _, _, _, m in last])
    phi_ss = np.mean([order_parameter(vx, vy) for _, _, vx, vy, _ in last])
    calm_phi = []
    for _, _, vx, vy, m in last:
        calm = ~m
        if calm.sum() > 5:
            calm_phi.append(order_parameter(vx[calm], vy[calm]))
    calm_phi_ss = np.mean(calm_phi) if calm_phi else float('nan')
    return f_ss, phi_ss, calm_phi_ss


# =============================================================================
# EXP 1: BETA SWEEP at fixed gamma=1.0
# =============================================================================
print('Exp 1: beta sweep at gamma=1.0 (%d seeds, f0=5%%)' % N_SEEDS)
beta_vals = [0.0, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
gamma_fix = 1.0
res1 = {}
for beta in beta_vals:
    fs=[]; phis=[]; cphis=[]
    for s in range(N_SEEDS):
        frames = run_sis(beta=beta, gamma=gamma_fix, f0=0.05, seed=s)
        f_ss, phi_ss, c_ss = summarize(frames)
        fs.append(f_ss); phis.append(phi_ss); cphis.append(c_ss)
    res1[beta] = dict(f=np.mean(fs), f_std=np.std(fs),
                      phi=np.mean(phis), calm=np.nanmean(cphis))
    print('  beta=%4.1f  f_ss=%.3f +/- %.3f  Phi=%.3f  calm_Phi=%.3f' % (
        beta, res1[beta]['f'], res1[beta]['f_std'],
        res1[beta]['phi'], res1[beta]['calm']))


# =============================================================================
# EXP 2: GAMMA SWEEP at fixed beta=2.0
# =============================================================================
print('\nExp 2: gamma sweep at beta=2.0')
gamma_vals = [0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
beta_fix = 2.0
res2 = {}
for gamma in gamma_vals:
    fs=[]; phis=[]; cphis=[]
    for s in range(N_SEEDS):
        frames = run_sis(beta=beta_fix, gamma=gamma, f0=0.05, seed=s)
        f_ss, phi_ss, c_ss = summarize(frames)
        fs.append(f_ss); phis.append(phi_ss); cphis.append(c_ss)
    res2[gamma] = dict(f=np.mean(fs), f_std=np.std(fs),
                       phi=np.mean(phis), calm=np.nanmean(cphis))
    print('  gamma=%4.1f  f_ss=%.3f +/- %.3f  Phi=%.3f  calm_Phi=%.3f' % (
        gamma, res2[gamma]['f'], res2[gamma]['f_std'],
        res2[gamma]['phi'], res2[gamma]['calm']))


# =============================================================================
# EXP 3: 2D PHASE DIAGRAM
# =============================================================================
print('\nExp 3: 2D phase diagram (low res, %d seeds)' % N_SEEDS)
beta_g  = [0.2, 0.5, 1.0, 2.0, 4.0]
gamma_g = [0.3, 1.0, 3.0, 10.0]
phase_f = np.zeros((len(gamma_g), len(beta_g)))
for j, b in enumerate(beta_g):
    for i, g in enumerate(gamma_g):
        fs=[]
        for s in range(N_SEEDS):
            frames = run_sis(beta=b, gamma=g, f0=0.05, seed=s)
            f_ss, _, _ = summarize(frames)
            fs.append(f_ss)
        phase_f[i, j] = np.mean(fs)
        print('  beta=%.1f gamma=%.1f -> f_ss=%.3f' % (b, g, phase_f[i, j]))


# =============================================================================
# FIGURES
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('SIS contagion: epidemic threshold (N=350, f0=5%%, %d seeds)' % N_SEEDS, fontsize=11)

ax = axes[0]
fs   = [res1[b]['f']     for b in beta_vals]
fstd = [res1[b]['f_std'] for b in beta_vals]
phis = [res1[b]['phi']   for b in beta_vals]
cphi = [res1[b]['calm']  for b in beta_vals]
ax.errorbar(beta_vals, fs, yerr=fstd, fmt='o-', color='crimson', lw=2, capsize=4,
            label='steady-state panic frac')
ax.plot(beta_vals, phis, 's--', color='steelblue', lw=2, label='Global Phi')
ax.plot(beta_vals, cphi, '^:', color='seagreen', lw=2, label='Calm-agent Phi')
ax.set_xlabel('Contagion rate beta'); ax.set_ylabel('value')
ax.set_title('Beta sweep at gamma=1.0')
ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1]
fs2  = [res2[g]['f']    for g in gamma_vals]
fs2s = [res2[g]['f_std']for g in gamma_vals]
phi2 = [res2[g]['phi']  for g in gamma_vals]
cph2 = [res2[g]['calm'] for g in gamma_vals]
ax.errorbar(gamma_vals, fs2, yerr=fs2s, fmt='o-', color='crimson', lw=2, capsize=4,
            label='steady-state panic frac')
ax.plot(gamma_vals, phi2, 's--', color='steelblue', lw=2, label='Global Phi')
ax.plot(gamma_vals, cph2, '^:', color='seagreen', lw=2, label='Calm-agent Phi')
ax.set_xlabel('Recovery rate gamma'); ax.set_ylabel('value')
ax.set_title('Gamma sweep at beta=2.0')
ax.set_xscale('log')
ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[2]
im = ax.imshow(phase_f, origin='lower', aspect='auto',
               extent=[beta_g[0], beta_g[-1], gamma_g[0], gamma_g[-1]],
               cmap='RdBu_r', vmin=0, vmax=1)
ax.set_xlabel('beta'); ax.set_ylabel('gamma')
ax.set_title('Phase diagram: f_ss')
plt.colorbar(im, ax=ax, label='f_ss')
# overlay diagonal where beta = gamma
xs = np.linspace(min(beta_g), max(beta_g), 50)
ax.plot(xs, xs, 'k--', lw=1, alpha=0.5, label='beta=gamma')
ax.legend(fontsize=8, loc='lower right')

plt.tight_layout()
plt.savefig('figures/contagion_sis_1_sweeps.png', dpi=120)
plt.close()
print('  --> figures/contagion_sis_1_sweeps.png')

print('\nSIS contagion analysis complete.')
