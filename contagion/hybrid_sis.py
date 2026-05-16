# hybrid_sis.py -- Encirclement + sub-threshold SIS contagion
#
# Finding 25 established a clean SIS epidemic threshold at beta/gamma ~ 1.
# Finding 16 established that encirclement compresses the flock into a small
# number of denser sub-clusters.  Spatial compression should raise the local
# contact count <k>, which scales the effective contagion rate.
#
# Hypothesis: in a parameter regime where contagion alone is sub-threshold
# (no outbreak), adding encircling predators could push the system over the
# threshold by raising effective <k>.
#
# Method
# ------
# Pick (beta, gamma) just on the safe side of the epidemic threshold so that
# contagion-only fizzles (f_ss ~ 0).  Then add 6 encircling predators and ask
# whether the same (beta, gamma) now produces an outbreak.
#
# Conditions:
#   none      : no stressor             (baseline coherence)
#   sis_only  : sub-threshold contagion (should fizzle)
#   encircle  : encirclement only       (Phi ~ 0.7)
#   both      : sub-threshold contagion + encirclement
#
# Metrics: steady-state Phi, f_ss, calm_Phi, max f(t) reached.

import os
import numpy as np
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from flocking import buffer as buf_fn, order_parameter
from model import _periodic_com, _periodic_disp

os.makedirs('figures', exist_ok=True)

N_SEEDS = 6

BASE = dict(
    N=350, r0=0.005, eps=0.1, rf=0.1,
    alpha=1.0, v0=0.02, mu=10.0, ramp=0.1, dt=0.01,
    n_iter=6000,    # longer so contagion has time to either spread or die
)
PANIC_ALPHA = 0.1
PANIC_RAMP  = 10.0
R_CONT      = 0.05

# SIS parameters: chosen sub-threshold so contagion-only fizzles.
# Finding 25 showed beta_c ~ gamma; pick beta below gamma.
BETA  = 1.0
GAMMA = 3.0       # beta/gamma = 0.33  -- well below threshold

PRED = dict(v0=0.05, mu=10.0, alpha=5.0, r0=0.1, eps=2.0, ramp=1.0,
            enc_radius=0.15)


def run(condition='none', n_frames=200, seed=None):
    if seed is not None:
        np.random.seed(seed)

    use_encircle = condition in ('encircle', 'both')
    use_contag   = condition in ('sis_only', 'both')

    p = BASE.copy()
    N      = p['N']
    dt     = p['dt']
    n_iter = p['n_iter']
    r0, eps, rf = p['r0'], p['eps'], p['rf']
    v0, mu      = p['v0'], p['mu']
    frame_every = max(1, n_iter // n_frames)

    x  = np.zeros(2*N)
    x[:N] = np.random.uniform(0., 1., N)
    x[N:] = np.random.uniform(0., 1., N)
    vx = np.random.uniform(-1., 1., N) * v0
    vy = np.random.uniform(-1., 1., N) * v0

    is_panicked = np.zeros(N, dtype=bool)
    if use_contag:
        n0 = max(1, round(0.05 * N))
        idx0 = np.random.choice(N, size=n0, replace=False)
        is_panicked[idx0] = True

    if use_encircle:
        n_pred = 6
        pred_x  = np.random.uniform(0., 1., n_pred)
        pred_y  = np.random.uniform(0., 1., n_pred)
        pred_vx = np.random.uniform(-1., 1., n_pred) * PRED['v0']
        pred_vy = np.random.uniform(-1., 1., n_pred) * PRED['v0']
        pred_angles = np.radians(np.arange(n_pred) * 60.0)
    else:
        n_pred = 0

    p_recover_per_step = 1. - np.exp(-GAMMA * dt)
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
        base_r   = np.where(rep_mask, 1. - d_safe/(2.*r0), 0.)
        strength = np.where(rep_mask, eps * base_r**1.5 / d_safe, 0.)
        repx = (-strength * dx).sum(axis=1)
        repy = (-strength * dy).sum(axis=1)

        vnorm  = np.sqrt(vx**2 + vy**2)
        vnorms = np.where(vnorm == 0, 1., vnorm)
        fpropx = mu * (v0 - vnorm) * vx / vnorms
        fpropy = mu * (v0 - vnorm) * vy / vnorms

        frandx = ramp_arr * np.random.uniform(-1., 1., N)
        frandy = ramp_arr * np.random.uniform(-1., 1., N)

        fx_total = flockx + repx + fpropx + frandx
        fy_total = flocky + repy + fpropy + frandy

        if use_encircle:
            for k in range(n_pred):
                ddx = _periodic_disp(pred_x[k], x[:N])
                ddy = _periodic_disp(pred_y[k], x[N:])
                d = np.sqrt(ddx**2 + ddy**2)
                mask_p = (d > 0) & (d <= PRED['r0'])
                if mask_p.any():
                    strength_p = PRED['eps'] * (1. - d[mask_p]/PRED['r0'])**1.5 / d[mask_p]
                    fx_total[mask_p] -= strength_p * ddx[mask_p]
                    fy_total[mask_p] -= strength_p * ddy[mask_p]

        vx += fx_total * dt
        vy += fy_total * dt
        x[:N] = (x[:N] + vx*dt) % 1.
        x[N:] = (x[N:] + vy*dt) % 1.

        if use_encircle:
            cx, cy = _periodic_com(x[:N]), _periodic_com(x[N:])
            for k in range(n_pred):
                tx = (cx + PRED['enc_radius'] * np.cos(pred_angles[k])) % 1.
                ty = (cy + PRED['enc_radius'] * np.sin(pred_angles[k])) % 1.
                ddx = _periodic_disp(tx, pred_x[k])
                ddy = _periodic_disp(ty, pred_y[k])
                dist = np.sqrt(ddx**2 + ddy**2)
                if dist > 0:
                    ddx /= dist; ddy /= dist
                sp_p = np.sqrt(pred_vx[k]**2 + pred_vy[k]**2)
                if sp_p > 0:
                    pfx = PRED['alpha']*ddx + PRED['mu']*(PRED['v0']-sp_p)*pred_vx[k]/sp_p
                    pfy = PRED['alpha']*ddy + PRED['mu']*(PRED['v0']-sp_p)*pred_vy[k]/sp_p
                else:
                    pfx = PRED['alpha']*ddx; pfy = PRED['alpha']*ddy
                pfx += PRED['ramp']*np.random.uniform(-1., 1.)
                pfy += PRED['ramp']*np.random.uniform(-1., 1.)
                pred_vx[k] += pfx*dt; pred_vy[k] += pfy*dt
                pred_x[k] = (pred_x[k] + pred_vx[k]*dt) % 1.
                pred_y[k] = (pred_y[k] + pred_vy[k]*dt) % 1.

        # Contagion + recovery
        if use_contag and is_panicked.any() and (~is_panicked).any():
            real_dx = x[:N][np.newaxis, :] - x[:N][:, np.newaxis]
            real_dy = x[N:][np.newaxis, :] - x[N:][:, np.newaxis]
            real_dx -= np.round(real_dx); real_dy -= np.round(real_dy)
            rd2 = real_dx**2 + real_dy**2
            within = (rd2 <= R_CONT**2) & (rd2 > 0)
            k_arr = within @ is_panicked.astype(np.int32)
            calm_idx = np.where(~is_panicked)[0]
            if calm_idx.size:
                k_calm = k_arr[calm_idx]
                p_trans = 1. - np.exp(-BETA * k_calm * dt)
                r = np.random.uniform(0., 1., calm_idx.size)
                flipped = calm_idx[r < p_trans]
                if flipped.size:
                    is_panicked[flipped] = True
        if use_contag and is_panicked.any():
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


# Local density measurement for diagnostic
def mean_local_density(px, py, r=R_CONT):
    dx = px[np.newaxis, :] - px[:, np.newaxis]
    dy = py[np.newaxis, :] - py[:, np.newaxis]
    dx -= np.round(dx); dy -= np.round(dy)
    d2 = dx**2 + dy**2
    k = ((d2 <= r**2) & (d2 > 0)).sum(axis=1)
    return float(k.mean())


print('Hybrid SIS+encirclement experiment')
print('  beta=%.2f  gamma=%.2f  (beta/gamma=%.2f -- sub-threshold)' %
      (BETA, GAMMA, BETA/GAMMA))
print('  %d seeds' % N_SEEDS)

conds = ['none', 'sis_only', 'encircle', 'both']
results = {c: [] for c in conds}
for c in conds:
    print('Condition: %s' % c)
    for s in range(N_SEEDS):
        results[c].append(run(condition=c, seed=s))


# Summarize: steady state and peak panic
def summarize(frames):
    last = frames[-30:]
    phi_ss   = np.mean([order_parameter(vx, vy) for _, _, vx, vy, _ in last])
    f_ss     = np.mean([m.mean() for _, _, _, _, m in last])
    cphi=[]
    for _, _, vx, vy, m in last:
        calm = ~m
        if calm.sum() > 5:
            cphi.append(order_parameter(vx[calm], vy[calm]))
    calm_phi = np.mean(cphi) if cphi else float('nan')
    f_max = max(m.mean() for _, _, _, _, m in frames)
    k_t = [mean_local_density(px, py) for px, py, _, _, _ in frames[-30:]]
    k_ss = np.mean(k_t)
    return phi_ss, f_ss, calm_phi, f_max, k_ss


print('\n=== Summary (mean over %d seeds, last 30 frames) ===' % N_SEEDS)
summary = {}
for c in conds:
    phi_runs=[]; f_runs=[]; calm_runs=[]; fmax_runs=[]; k_runs=[]
    for frames in results[c]:
        phi, f, cphi, fmax, k = summarize(frames)
        phi_runs.append(phi); f_runs.append(f)
        calm_runs.append(cphi); fmax_runs.append(fmax); k_runs.append(k)
    summary[c] = dict(
        phi=np.mean(phi_runs), phi_std=np.std(phi_runs),
        f=np.mean(f_runs), f_std=np.std(f_runs),
        calm=np.nanmean(calm_runs),
        f_max=np.mean(fmax_runs),
        k=np.mean(k_runs),
    )
    print('  %-10s Phi=%.3f  f_ss=%.3f +/- %.3f  f_max=%.3f  calm_Phi=%.3f  <k>=%.2f' % (
        c, summary[c]['phi'], summary[c]['f'], summary[c]['f_std'],
        summary[c]['f_max'], summary[c]['calm'], summary[c]['k']))


# Figure
dt_val = BASE['dt']; n_iter = BASE['n_iter']
fs_step = max(1, n_iter // 200)
t_axis = np.arange(200) * fs_step * dt_val

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Sub-threshold SIS + encirclement (beta=%.1f, gamma=%.1f, beta/gamma=%.2f; %d seeds)'
             % (BETA, GAMMA, BETA/GAMMA, N_SEEDS), fontsize=11)
colors = {'none':'steelblue', 'sis_only':'crimson',
          'encircle':'darkorange', 'both':'purple'}

ax = axes[0]
for c in conds:
    f_ts = np.array([[m.mean() for _, _, _, _, m in frames]
                     for frames in results[c]])
    mean_f, std_f = f_ts.mean(0), f_ts.std(0)
    ax.plot(t_axis[:len(mean_f)], mean_f, color=colors[c], lw=2, label=c)
    ax.fill_between(t_axis[:len(mean_f)], mean_f-std_f, mean_f+std_f, color=colors[c], alpha=0.15)
ax.set_xlabel('Time'); ax.set_ylabel('Panic fraction f(t)')
ax.set_ylim(-0.05, 1.05); ax.set_title('Panic propagation')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1]
for c in conds:
    phi_ts = np.array([[order_parameter(vx, vy) for _, _, vx, vy, _ in frames]
                       for frames in results[c]])
    m, s = phi_ts.mean(0), phi_ts.std(0)
    ax.plot(t_axis[:len(m)], m, color=colors[c], lw=2, label=c)
    ax.fill_between(t_axis[:len(m)], m-s, m+s, color=colors[c], alpha=0.15)
ax.set_xlabel('Time'); ax.set_ylabel('Global Phi')
ax.set_ylim(0, 1.05); ax.set_title('Global coherence')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[2]
xs = np.arange(len(conds))
phis = [summary[c]['phi'] for c in conds]
fs   = [summary[c]['f']   for c in conds]
ks   = [summary[c]['k']   for c in conds]
ax.bar(xs-0.27, phis, 0.25, color='steelblue', label='Phi')
ax.bar(xs,      fs,   0.25, color='crimson',  label='f_ss')
ax2 = ax.twinx()
ax2.bar(xs+0.27, ks, 0.25, color='gray', label='<k>')
ax2.set_ylabel('mean local contact count <k>', color='gray')
ax.set_xticks(xs); ax.set_xticklabels(conds)
ax.set_ylim(0, 1.05); ax.set_ylabel('Phi / f_ss')
ax.set_title('Steady-state summary')
ax.legend(fontsize=8, loc='upper left'); ax2.legend(fontsize=8, loc='upper right')

plt.tight_layout()
plt.savefig('figures/hybrid_sis_1_summary.png', dpi=120)
plt.close()
print('  --> figures/hybrid_sis_1_summary.png')

print('\nHybrid SIS analysis complete.')
