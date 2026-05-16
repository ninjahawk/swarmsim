# hybrid_stressors.py -- Combined predation + panic contagion
#
# Findings 14-16 show that encirclement (n_pred=6) divides the flock into
# coherent sub-flocks (Phi=0.77 globally, but largest_phi=0.997 within each).
# Finding 20 shows that contagious panic at any beta>0 dissolves the flock
# globally (Phi ~0.10, calm pool drained).
#
# Two distinct disruption modes:
#   - Encirclement: spatial herding + directional split.  Sub-flocks survive.
#   - Contagion:    internal alignment collapse.  No sub-flocks survive.
#
# Question: how do they combine?  Three possibilities:
#   A. Independent: outcome ~ worse of the two single-stressor results.
#   B. Linear interference: encirclement-induced separation breaks contagion
#      contacts, slowing the outbreak.  Net Phi BETTER than contagion alone.
#   C. Super-linear: predators concentrate prey into sub-flocks, increasing
#      local density and accelerating contact rates.  Net Phi WORSE than either.
#
# This experiment runs four conditions side-by-side with matched seeds:
#   (1) no stressor
#   (2) encirclement only       (n_pred=6, R_enc=0.15)
#   (3) contagion only          (beta=0.5, f0=1%)
#   (4) both                    (encircle + contagion)
# and reports steady-state Phi, panic fraction, calm-Phi, and time-to-saturate.

import os
import numpy as np
import matplotlib.pyplot as plt
from flocking import buffer as buf_fn, order_parameter
from model import _periodic_com, _periodic_disp

os.makedirs('figures', exist_ok=True)

N_SEEDS = 6

BASE = dict(
    N=350, r0=0.005, eps=0.1, rf=0.1,
    alpha=1.0, v0=0.02, mu=10.0, ramp=0.1, dt=0.01,
    n_iter=5000,
)
# Slow-prey regime so v0=0.05 predators can actually pursue (matches
# encirclement.py / Findings 14-16).  Contagion still spreads -- contact
# patterns are determined by spatial proximity, not absolute speed.
PANIC_ALPHA = 0.1
PANIC_RAMP  = 10.0
R_CONT      = 0.05

# Predator parameters (match encirclement.py / Finding 16 setup)
PRED = dict(v0=0.05, mu=10.0, alpha=5.0, r0=0.1, eps=2.0, ramp=1.0,
            enc_radius=0.15)


def run(condition='none', n_frames=200, seed=None):
    """
    condition in {'none','encircle','contagion','both'}
    Returns frames: (px, py, vx, vy, is_panicked, [pred_x, pred_y, ...])
    """
    if seed is not None:
        np.random.seed(seed)

    use_encircle = condition in ('encircle', 'both')
    use_contag   = condition in ('contagion', 'both')

    p = BASE.copy()
    N      = p['N']
    dt     = p['dt']
    n_iter = p['n_iter']
    r0, eps, rf = p['r0'], p['eps'], p['rf']
    v0, mu      = p['v0'], p['mu']
    frame_every = max(1, n_iter // n_frames)

    # Prey state
    x  = np.zeros(2*N)
    x[:N] = np.random.uniform(0., 1., N)
    x[N:] = np.random.uniform(0., 1., N)
    vx = np.random.uniform(-1., 1., N) * v0
    vy = np.random.uniform(-1., 1., N) * v0

    is_panicked = np.zeros(N, dtype=bool)
    if use_contag:
        n0 = max(1, round(0.01 * N))
        idx0 = np.random.choice(N, size=n0, replace=False)
        is_panicked[idx0] = True

    # Predators (6 encirclers at angles k*60)
    if use_encircle:
        n_pred = 6
        pred_x  = np.random.uniform(0., 1., n_pred)
        pred_y  = np.random.uniform(0., 1., n_pred)
        pred_vx = np.random.uniform(-1., 1., n_pred) * PRED['v0']
        pred_vy = np.random.uniform(-1., 1., n_pred) * PRED['v0']
        pred_angles = np.radians(np.arange(n_pred) * 60.0)
    else:
        n_pred = 0

    rb = max(r0, rf, R_CONT)
    beta = 0.5
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

        frandx = ramp_arr * np.random.uniform(-1., 1., N)
        frandy = ramp_arr * np.random.uniform(-1., 1., N)

        fx_total = flockx + repx + fpropx + frandx
        fy_total = flocky + repy + fpropy + frandy

        # Predator force on prey
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

        # Update predators
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

        # Contagion
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
                p_trans = 1. - np.exp(-beta * k_calm * dt)
                r = np.random.uniform(0., 1., calm_idx.size)
                flipped = calm_idx[r < p_trans]
                if flipped.size:
                    is_panicked[flipped] = True

        if i % frame_every == 0:
            preds_snap = (pred_x.copy(), pred_y.copy()) if use_encircle else (None, None)
            frames.append((x[:N].copy(), x[N:].copy(),
                           vx.copy(), vy.copy(),
                           is_panicked.copy(), preds_snap))

    return frames


# Run all four conditions
conds = ['none', 'encircle', 'contagion', 'both']
all_results = {c: [] for c in conds}

for c in conds:
    print('Condition: %s' % c)
    for s in range(N_SEEDS):
        frames = run(condition=c, seed=s)
        all_results[c].append(frames)
    print('  done')


# Summarize
print('\n=== Summary (mean over %d seeds, last 30 frames) ===' % N_SEEDS)
summary = {}
for c in conds:
    phi_runs=[]; f_runs=[]; calm_runs=[]
    for frames in all_results[c]:
        last = frames[-30:]
        phi_runs.append(np.mean([order_parameter(vx, vy) for _, _, vx, vy, _, _ in last]))
        f_runs.append(np.mean([m.mean() for _, _, _, _, m, _ in last]))
        cphi = []
        for _, _, vx, vy, m, _ in last:
            calm = ~m
            if calm.sum() > 5:
                cphi.append(order_parameter(vx[calm], vy[calm]))
        calm_runs.append(np.mean(cphi) if cphi else float('nan'))
    summary[c] = dict(phi=np.mean(phi_runs), phi_std=np.std(phi_runs),
                      f=np.mean(f_runs), calm=np.nanmean(calm_runs))
    print('  %-10s Phi=%.3f +/- %.3f  f=%.3f  calm_Phi=%.3f' % (
        c, summary[c]['phi'], summary[c]['phi_std'],
        summary[c]['f'], summary[c]['calm']))


# Time series figure
dt_val = BASE['dt']; n_iter = BASE['n_iter']
fs_step = max(1, n_iter // 200)
t_axis = np.arange(200) * fs_step * dt_val

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle('Hybrid stressors: encirclement vs contagion vs both (%d seeds each)' % N_SEEDS,
             fontsize=11)
colors = {'none':'steelblue', 'encircle':'darkorange',
          'contagion':'crimson', 'both':'purple'}

ax = axes[0, 0]
for c in conds:
    phi_ts = np.array([[order_parameter(vx, vy) for _, _, vx, vy, _, _ in frames]
                       for frames in all_results[c]])
    m, s = phi_ts.mean(0), phi_ts.std(0)
    ax.plot(t_axis[:len(m)], m, color=colors[c], lw=2, label=c)
    ax.fill_between(t_axis[:len(m)], m-s, m+s, color=colors[c], alpha=0.15)
ax.set_xlabel('Time'); ax.set_ylabel('Global Phi')
ax.set_ylim(0, 1.05); ax.set_title('Global coherence')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[0, 1]
for c in conds:
    f_ts = np.array([[m.mean() for _, _, _, _, m, _ in frames]
                     for frames in all_results[c]])
    mean_f, std_f = f_ts.mean(0), f_ts.std(0)
    ax.plot(t_axis[:len(mean_f)], mean_f, color=colors[c], lw=2, label=c)
    ax.fill_between(t_axis[:len(mean_f)], mean_f-std_f, mean_f+std_f, color=colors[c], alpha=0.15)
ax.set_xlabel('Time'); ax.set_ylabel('Panic fraction')
ax.set_ylim(-0.05, 1.05); ax.set_title('Panic propagation')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1, 0]
for c in conds:
    cphi_ts = []
    for frames in all_results[c]:
        row = []
        for _, _, vx, vy, m, _ in frames:
            calm = ~m
            row.append(order_parameter(vx[calm], vy[calm]) if calm.sum() > 5 else 1.0)
        cphi_ts.append(row)
    cphi_ts = np.array(cphi_ts)
    m, s = cphi_ts.mean(0), cphi_ts.std(0)
    ax.plot(t_axis[:len(m)], m, color=colors[c], lw=2, label=c)
    ax.fill_between(t_axis[:len(m)], m-s, m+s, color=colors[c], alpha=0.15)
ax.set_xlabel('Time'); ax.set_ylabel('Calm-agent Phi')
ax.set_ylim(0, 1.05); ax.set_title('Calm sub-flock coherence')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1, 1]
xs = np.arange(len(conds))
phis = [summary[c]['phi'] for c in conds]
phis_std = [summary[c]['phi_std'] for c in conds]
fs    = [summary[c]['f']   for c in conds]
cphi  = [summary[c]['calm'] for c in conds]
ax.bar(xs-0.27, phis, 0.25, yerr=phis_std, color='steelblue', label='Global Phi', capsize=4)
ax.bar(xs,      cphi, 0.25, color='seagreen', label='Calm-agent Phi')
ax.bar(xs+0.27, fs,   0.25, color='crimson',  label='Panic fraction')
ax.set_xticks(xs); ax.set_xticklabels(conds)
ax.set_ylim(0, 1.05); ax.set_ylabel('value')
ax.set_title('Steady-state summary')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/hybrid_1_summary.png', dpi=120)
plt.close()
print('  --> figures/hybrid_1_summary.png')

print('\nHybrid-stressor analysis complete.')
