# multi_predator.py -- What happens with multiple predators?
#
# Does the flock split to confuse multiple predators?
# Does coherence break down faster?
# Is there an optimal number of predators that the flock can handle?

import os
import numpy as np
import matplotlib.pyplot as plt
from flocking import run, params, buffer, force, order_parameter
from predator import PREY_DEFAULT, PRED_DEFAULT
from geometry import gyration_radius, aspect_ratio, geometry_series_pred

os.makedirs('figures', exist_ok=True)
SEED = 7


# =============================================================================
# MULTI-PREDATOR SIMULATION
# =============================================================================
def run_multi_predator(n_pred=2, prey_overrides=None, pred_overrides=None,
                       n_frames=200, seed=None):
    """
    Run with multiple predators. Each predator chases prey CoM independently.
    Returns frames of (prey_px, prey_py, prey_vx, prey_vy,
                       pred_xs, pred_ys) where pred_xs/ys are arrays of length n_pred.
    """
    if seed is not None:
        np.random.seed(seed)

    pp = PREY_DEFAULT.copy()
    if prey_overrides:
        pp.update(prey_overrides)
    pd = PRED_DEFAULT.copy()
    if pred_overrides:
        pd.update(pred_overrides)

    p = params(pp)
    N = p['N']
    dt = p['dt']
    frame_every = max(1, p['n_iter'] // n_frames)

    # initialise prey
    x  = np.zeros(2*N)
    x[:N] = np.random.uniform(0., 1., N)
    x[N:] = np.random.uniform(0., 1., N)
    vx = np.random.uniform(-1., 1., N) * p['v0']
    vy = np.random.uniform(-1., 1., N) * p['v0']

    # initialise predators at random positions
    pred_x  = np.random.uniform(0., 1., n_pred)
    pred_y  = np.random.uniform(0., 1., n_pred)
    pred_vx = np.random.uniform(-1., 1., n_pred) * pd['v0_pred']
    pred_vy = np.random.uniform(-1., 1., n_pred) * pd['v0_pred']

    frames = []

    for i in range(p['n_iter']):
        rb = max(p['r0'], p['rf'])
        nb, xb, yb, vxb, vyb = buffer(rb, x, vx, vy, N)
        fx, fy = force(nb, xb, yb, vxb, vyb, x, vx, vy, p)

        # repulsion from each predator
        for ip in range(n_pred):
            for j in range(N):
                ddx = pred_x[ip] - x[j];  ddx -= round(ddx)
                ddy = pred_y[ip] - x[N+j]; ddy -= round(ddy)
                d = np.sqrt(ddx**2 + ddy**2)
                if 0 < d <= pd['r0_pred']:
                    strength = pd['eps_pred'] * (1. - d/pd['r0_pred'])**1.5 / d
                    fx[j] -= strength * ddx
                    fy[j] -= strength * ddy

        # prey CoM
        cx = np.arctan2(np.sin(2*np.pi*x[:N]).mean(),
                        np.cos(2*np.pi*x[:N]).mean()) / (2*np.pi) % 1.
        cy = np.arctan2(np.sin(2*np.pi*x[N:]).mean(),
                        np.cos(2*np.pi*x[N:]).mean()) / (2*np.pi) % 1.

        # each predator chases CoM
        for ip in range(n_pred):
            tx = cx - pred_x[ip];  tx -= round(tx)
            ty = cy - pred_y[ip];  ty -= round(ty)
            dist = np.sqrt(tx**2 + ty**2)
            if dist > 0:
                tx /= dist; ty /= dist

            sp = np.sqrt(pred_vx[ip]**2 + pred_vy[ip]**2)
            pfx = pd['alpha_pred'] * tx
            pfy = pd['alpha_pred'] * ty
            if sp > 0:
                pfx += pd['mu_pred'] * (pd['v0_pred'] - sp) * pred_vx[ip]/sp
                pfy += pd['mu_pred'] * (pd['v0_pred'] - sp) * pred_vy[ip]/sp
            pfx += pd['ramp_pred'] * np.random.uniform(-1., 1.)
            pfy += pd['ramp_pred'] * np.random.uniform(-1., 1.)

            pred_vx[ip] += pfx * dt
            pred_vy[ip] += pfy * dt
            pred_x[ip] = (pred_x[ip] + pred_vx[ip]*dt) % 1.
            pred_y[ip] = (pred_y[ip] + pred_vy[ip]*dt) % 1.

        vx += fx * dt; vy += fy * dt
        x[:N] = (x[:N] + vx*dt) % 1.
        x[N:] = (x[N:] + vy*dt) % 1.

        if i % frame_every == 0:
            frames.append((x[:N].copy(), x[N:].copy(),
                           vx.copy(), vy.copy(),
                           pred_x.copy(), pred_y.copy()))

    return frames

def geom_multi(frames):
    Rg  = [gyration_radius(px, py) for px, py, _, _, _, _ in frames]
    AR  = [aspect_ratio(px, py)    for px, py, _, _, _, _ in frames]
    Phi = [order_parameter(vx, vy) for _, _, vx, vy, _, _ in frames]
    return np.array(Rg), np.array(AR), np.array(Phi)

def mean_min_pred_dist(frames):
    """Mean of the minimum predator-to-any-prey distance per frame."""
    dists = []
    for px, py, _, _, pxs, pys in frames:
        min_d = np.inf
        for ip in range(len(pxs)):
            ddx = pxs[ip] - px; ddx -= np.round(ddx)
            ddy = pys[ip] - py; ddy -= np.round(ddy)
            d = np.sqrt(ddx**2+ddy**2).min()
            min_d = min(min_d, d)
        dists.append(min_d)
    return np.array(dists)


# =============================================================================
# EXPERIMENT 1: 1 vs 2 vs 3 vs 4 PREDATORS
# =============================================================================
print('Exp 1: Flock response vs number of predators  (8 seeds)')
N_SEEDS = 8
n_pred_vals = [1, 2, 3, 4]
n_it = 4000; n_fr = 200
colors = ['steelblue', 'seagreen', 'darkorange', 'crimson']

Phi_np = {}; AR_np = {}; Rg_np = {}; Dist_np = {}

for n_pred in n_pred_vals:
    pp_e = PREY_DEFAULT.copy(); pp_e['n_iter'] = n_it
    Phi_here = []; AR_here = []; Rg_here = []; Dist_here = []
    for s in range(N_SEEDS):
        f = run_multi_predator(n_pred=n_pred, prey_overrides=pp_e,
                               n_frames=n_fr, seed=s)
        rg, ar, phi = geom_multi(f)
        Phi_here.append(phi); AR_here.append(ar); Rg_here.append(rg)
        Dist_here.append(mean_min_pred_dist(f))
    Phi_np[n_pred] = np.array(Phi_here)
    AR_np[n_pred]  = np.array(AR_here)
    Rg_np[n_pred]  = np.array(Rg_here)
    Dist_np[n_pred]= np.array(Dist_here)

    ss_phi = Phi_np[n_pred][:,-40:].mean()
    ss_ar  = AR_np[n_pred][:,-40:].mean()
    ss_rg  = Rg_np[n_pred][:,-40:].mean()
    ss_dist= Dist_np[n_pred][:,-40:].mean()
    print(f'  n_pred={n_pred}  Phi={ss_phi:.3f}  AR={ss_ar:.2f}  '
          f'Rg={ss_rg:.3f}  min_pred_dist={ss_dist:.3f}')

pp_base = PREY_DEFAULT.copy(); pp_base['n_iter'] = n_it
t = np.arange(n_fr) * max(1, n_it // n_fr) * PREY_DEFAULT['dt']

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('Flock response vs number of predators  (N=100, 8 seeds)', fontsize=11)

def band2(ax, t, arr_dict, keys, colors, ylabel, title, ylim=None):
    for i, k in enumerate(keys):
        arr = arr_dict[k]
        ax.plot(t, arr.mean(0), color=colors[i], lw=2, label=f'{k} pred')
        ax.fill_between(t, arr.mean(0)-arr.std(0), arr.mean(0)+arr.std(0),
                        color=colors[i], alpha=0.15)
    ax.set_xlabel('Time'); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(fontsize=8)
    if ylim: ax.set_ylim(*ylim)

band2(axes[0], t, Phi_np, n_pred_vals, colors, 'Order parameter Phi', 'Flock coherence', (0,1))
band2(axes[1], t, AR_np,  n_pred_vals, colors, 'Aspect ratio', 'Flock elongation')
axes[1].axhline(1, color='gray', ls='--', lw=0.8)
band2(axes[2], t, Dist_np, n_pred_vals, colors, 'Min predator-prey dist',
      'Nearest predator proximity')

plt.tight_layout()
plt.savefig('figures/multi_pred_1_npred_sweep.png', dpi=120)
plt.close()
print('  --> figures/multi_pred_1_npred_sweep.png')


# =============================================================================
# EXPERIMENT 2: SNAPSHOT SEQUENCE WITH 2 AND 4 PREDATORS
# =============================================================================
print('\nExp 2: Snapshot sequences')
for n_pred, n_it2 in [(2, 6000), (4, 6000)]:
    pp2 = PREY_DEFAULT.copy(); pp2['n_iter'] = n_it2; pp2['N'] = 150
    f2 = run_multi_predator(n_pred=n_pred, prey_overrides=pp2,
                            n_frames=300, seed=SEED)
    fs2 = max(1, n_it2//300); dt2 = PREY_DEFAULT['dt']
    rg2, ar2, phi2 = geom_multi(f2)

    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    fig.suptitle(f'{n_pred} predators, N=150 prey', fontsize=11)
    for ax, ts in zip(axes.flat, [0., 5., 10., 20., 30., 40., 50., 59.]):
        fi = min(int(ts/(fs2*dt2)), len(f2)-1)
        px, py, vx, vy, pxs, pys = f2[fi]
        sp = np.sqrt(vx**2+vy**2); sp[sp==0]=1.
        ax.scatter(px, py, s=3, color='steelblue', zorder=3)
        ax.quiver(px, py, vx/sp, vy/sp, scale=80, width=0.003,
                  color='steelblue', alpha=0.4)
        ax.scatter(pxs, pys, s=150, color='crimson', marker='*', zorder=5)
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect('equal')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f't={fi*fs2*dt2:.0f}  AR={ar2[fi]:.2f}', fontsize=8)
    plt.tight_layout()
    plt.savefig(f'figures/multi_pred_2_{n_pred}pred_snapshots.png', dpi=120)
    plt.close()
    print(f'  --> figures/multi_pred_2_{n_pred}pred_snapshots.png')


# =============================================================================
# EXPERIMENT 3: STEADY-STATE SUMMARY BAR CHART
# =============================================================================
print('\nExp 3: Steady-state summary')
ss_phi  = [Phi_np[n][:,-40:].mean()  for n in n_pred_vals]
ss_ar   = [AR_np[n][:,-40:].mean()   for n in n_pred_vals]
ss_dist = [Dist_np[n][:,-40:].mean() for n in n_pred_vals]
ss_phi_std  = [Phi_np[n][:,-40:].std()  for n in n_pred_vals]
ss_ar_std   = [AR_np[n][:,-40:].std()   for n in n_pred_vals]
ss_dist_std = [Dist_np[n][:,-40:].std() for n in n_pred_vals]

fig, axes = plt.subplots(1, 3, figsize=(13, 5))
fig.suptitle('Steady-state flock metrics vs number of predators\n(N=100, 8 seeds)', fontsize=11)
x_pos = np.array(n_pred_vals)
axes[0].bar(x_pos, ss_phi,  yerr=ss_phi_std,  color='steelblue', capsize=5, width=0.6)
axes[0].set_xlabel('Number of predators'); axes[0].set_ylabel('Phi')
axes[0].set_title('Flock coherence'); axes[0].set_ylim(0,1)
axes[1].bar(x_pos, ss_ar,   yerr=ss_ar_std,   color='seagreen',  capsize=5, width=0.6)
axes[1].set_xlabel('Number of predators'); axes[1].set_ylabel('Aspect ratio')
axes[1].set_title('Flock elongation'); axes[1].axhline(1, color='gray', ls='--', lw=0.8)
axes[2].bar(x_pos, ss_dist, yerr=ss_dist_std, color='crimson',   capsize=5, width=0.6)
axes[2].set_xlabel('Number of predators'); axes[2].set_ylabel('Min predator-prey dist')
axes[2].set_title('Evasion distance')
plt.tight_layout()
plt.savefig('figures/multi_pred_3_summary.png', dpi=120)
plt.close()
print('  --> figures/multi_pred_3_summary.png')

print('\nMulti-predator analysis complete.')
