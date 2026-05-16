# coordinated_predators.py -- Do coordinated predators break the flock?
#
# Finding 11 showed that naive predators all pile up at the prey CoM (pred-pred
# dist ~0.001), inadvertently helping the flock by concentrating repulsion at one
# point. This script asks: if predators repel each other (forcing them to spread
# out and approach from different angles), does that collapse the prey flock?
#
# Coordination mechanism: predator-predator repulsion force with tunable strength
# alpha_coord and range r_coord. When alpha_coord=0, this reduces exactly to the
# naive multi-predator model from multi_predator.py.
#
# Experiments:
#   1. Coordination strength sweep (n_pred=3, vary alpha_coord) -- does spreading
#      out actually let predators get closer?
#   2. Naive vs coordinated, n_pred=1..4 -- does coordination flip the trend?
#   3. Flock-breaking threshold -- how many coordinated predators does it take
#      to collapse Phi below 0.5?

import os
import numpy as np
import matplotlib.pyplot as plt
from flocking import run, params, buffer, force, order_parameter
from predator import PREY_DEFAULT, PRED_DEFAULT
from geometry import gyration_radius, aspect_ratio

os.makedirs('figures', exist_ok=True)
N_SEEDS = 8
TAIL_FRAC = 0.4


# =============================================================================
# COORDINATED MULTI-PREDATOR SIMULATION
# =============================================================================
def run_coordinated(n_pred=2, alpha_coord=5.0, r_coord=0.4,
                    prey_overrides=None, pred_overrides=None,
                    n_frames=200, seed=None):
    """
    Multi-predator run with predator-predator repulsion.
    alpha_coord: strength of inter-predator repulsion (0 = naive baseline)
    r_coord:     range of inter-predator repulsion
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
    N = p['N']; dt = p['dt']
    frame_every = max(1, p['n_iter'] // n_frames)

    x  = np.zeros(2*N)
    x[:N] = np.random.uniform(0., 1., N)
    x[N:] = np.random.uniform(0., 1., N)
    vx = np.random.uniform(-1., 1., N) * p['v0']
    vy = np.random.uniform(-1., 1., N) * p['v0']

    pred_x  = np.random.uniform(0., 1., n_pred)
    pred_y  = np.random.uniform(0., 1., n_pred)
    pred_vx = np.random.uniform(-1., 1., n_pred) * pd['v0_pred']
    pred_vy = np.random.uniform(-1., 1., n_pred) * pd['v0_pred']

    frames = []

    for i in range(p['n_iter']):
        rb = max(p['r0'], p['rf'])
        nb, xb, yb, vxb, vyb = buffer(rb, x, vx, vy, N)
        fx, fy = force(nb, xb, yb, vxb, vyb, x, vx, vy, p)

        # Predator repulsion on prey
        for ip in range(n_pred):
            for j in range(N):
                ddx = pred_x[ip] - x[j];   ddx -= round(ddx)
                ddy = pred_y[ip] - x[N+j]; ddy -= round(ddy)
                d = np.sqrt(ddx**2 + ddy**2)
                if 0 < d <= pd['r0_pred']:
                    s = pd['eps_pred'] * (1. - d/pd['r0_pred'])**1.5 / d
                    fx[j] -= s * ddx
                    fy[j] -= s * ddy

        # Prey CoM
        cx = np.arctan2(np.sin(2*np.pi*x[:N]).mean(),
                        np.cos(2*np.pi*x[:N]).mean()) / (2*np.pi) % 1.
        cy = np.arctan2(np.sin(2*np.pi*x[N:]).mean(),
                        np.cos(2*np.pi*x[N:]).mean()) / (2*np.pi) % 1.

        # Each predator: chase CoM + repel from other predators
        pfx_all = np.zeros(n_pred)
        pfy_all = np.zeros(n_pred)

        for ip in range(n_pred):
            # Chase CoM
            tx = cx - pred_x[ip]; tx -= round(tx)
            ty = cy - pred_y[ip]; ty -= round(ty)
            dist = np.sqrt(tx**2 + ty**2)
            if dist > 0:
                tx /= dist; ty /= dist

            sp = np.sqrt(pred_vx[ip]**2 + pred_vy[ip]**2)
            pfx_all[ip] += pd['alpha_pred'] * tx
            pfy_all[ip] += pd['alpha_pred'] * ty
            if sp > 0:
                pfx_all[ip] += pd['mu_pred'] * (pd['v0_pred']-sp) * pred_vx[ip]/sp
                pfy_all[ip] += pd['mu_pred'] * (pd['v0_pred']-sp) * pred_vy[ip]/sp
            pfx_all[ip] += pd['ramp_pred'] * np.random.uniform(-1., 1.)
            pfy_all[ip] += pd['ramp_pred'] * np.random.uniform(-1., 1.)

            # Repel from other predators
            for jp in range(n_pred):
                if jp == ip:
                    continue
                ddx = pred_x[jp] - pred_x[ip]; ddx -= round(ddx)
                ddy = pred_y[jp] - pred_y[ip]; ddy -= round(ddy)
                d = np.sqrt(ddx**2 + ddy**2)
                if 0 < d <= r_coord:
                    s = alpha_coord * (1. - d/r_coord)**1.5 / d
                    pfx_all[ip] -= s * ddx
                    pfy_all[ip] -= s * ddy

        for ip in range(n_pred):
            pred_vx[ip] += pfx_all[ip] * dt
            pred_vy[ip] += pfy_all[ip] * dt
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


def tail_stats(frames):
    n = max(1, int(TAIL_FRAC * len(frames)))
    tail = frames[-n:]
    Phi = np.mean([order_parameter(f[2], f[3]) for f in tail])
    AR  = np.mean([aspect_ratio(f[0], f[1])    for f in tail])
    Rg  = np.mean([gyration_radius(f[0], f[1]) for f in tail])
    # min predator-prey distance
    mind_list = []
    for px, py, _, _, pxs, pys in tail:
        min_d = np.inf
        for ip in range(len(pxs)):
            ddx = pxs[ip]-px; ddx -= np.round(ddx)
            ddy = pys[ip]-py; ddy -= np.round(ddy)
            d = np.sqrt(ddx**2+ddy**2).min()
            min_d = min(min_d, d)
        mind_list.append(min_d)
    mind = np.mean(mind_list)
    # pred-pred mean distance
    ppd_list = []
    for _, _, _, _, pxs, pys in tail:
        np_ = len(pxs)
        if np_ < 2:
            continue
        ds = []
        for ip in range(np_):
            for jp in range(ip+1, np_):
                ddx = pxs[ip]-pxs[jp]; ddx -= round(ddx)
                ddy = pys[ip]-pys[jp]; ddy -= round(ddy)
                ds.append(np.sqrt(ddx**2+ddy**2))
        ppd_list.append(np.mean(ds))
    ppd = np.mean(ppd_list) if ppd_list else np.nan
    return Phi, AR, Rg, mind, ppd


# =============================================================================
# EXPERIMENT 1: COORDINATION STRENGTH SWEEP  (n_pred=3)
# =============================================================================
print('Exp 1: Coordination strength sweep  (n_pred=3, 8 seeds)')
N_IT = 4000; N_FR = 200
pp_e = PREY_DEFAULT.copy(); pp_e['n_iter'] = N_IT
alpha_coord_vals = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

res1 = {a: {'Phi':[], 'AR':[], 'mind':[], 'ppd':[]} for a in alpha_coord_vals}

for ac in alpha_coord_vals:
    for s in range(N_SEEDS):
        f = run_coordinated(n_pred=3, alpha_coord=ac, prey_overrides=pp_e,
                            n_frames=N_FR, seed=s)
        Phi, AR, Rg, mind, ppd = tail_stats(f)
        res1[ac]['Phi'].append(Phi)
        res1[ac]['AR'].append(AR)
        res1[ac]['mind'].append(mind)
        res1[ac]['ppd'].append(ppd)
    print(f'  alpha_coord={ac:5.1f}  Phi={np.mean(res1[ac]["Phi"]):.3f}  '
          f'AR={np.mean(res1[ac]["AR"]):.2f}  '
          f'min_dist={np.mean(res1[ac]["mind"]):.3f}  '
          f'pred_sep={np.nanmean(res1[ac]["ppd"]):.3f}')

fig, axes = plt.subplots(1, 4, figsize=(16, 5))
fig.suptitle('Exp 1: Coordination strength sweep  (n_pred=3, N=100 prey, 8 seeds)', fontsize=11)
keys = ['Phi', 'AR', 'mind', 'ppd']
labels = ['Order parameter Phi', 'Aspect ratio AR', 'Min predator-prey dist', 'Pred-pred separation']
colors_e1 = ['steelblue', 'seagreen', 'crimson', 'darkorange']
for ax, k, lab, col in zip(axes, keys, labels, colors_e1):
    means = [np.nanmean(res1[a][k]) for a in alpha_coord_vals]
    stds  = [np.nanstd(res1[a][k])  for a in alpha_coord_vals]
    ax.errorbar(alpha_coord_vals, means, yerr=stds, fmt='o-', color=col,
                capsize=4, lw=2, ms=5)
    ax.set_xlabel('Coordination strength (alpha_coord)')
    ax.set_ylabel(lab)
    ax.set_title(lab)
axes[0].set_ylim(0, 1.05)
axes[0].axhline(0.5, color='gray', ls='--', lw=0.8, label='Phi=0.5 threshold')
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.savefig('figures/coord_1_strength_sweep.png', dpi=120)
plt.close()
print('  --> figures/coord_1_strength_sweep.png\n')


# =============================================================================
# EXPERIMENT 2: NAIVE vs COORDINATED, n_pred = 1..4
# =============================================================================
print('Exp 2: Naive vs coordinated predators, n_pred=1..4  (8 seeds)')
# Use alpha_coord=10 as "coordinated" -- strong enough to spread predators well
ALPHA_COORD = 10.0
n_pred_vals = [1, 2, 3, 4]
colors2 = ['steelblue', 'crimson']

res2 = {'naive': {n: {'Phi':[], 'mind':[], 'ppd':[]} for n in n_pred_vals},
        'coord': {n: {'Phi':[], 'mind':[], 'ppd':[]} for n in n_pred_vals}}

for n_pred in n_pred_vals:
    pp_e2 = PREY_DEFAULT.copy(); pp_e2['n_iter'] = N_IT
    for s in range(N_SEEDS):
        # Naive (alpha_coord=0)
        f_n = run_coordinated(n_pred=n_pred, alpha_coord=0.0,
                              prey_overrides=pp_e2, n_frames=N_FR, seed=s)
        Phi, _, _, mind, ppd = tail_stats(f_n)
        res2['naive'][n_pred]['Phi'].append(Phi)
        res2['naive'][n_pred]['mind'].append(mind)
        res2['naive'][n_pred]['ppd'].append(ppd)

        # Coordinated
        f_c = run_coordinated(n_pred=n_pred, alpha_coord=ALPHA_COORD,
                              prey_overrides=pp_e2, n_frames=N_FR, seed=s)
        Phi, _, _, mind, ppd = tail_stats(f_c)
        res2['coord'][n_pred]['Phi'].append(Phi)
        res2['coord'][n_pred]['mind'].append(mind)
        res2['coord'][n_pred]['ppd'].append(ppd)

    print(f'  n_pred={n_pred}:')
    print(f'    naive:  Phi={np.mean(res2["naive"][n_pred]["Phi"]):.3f}  '
          f'dist={np.mean(res2["naive"][n_pred]["mind"]):.3f}  '
          f'sep={np.nanmean(res2["naive"][n_pred]["ppd"]):.3f}')
    print(f'    coord:  Phi={np.mean(res2["coord"][n_pred]["Phi"]):.3f}  '
          f'dist={np.mean(res2["coord"][n_pred]["mind"]):.3f}  '
          f'sep={np.nanmean(res2["coord"][n_pred]["ppd"]):.3f}')

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle(f'Exp 2: Naive vs coordinated predators  (alpha_coord={ALPHA_COORD}, 8 seeds)', fontsize=11)
x_pos = np.array(n_pred_vals)
metrics = [('Phi', 'Order parameter Phi', (0,1.05)),
           ('mind', 'Min predator-prey dist', None),
           ('ppd', 'Pred-pred separation', None)]
for ax, (k, lab, ylim) in zip(axes, metrics):
    for mode, col, lbl in [('naive','steelblue','Naive'), ('coord','crimson',f'Coordinated (ac={ALPHA_COORD})')]:
        means = [np.nanmean(res2[mode][n][k]) for n in n_pred_vals]
        stds  = [np.nanstd(res2[mode][n][k])  for n in n_pred_vals]
        ax.errorbar(x_pos, means, yerr=stds, fmt='o-', color=col,
                    capsize=4, lw=2, ms=5, label=lbl)
    ax.set_xlabel('Number of predators')
    ax.set_ylabel(lab)
    ax.set_title(lab)
    ax.legend(fontsize=8)
    if ylim:
        ax.set_ylim(*ylim)
axes[0].axhline(0.5, color='gray', ls='--', lw=0.8)
plt.tight_layout()
plt.savefig('figures/coord_2_naive_vs_coord.png', dpi=120)
plt.close()
print('  --> figures/coord_2_naive_vs_coord.png\n')


# =============================================================================
# EXPERIMENT 3: FLOCK-BREAKING THRESHOLD
# How many coordinated predators does it take to collapse Phi < 0.5?
# =============================================================================
print('Exp 3: Flock-breaking threshold  (coordinated, alpha_coord=10, 8 seeds)')
n_pred_break = [1, 2, 3, 4, 6, 8, 10]

res3 = {n: {'Phi': [], 'mind': []} for n in n_pred_break}

for n_pred in n_pred_break:
    pp_e3 = PREY_DEFAULT.copy(); pp_e3['n_iter'] = N_IT
    for s in range(N_SEEDS):
        f = run_coordinated(n_pred=n_pred, alpha_coord=ALPHA_COORD,
                            prey_overrides=pp_e3, n_frames=N_FR, seed=s)
        Phi, _, _, mind, _ = tail_stats(f)
        res3[n_pred]['Phi'].append(Phi)
        res3[n_pred]['mind'].append(mind)
    print(f'  n_pred={n_pred:2d}  Phi={np.mean(res3[n_pred]["Phi"]):.3f} '
          f'+/- {np.std(res3[n_pred]["Phi"]):.3f}  '
          f'dist={np.mean(res3[n_pred]["mind"]):.3f}')

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle(f'Exp 3: Flock-breaking threshold  (coordinated, alpha_coord={ALPHA_COORD}, 8 seeds)', fontsize=11)

phi_means = [np.mean(res3[n]['Phi'])  for n in n_pred_break]
phi_stds  = [np.std(res3[n]['Phi'])   for n in n_pred_break]
dist_means= [np.mean(res3[n]['mind']) for n in n_pred_break]
dist_stds = [np.std(res3[n]['mind'])  for n in n_pred_break]

axes[0].errorbar(n_pred_break, phi_means, yerr=phi_stds, fmt='o-',
                 color='steelblue', capsize=4, lw=2, ms=5)
axes[0].axhline(0.5, color='crimson', ls='--', lw=1.2, label='Phi=0.5 (breakdown)')
axes[0].set_xlabel('Number of coordinated predators')
axes[0].set_ylabel('Order parameter Phi')
axes[0].set_title('Flock coherence vs predator count\n(coordinated)')
axes[0].set_ylim(0, 1.05)
axes[0].legend(fontsize=9)

axes[1].errorbar(n_pred_break, dist_means, yerr=dist_stds, fmt='o-',
                 color='crimson', capsize=4, lw=2, ms=5)
axes[1].set_xlabel('Number of coordinated predators')
axes[1].set_ylabel('Min predator-prey dist')
axes[1].set_title('Evasion distance vs predator count\n(coordinated)')

plt.tight_layout()
plt.savefig('figures/coord_3_breaking_threshold.png', dpi=120)
plt.close()
print('  --> figures/coord_3_breaking_threshold.png\n')

print('Coordinated predator analysis complete.')
