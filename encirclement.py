# encirclement.py -- Can predators break the flock by approaching from all sides?
#
# Strategy: each predator k targets the point
#   prey_CoM + R_enc * (cos(2*pi*k/n_pred), sin(2*pi*k/n_pred))
# instead of CoM directly. This assigns each predator a fixed compass direction,
# so n_pred predators approach from n_pred equally spaced angles simultaneously.
#
# Experiments:
#   1. Encirclement radius sweep  (R_enc = 0..0.25, n_pred=3)
#   2. Encirclement vs naive      (n_pred=1..4, R_enc=0.15)
#   3. Flock-breaking threshold   (n_pred=1..8, R_enc=0.15)

import os
import numpy as np
import matplotlib.pyplot as plt
from flocking import params, buffer, force, order_parameter
from predator import PREY_DEFAULT, PRED_DEFAULT
from multi_predator import geom_multi, mean_min_pred_dist

os.makedirs('figures', exist_ok=True)
N_SEEDS = 8


# =============================================================================
# CORE SIMULATION
# =============================================================================
def run_encirclement(n_pred=3, R_enc=0.15, prey_overrides=None, pred_overrides=None,
                     n_frames=200, seed=None):
    """
    Each predator k chases prey_CoM + R_enc*(cos(theta_k), sin(theta_k))
    where theta_k = 2*pi*k/n_pred.  R_enc=0 recovers naive CoM-chasing.
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

    angles = np.array([2*np.pi*k/n_pred for k in range(n_pred)])

    x   = np.zeros(2*N)
    x[:N] = np.random.uniform(0., 1., N)
    x[N:] = np.random.uniform(0., 1., N)
    vx  = np.random.uniform(-1., 1., N) * p['v0']
    vy  = np.random.uniform(-1., 1., N) * p['v0']

    # start each predator far out on its assigned angle
    pred_x  = np.array([(0.5 + 0.45*np.cos(a)) % 1. for a in angles])
    pred_y  = np.array([(0.5 + 0.45*np.sin(a)) % 1. for a in angles])
    pred_vx = np.zeros(n_pred)
    pred_vy = np.zeros(n_pred)

    frames = []

    for i in range(p['n_iter']):
        rb = max(p['r0'], p['rf'])
        nb, xb, yb, vxb, vyb = buffer(rb, x, vx, vy, N)
        fx, fy = force(nb, xb, yb, vxb, vyb, x, vx, vy, p)

        # repulsion from each predator onto prey
        for ip in range(n_pred):
            for j in range(N):
                ddx = pred_x[ip] - x[j];   ddx -= round(ddx)
                ddy = pred_y[ip] - x[N+j]; ddy -= round(ddy)
                d = np.sqrt(ddx**2 + ddy**2)
                if 0 < d <= pd['r0_pred']:
                    s = pd['eps_pred'] * (1. - d/pd['r0_pred'])**1.5 / d
                    fx[j] -= s * ddx
                    fy[j] -= s * ddy

        # prey CoM (circular mean for periodic boundary)
        cx = np.arctan2(np.sin(2*np.pi*x[:N]).mean(),
                        np.cos(2*np.pi*x[:N]).mean()) / (2*np.pi) % 1.
        cy = np.arctan2(np.sin(2*np.pi*x[N:]).mean(),
                        np.cos(2*np.pi*x[N:]).mean()) / (2*np.pi) % 1.

        # each predator chases its offset point
        for ip in range(n_pred):
            tx_tgt = (cx + R_enc * np.cos(angles[ip])) % 1.
            ty_tgt = (cy + R_enc * np.sin(angles[ip])) % 1.

            tx = tx_tgt - pred_x[ip]; tx -= round(tx)
            ty = ty_tgt - pred_y[ip]; ty -= round(ty)
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


def pred_sep(frames):
    """Mean predator-predator separation (averaged over pairs and steady-state frames)."""
    seps = []
    for _, _, _, _, pxs, pys in frames[-40:]:
        n = len(pxs)
        if n < 2:
            return np.nan
        ds = []
        for ia in range(n):
            for ib in range(ia+1, n):
                dx = pxs[ia]-pxs[ib]; dx -= round(dx)
                dy = pys[ia]-pys[ib]; dy -= round(dy)
                ds.append(np.sqrt(dx**2+dy**2))
        seps.append(np.mean(ds))
    return np.mean(seps)


# =============================================================================
# EXP 1: ENCIRCLEMENT RADIUS SWEEP  (n_pred=3, R_enc = 0..0.25)
# =============================================================================
print('Exp 1: Encirclement radius sweep  (n_pred=3, %d seeds)' % N_SEEDS)
r_enc_vals = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25]
n_it = 4000; n_fr = 200
pp_e1 = PREY_DEFAULT.copy(); pp_e1['n_iter'] = n_it

res1 = {}
for R in r_enc_vals:
    Phi_r = []; AR_r = []; dist_r = []; sep_r = []
    for s in range(N_SEEDS):
        f = run_encirclement(n_pred=3, R_enc=R, prey_overrides=pp_e1,
                             n_frames=n_fr, seed=s)
        _, ar, phi = geom_multi(f)
        Phi_r.append(phi[-40:].mean())
        AR_r.append(ar[-40:].mean())
        dist_r.append(mean_min_pred_dist(f)[-40:].mean())
        sep_r.append(pred_sep(f))
    res1[R] = dict(phi=np.mean(Phi_r), ar=np.mean(AR_r),
                   dist=np.mean(dist_r), sep=np.nanmean(sep_r))
    print('  R_enc=%4.2f  Phi=%.3f  AR=%.2f  min_dist=%.3f  pred_sep=%.3f' % (
        R, res1[R]['phi'], res1[R]['ar'], res1[R]['dist'], res1[R]['sep']))

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle('Encirclement radius sweep  (n_pred=3, %d seeds)' % N_SEEDS, fontsize=11)
for ax, key, ylabel, title in zip(
        axes,
        ['phi', 'ar', 'dist', 'sep'],
        ['Order parameter Phi', 'Aspect ratio', 'Min pred-prey dist', 'Mean pred-pred sep'],
        ['Flock coherence', 'Elongation', 'Evasion distance', 'Predator spread']):
    vals = [res1[R][key] for R in r_enc_vals]
    ax.plot(r_enc_vals, vals, 'o-', color='steelblue', lw=2)
    ax.set_xlabel('Encirclement radius R_enc')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
if axes[0].get_ylim()[0] > 0:
    axes[0].set_ylim(0, 1)
plt.tight_layout()
plt.savefig('figures/encircle_1_radius_sweep.png', dpi=120)
plt.close()
print('  --> figures/encircle_1_radius_sweep.png')


# =============================================================================
# EXP 2: ENCIRCLEMENT vs NAIVE  (n_pred=1..4, R_enc=0.15)
# =============================================================================
print('\nExp 2: Encirclement vs naive  (n_pred=1..4, R_enc=0.15, %d seeds)' % N_SEEDS)
from multi_predator import run_multi_predator

n_pred_vals = [1, 2, 3, 4]
R_ENC = 0.15
pp_e2 = PREY_DEFAULT.copy(); pp_e2['n_iter'] = n_it

res2_naive = {}; res2_enc = {}

for n_pred in n_pred_vals:
    phi_n=[]; dist_n=[]; sep_n=[]
    phi_e=[]; dist_e=[]; sep_e=[]
    for s in range(N_SEEDS):
        fn = run_multi_predator(n_pred=n_pred, prey_overrides=pp_e2,
                                n_frames=n_fr, seed=s)
        fe = run_encirclement(n_pred=n_pred, R_enc=R_ENC, prey_overrides=pp_e2,
                              n_frames=n_fr, seed=s)
        _, _, phi = geom_multi(fn)
        phi_n.append(phi[-40:].mean())
        dist_n.append(mean_min_pred_dist(fn)[-40:].mean())
        sep_n.append(pred_sep(fn))

        _, _, phi = geom_multi(fe)
        phi_e.append(phi[-40:].mean())
        dist_e.append(mean_min_pred_dist(fe)[-40:].mean())
        sep_e.append(pred_sep(fe))

    res2_naive[n_pred] = dict(phi=np.mean(phi_n), dist=np.mean(dist_n),
                               sep=np.nanmean(sep_n))
    res2_enc[n_pred]   = dict(phi=np.mean(phi_e), dist=np.mean(dist_e),
                               sep=np.nanmean(sep_e))
    print('  n_pred=%d:' % n_pred)
    print('    naive:  Phi=%.3f  dist=%.3f  sep=%.3f' % (
        res2_naive[n_pred]['phi'], res2_naive[n_pred]['dist'],
        res2_naive[n_pred]['sep'] if not np.isnan(res2_naive[n_pred]['sep']) else float('nan')))
    print('    encircle: Phi=%.3f  dist=%.3f  sep=%.3f' % (
        res2_enc[n_pred]['phi'], res2_enc[n_pred]['dist'], res2_enc[n_pred]['sep']))

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('Naive vs encirclement predators  (R_enc=0.15, %d seeds)' % N_SEEDS, fontsize=11)
x_pos = np.array(n_pred_vals)
w = 0.35
for ax, key, ylabel, title, ylim in zip(
        axes,
        ['phi', 'dist', 'sep'],
        ['Order parameter Phi', 'Min pred-prey dist', 'Mean pred-pred sep'],
        ['Flock coherence', 'Evasion distance', 'Predator spread'],
        [(0, 1), None, None]):
    naive_v = [res2_naive[n][key] for n in n_pred_vals]
    enc_v   = [res2_enc[n][key]   for n in n_pred_vals]
    ax.bar(x_pos - w/2, naive_v,   w, label='Naive', color='steelblue', alpha=0.85)
    ax.bar(x_pos + w/2, enc_v,     w, label='Encircle', color='darkorange', alpha=0.85)
    ax.set_xlabel('Number of predators'); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.set_xticks(x_pos)
    ax.legend(fontsize=8)
    if ylim:
        ax.set_ylim(*ylim)
plt.tight_layout()
plt.savefig('figures/encircle_2_naive_vs_enc.png', dpi=120)
plt.close()
print('  --> figures/encircle_2_naive_vs_enc.png')


# =============================================================================
# EXP 3: FLOCK-BREAKING THRESHOLD  (encirclement, R_enc=0.15, n_pred=1..8)
# =============================================================================
print('\nExp 3: Flock-breaking threshold  (encirclement, R_enc=0.15, %d seeds)' % N_SEEDS)
n_pred_break = [1, 2, 3, 4, 6, 8]
pp_e3 = PREY_DEFAULT.copy(); pp_e3['n_iter'] = n_it

res3 = {}
for n_pred in n_pred_break:
    phi_all = []; dist_all = []
    for s in range(N_SEEDS):
        f = run_encirclement(n_pred=n_pred, R_enc=R_ENC, prey_overrides=pp_e3,
                             n_frames=n_fr, seed=s)
        _, _, phi = geom_multi(f)
        phi_all.append(phi[-40:].mean())
        dist_all.append(mean_min_pred_dist(f)[-40:].mean())
    res3[n_pred] = dict(phi=np.mean(phi_all), phi_std=np.std(phi_all),
                        dist=np.mean(dist_all))
    print('  n_pred=%2d  Phi=%.3f +/- %.3f  dist=%.3f' % (
        n_pred, res3[n_pred]['phi'], res3[n_pred]['phi_std'], res3[n_pred]['dist']))

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle('Flock-breaking threshold with encirclement  (R_enc=0.15, %d seeds)' % N_SEEDS,
             fontsize=11)
x = n_pred_break
phi_m   = [res3[n]['phi']     for n in x]
phi_s   = [res3[n]['phi_std'] for n in x]
dist_m  = [res3[n]['dist']    for n in x]
axes[0].errorbar(x, phi_m, yerr=phi_s, fmt='o-', color='steelblue', lw=2, capsize=5)
axes[0].axhline(0.5, color='gray', ls='--', lw=0.8, label='Phi=0.5 threshold')
axes[0].set_xlabel('Number of predators'); axes[0].set_ylabel('Phi')
axes[0].set_title('Flock coherence vs n_pred'); axes[0].set_ylim(0, 1)
axes[0].legend(fontsize=8)
axes[1].plot(x, dist_m, 'o-', color='crimson', lw=2)
axes[1].set_xlabel('Number of predators'); axes[1].set_ylabel('Min pred-prey dist')
axes[1].set_title('Evasion distance vs n_pred')
plt.tight_layout()
plt.savefig('figures/encircle_3_breaking_threshold.png', dpi=120)
plt.close()
print('  --> figures/encircle_3_breaking_threshold.png')

print('\nEncirclement analysis complete.')
