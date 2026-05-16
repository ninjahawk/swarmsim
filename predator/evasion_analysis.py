# evasion_analysis.py -- Why does evasion distance increase with more predators?
#
# Hypothesis A: Predators cluster near the CoM and block each other -- competing
#               for the same target leaves each one farther from the nearest prey.
# Hypothesis B: The flock elongates perpendicular to the predator attack directions,
#               presenting a harder target to surround.
#
# Diagnostics:
#   1. Angular spread of predators around flock CoM (do they spread or cluster?)
#   2. Predator-predator mean distance (do more predators push each other apart?)
#   3. Flock elongation axis vs predator centroid angle (does the flock orient away?)

import os
import numpy as np
import matplotlib.pyplot as plt
from multi_predator import run_multi_predator, geom_multi, mean_min_pred_dist
from predator import PREY_DEFAULT, PRED_DEFAULT

os.makedirs('figures', exist_ok=True)

N_SEEDS   = 8
N_PRED_VALS = [1, 2, 3, 4]
N_ITER    = 4000
N_FRAMES  = 200
TAIL_FRAC = 0.4    # use last 40% of frames as steady state
colors    = ['steelblue', 'seagreen', 'darkorange', 'crimson']


def angular_spread(pred_xs, pred_ys, cx, cy):
    """
    Angular spread of predators around flock CoM (cx, cy).
    Returns std of angles (radians). 0 = all clustered, pi = maximally spread.
    For n_pred=1 returns 0.
    """
    if len(pred_xs) < 2:
        return 0.0
    ddx = pred_xs - cx; ddx -= np.round(ddx)
    ddy = pred_ys - cy; ddy -= np.round(ddy)
    angles = np.arctan2(ddy, ddx)
    # circular std
    C = np.cos(angles).mean(); S = np.sin(angles).mean()
    R = np.sqrt(C**2 + S**2)
    return float(np.sqrt(-2 * np.log(np.clip(R, 1e-9, 1))))


def pred_pred_mean_dist(pred_xs, pred_ys):
    """Mean pairwise predator-predator distance (periodic)."""
    n = len(pred_xs)
    if n < 2:
        return np.nan
    dists = []
    for i in range(n):
        for j in range(i+1, n):
            ddx = pred_xs[i]-pred_xs[j]; ddx -= round(ddx)
            ddy = pred_ys[i]-pred_ys[j]; ddy -= round(ddy)
            dists.append(np.sqrt(ddx**2+ddy**2))
    return float(np.mean(dists))


def flock_orientation_vs_predator_centroid(px, py, vx, vy, pred_xs, pred_ys):
    """
    Returns the angle between:
      - the flock major axis (eigenvector of spatial covariance)
      - the direction from flock CoM to predator centroid
    A small angle means the flock is elongated TOWARD the predator.
    An angle near pi/2 means the flock is elongated PERPENDICULAR (broad side to predator).
    """
    cx = np.arctan2(np.sin(2*np.pi*px).mean(),
                    np.cos(2*np.pi*px).mean()) / (2*np.pi) % 1.
    cy = np.arctan2(np.sin(2*np.pi*py).mean(),
                    np.cos(2*np.pi*py).mean()) / (2*np.pi) % 1.

    # flock spatial covariance -> major axis
    dx = px - cx; dx -= np.round(dx)
    dy = py - cy; dy -= np.round(dy)
    cov = np.cov(dx, dy)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major_axis = eigvecs[:, np.argmax(eigvals)]   # unit vector

    # predator centroid direction from CoM
    pcx = pred_xs.mean(); pcy = pred_ys.mean()
    ddx = pcx - cx; ddx -= round(ddx)
    ddy = pcy - cy; ddy -= round(ddy)
    d = np.sqrt(ddx**2 + ddy**2)
    if d < 1e-9:
        return np.nan
    pred_dir = np.array([ddx/d, ddy/d])

    cos_a = np.clip(abs(np.dot(major_axis, pred_dir)), 0, 1)
    return float(np.arccos(cos_a))   # 0 = aligned, pi/2 = perpendicular


# =============================================================================
# COLLECT DIAGNOSTICS
# =============================================================================
print('Evasion analysis: angular spread, predator separation, flock orientation')
print()

results = {}   # n_pred -> dict of arrays over seeds x frames

for n_pred in N_PRED_VALS:
    pp = PREY_DEFAULT.copy(); pp['n_iter'] = N_ITER
    ang_spread_all = []; pp_dist_all = []; orient_all = []; min_d_all = []

    for s in range(N_SEEDS):
        frames = run_multi_predator(n_pred=n_pred, prey_overrides=pp,
                                   n_frames=N_FRAMES, seed=s)
        n_tail = max(1, int(TAIL_FRAC * len(frames)))
        tail = frames[-n_tail:]

        ang  = [angular_spread(f[4], f[5],
                               np.arctan2(np.sin(2*np.pi*f[0]).mean(),
                                          np.cos(2*np.pi*f[0]).mean())/(2*np.pi)%1.,
                               np.arctan2(np.sin(2*np.pi*f[1]).mean(),
                                          np.cos(2*np.pi*f[1]).mean())/(2*np.pi)%1.)
                for f in tail]
        ppd  = [pred_pred_mean_dist(f[4], f[5]) for f in tail]
        ori  = [flock_orientation_vs_predator_centroid(f[0], f[1], f[2], f[3], f[4], f[5])
                for f in tail]
        mind = []
        for f in tail:
            px, py, _, _, pxs, pys = f
            min_d = np.inf
            for ip in range(len(pxs)):
                ddx = pxs[ip]-px; ddx -= np.round(ddx)
                ddy = pys[ip]-py; ddy -= np.round(ddy)
                d = np.sqrt(ddx**2+ddy**2).min()
                min_d = min(min_d, d)
            mind.append(min_d)

        ang_spread_all.append(np.nanmean(ang))
        pp_dist_all.append(np.nanmean([x for x in ppd if not np.isnan(x)]) if n_pred > 1 else np.nan)
        orient_all.append(np.nanmean([x for x in ori if not np.isnan(x)]))
        min_d_all.append(np.mean(mind))

    results[n_pred] = dict(
        ang_spread = np.array(ang_spread_all),
        pp_dist    = np.array(pp_dist_all),
        orient     = np.array(orient_all),
        min_d      = np.array(min_d_all),
    )
    print(f'n_pred={n_pred}:')
    print(f'  angular spread (rad): {np.nanmean(ang_spread_all):.3f} +/- {np.nanstd(ang_spread_all):.3f}')
    if n_pred > 1:
        print(f'  pred-pred dist:       {np.nanmean(pp_dist_all):.3f} +/- {np.nanstd(pp_dist_all):.3f}')
    print(f'  flock orient vs pred: {np.nanmean(orient_all)*180/np.pi:.1f} deg (0=toward, 90=perp)')
    print(f'  min pred-prey dist:   {np.mean(min_d_all):.3f} +/- {np.std(min_d_all):.3f}')
    print()


# =============================================================================
# PLOTS
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('Evasion distance diagnostic: why does distance increase with more predators?\n'
             '(N=100 prey, 8 seeds)', fontsize=10)

x_pos = np.array(N_PRED_VALS)

# Panel 1: Predator angular spread
ang_means = [np.nanmean(results[n]['ang_spread']) for n in N_PRED_VALS]
ang_stds  = [np.nanstd(results[n]['ang_spread'])  for n in N_PRED_VALS]
axes[0].bar(x_pos, ang_means, yerr=ang_stds, color='steelblue', capsize=5, width=0.6)
axes[0].set_xlabel('Number of predators')
axes[0].set_ylabel('Angular spread (rad, circular std)')
axes[0].set_title('Predator angular spread\naround flock CoM\n(higher = more spread out)')
axes[0].set_xticks(x_pos)

# Panel 2: Predator-predator mean distance (n_pred >= 2)
valid_n = [n for n in N_PRED_VALS if n > 1]
ppd_means = [np.nanmean(results[n]['pp_dist']) for n in valid_n]
ppd_stds  = [np.nanstd(results[n]['pp_dist'])  for n in valid_n]
axes[1].bar(np.array(valid_n), ppd_means, yerr=ppd_stds, color='seagreen', capsize=5, width=0.6)
axes[1].set_xlabel('Number of predators')
axes[1].set_ylabel('Mean predator-predator distance')
axes[1].set_title('Predator separation\n(do predators spread out\nor cluster at CoM?)')
axes[1].set_xticks(valid_n)

# Panel 3: Flock orientation angle vs predator centroid
ori_means_deg = [np.nanmean(results[n]['orient'])*180/np.pi for n in N_PRED_VALS]
ori_stds_deg  = [np.nanstd(results[n]['orient'])*180/np.pi  for n in N_PRED_VALS]
axes[2].bar(x_pos, ori_means_deg, yerr=ori_stds_deg, color='darkorange', capsize=5, width=0.6)
axes[2].axhline(45, color='gray', ls='--', lw=1, label='45 deg (random)')
axes[2].axhline(90, color='crimson', ls=':', lw=1, label='90 deg (perpendicular)')
axes[2].set_xlabel('Number of predators')
axes[2].set_ylabel('Flock major axis angle vs predator (deg)')
axes[2].set_title('Flock orientation vs predator\n(90 deg = broad side toward\npredator; 0 deg = narrow end)')
axes[2].set_xticks(x_pos)
axes[2].legend(fontsize=8)
axes[2].set_ylim(0, 100)

plt.tight_layout()
plt.savefig('figures/evasion_diagnostic.png', dpi=120)
plt.close()
print('Saved: figures/evasion_diagnostic.png')
print()
print('Interpretation guide:')
print('  Angular spread > 0 with more predators -> predators spread around the flock')
print('  Pred-pred distance increases with n_pred -> predators are not simply clustering at CoM')
print('  Flock orient ~90 deg -> flock presents broad side to predator centroid (harder to surround)')
print('  Flock orient ~0 deg -> flock turns narrow end toward predator (streamlining)')
