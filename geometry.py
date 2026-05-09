# geometry.py -- Flock shape analysis: radius of gyration, aspect ratio, arching
#
# The book predicts that a predator causes the flock to arch and split.
# This script quantifies flock geometry over time, with and without a predator,
# to test whether those shape changes are real and measurable.
#
# Metrics:
#   Rg   -- radius of gyration: sqrt(mean squared distance from CoM). Measures spread.
#   AR   -- aspect ratio: ratio of major to minor axis of the flock's inertia ellipse.
#           AR=1 means circular flock, AR>>1 means elongated/arched.
#   Phi  -- order parameter (already defined in flocking.py)

import os
import numpy as np
import matplotlib.pyplot as plt
from flocking import run, params, order_parameter
from predator import run_predator, PREY_DEFAULT, PRED_DEFAULT

os.makedirs('figures', exist_ok=True)
SEED = 7


# =============================================================================
# GEOMETRY UTILITIES
# =============================================================================
def periodic_com(pos, L=1.0):
    """Center of mass on a torus using the standard angle trick."""
    theta = 2 * np.pi * pos / L
    cx = np.arctan2(np.sin(theta).mean(), np.cos(theta).mean()) / (2*np.pi) * L % L
    return cx

def periodic_disp(pos, com, L=1.0):
    """Displacement from CoM on a torus (shortest path)."""
    d = pos - com
    d -= np.round(d / L) * L
    return d

def gyration_radius(px, py):
    """Radius of gyration: sqrt(mean squared distance from CoM)."""
    cx = periodic_com(px); cy = periodic_com(py)
    dx = periodic_disp(px, cx); dy = periodic_disp(py, cy)
    return np.sqrt((dx**2 + dy**2).mean())

def aspect_ratio(px, py):
    """
    Aspect ratio of flock from inertia tensor eigenvalues.
    Returns ratio of largest to smallest eigenvalue of the covariance matrix.
    AR = 1 means circular; AR > 1 means elongated.
    """
    cx = periodic_com(px); cy = periodic_com(py)
    dx = periodic_disp(px, cx); dy = periodic_disp(py, cy)
    cov = np.cov(dx, dy)
    evals = np.linalg.eigvalsh(cov)
    evals = np.sort(np.abs(evals))
    if evals[0] < 1e-10:
        return 1.0
    return float(evals[1] / evals[0])

def geometry_series(frames):
    """Return (Rg, AR, Phi) time series from a frame list."""
    Rg  = [gyration_radius(px, py)       for px, py, _, _    in frames]
    AR  = [aspect_ratio(px, py)           for px, py, _, _    in frames]
    Phi = [order_parameter(vx, vy)        for _, _, vx, vy    in frames]
    return np.array(Rg), np.array(AR), np.array(Phi)

def geometry_series_pred(frames):
    """Same but for predator frames (px,py,vx,vy,prdx,prdy,pvx,pvy)."""
    Rg  = [gyration_radius(px, py)       for px, py, _, _, _, _, _, _ in frames]
    AR  = [aspect_ratio(px, py)           for px, py, _, _, _, _, _, _ in frames]
    Phi = [order_parameter(vx, vy)        for _, _, vx, vy, _, _, _, _ in frames]
    return np.array(Rg), np.array(AR), np.array(Phi)

def pred_dist_series(frames):
    """Mean distance from predator to nearest 20% of prey, per frame."""
    dists = []
    N_prey = len(frames[0][0])
    k = max(1, N_prey // 5)
    for px, py, _, _, prdx, prdy, _, _ in frames:
        ddx = prdx - px; ddx -= np.round(ddx)
        ddy = prdy - py; ddy -= np.round(ddy)
        d = np.sqrt(ddx**2 + ddy**2)
        dists.append(np.sort(d)[:k].mean())
    return np.array(dists)


# =============================================================================
# EXPERIMENT 1: FLOCK GEOMETRY WITH AND WITHOUT PREDATOR
# =============================================================================
print('Exp 1: Flock geometry -- with and without predator (10 seeds)')
N_SEEDS = 10
n_it = 5000; n_fr = 250

# without predator
p_base = params(dict(N=100, n_iter=n_it, v0=0.02, alpha=1.0, ramp=0.1))
Rg_no  = []; AR_no  = []; Phi_no  = []
for s in range(N_SEEDS):
    f = run(p_base, n_frames=n_fr, seed=s)
    rg, ar, phi = geometry_series(f)
    Rg_no.append(rg); AR_no.append(ar); Phi_no.append(phi)

# with predator
pp = PREY_DEFAULT.copy(); pp['n_iter'] = n_it
Rg_yes = []; AR_yes = []; Phi_yes = []; Dist_yes = []
for s in range(N_SEEDS):
    f = run_predator(prey_overrides=pp, seed=s, n_frames=n_fr)
    rg, ar, phi = geometry_series_pred(f)
    Rg_yes.append(rg); AR_yes.append(ar); Phi_yes.append(phi)
    Dist_yes.append(pred_dist_series(f))

Rg_no  = np.array(Rg_no);  AR_no  = np.array(AR_no);  Phi_no  = np.array(Phi_no)
Rg_yes = np.array(Rg_yes); AR_yes = np.array(AR_yes); Phi_yes = np.array(Phi_yes)
Dist_yes = np.array(Dist_yes)
fs = max(1, n_it // n_fr); dt = p_base['dt']
t = np.arange(n_fr) * fs * dt

print(f'  No predator:   Rg={Rg_no[:,-20:].mean():.3f}  AR={AR_no[:,-20:].mean():.2f}  Phi={Phi_no[:,-20:].mean():.3f}')
print(f'  With predator: Rg={Rg_yes[:,-20:].mean():.3f}  AR={AR_yes[:,-20:].mean():.2f}  Phi={Phi_yes[:,-20:].mean():.3f}')

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('Flock geometry: with vs without predator  (N=100, 10 seeds)', fontsize=11)

def band(ax, t, arr, color, label):
    ax.plot(t, arr.mean(0), color=color, lw=2, label=label)
    ax.fill_between(t, arr.mean(0)-arr.std(0), arr.mean(0)+arr.std(0),
                    color=color, alpha=0.2)

band(axes[0], t, Rg_no,  'steelblue', 'No predator')
band(axes[0], t, Rg_yes, 'crimson',   'With predator')
axes[0].set_xlabel('Time'); axes[0].set_ylabel('Radius of gyration Rg')
axes[0].set_title('Flock spread (Rg)'); axes[0].legend()

band(axes[1], t, AR_no,  'steelblue', 'No predator')
band(axes[1], t, AR_yes, 'crimson',   'With predator')
axes[1].set_xlabel('Time'); axes[1].set_ylabel('Aspect ratio')
axes[1].set_title('Flock elongation (AR)\nAR=1: circular, AR>1: stretched')
axes[1].axhline(1.0, color='gray', ls='--', lw=0.8); axes[1].legend()

band(axes[2], t, Phi_no,  'steelblue', 'No predator')
band(axes[2], t, Phi_yes, 'crimson',   'With predator')
axes[2].set_xlabel('Time'); axes[2].set_ylabel('Order parameter Phi')
axes[2].set_title('Flock coherence (Phi)'); axes[2].set_ylim(0, 1); axes[2].legend()

plt.tight_layout()
plt.savefig('figures/geometry_1_pred_vs_no.png', dpi=120)
plt.close()
print('  --> figures/geometry_1_pred_vs_no.png')


# =============================================================================
# EXPERIMENT 2: DOES STRONGER FLOCKING CHANGE FLOCK SHAPE RESPONSE?
# =============================================================================
print('\nExp 2: Flock geometry vs flocking amplitude alpha  (8 seeds)')
N_SEEDS2 = 8
alpha_vals = [0.2, 0.5, 1.0, 2.0]
n_it2 = 4000; n_fr2 = 200
colors = ['steelblue', 'seagreen', 'darkorange', 'crimson']

Rg_a = {}; AR_a = {}; Phi_a = {}
for alpha in alpha_vals:
    pp2 = PREY_DEFAULT.copy(); pp2['n_iter'] = n_it2; pp2['alpha'] = alpha
    Rg_here = []; AR_here = []; Phi_here = []
    for s in range(N_SEEDS2):
        f = run_predator(prey_overrides=pp2, seed=s, n_frames=n_fr2)
        rg, ar, phi = geometry_series_pred(f)
        Rg_here.append(rg); AR_here.append(ar); Phi_here.append(phi)
    Rg_a[alpha]  = np.array(Rg_here)
    AR_a[alpha]  = np.array(AR_here)
    Phi_a[alpha] = np.array(Phi_here)

fs2 = max(1, n_it2 // n_fr2)
t2 = np.arange(n_fr2) * fs2 * PREY_DEFAULT['dt']

print('  Steady-state values (last 20% of frames):')
for alpha in alpha_vals:
    print(f'    alpha={alpha:.1f}  Rg={Rg_a[alpha][:,-40:].mean():.3f}  '
          f'AR={AR_a[alpha][:,-40:].mean():.2f}  '
          f'Phi={Phi_a[alpha][:,-40:].mean():.3f}')

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('Flock geometry vs flocking amplitude alpha  (with predator, 8 seeds)', fontsize=11)
for i, alpha in enumerate(alpha_vals):
    lbl = f'alpha={alpha}'
    band(axes[0], t2, Rg_a[alpha],  colors[i], lbl)
    band(axes[1], t2, AR_a[alpha],  colors[i], lbl)
    band(axes[2], t2, Phi_a[alpha], colors[i], lbl)
axes[0].set_xlabel('Time'); axes[0].set_ylabel('Radius of gyration')
axes[0].set_title('Flock spread'); axes[0].legend(fontsize=8)
axes[1].set_xlabel('Time'); axes[1].set_ylabel('Aspect ratio')
axes[1].set_title('Flock elongation'); axes[1].axhline(1, color='gray', ls='--', lw=0.8); axes[1].legend(fontsize=8)
axes[2].set_xlabel('Time'); axes[2].set_ylabel('Order parameter Phi')
axes[2].set_title('Flock coherence'); axes[2].set_ylim(0, 1); axes[2].legend(fontsize=8)
plt.tight_layout()
plt.savefig('figures/geometry_2_alpha_sweep.png', dpi=120)
plt.close()
print('  --> figures/geometry_2_alpha_sweep.png')


# =============================================================================
# EXPERIMENT 3: SNAPSHOT SEQUENCE SHOWING FLOCK SHAPE OVER TIME WITH PREDATOR
# =============================================================================
print('\nExp 3: Snapshot sequence -- flock shape evolution with predator')
pp3 = PREY_DEFAULT.copy(); pp3['n_iter'] = 8000; pp3['N'] = 150
f3 = run_predator(prey_overrides=pp3, seed=SEED, n_frames=400)
fs3 = max(1, 8000 // 400); dt3 = PREY_DEFAULT['dt']

rg3, ar3, phi3 = geometry_series_pred(f3)
dist3 = pred_dist_series(f3)
t3 = np.arange(400) * fs3 * dt3

fig, axes = plt.subplots(2, 4, figsize=(15, 7))
fig.suptitle('Predator-prey: flock shape evolution  (N=150)', fontsize=11)
times_snap = [0., 5., 10., 20., 30., 40., 60., 79.]
for ax, ts in zip(axes.flat, times_snap):
    fi = min(int(ts / (fs3 * dt3)), len(f3)-1)
    px, py, vx, vy, prdx, prdy, _, _ = f3[fi]
    sp = np.sqrt(vx**2+vy**2); sp[sp==0]=1.
    ax.scatter(px, py, s=3, color='steelblue', zorder=3)
    ax.quiver(px, py, vx/sp, vy/sp, scale=80, width=0.003, color='steelblue', alpha=0.4)
    ax.scatter([prdx], [prdy], s=150, color='crimson', marker='*', zorder=5)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    actual_t = fi * fs3 * dt3
    ax.set_title(f't={actual_t:.0f}  AR={ar3[fi]:.2f}  Rg={rg3[fi]:.3f}', fontsize=8)
plt.tight_layout()
plt.savefig('figures/geometry_3_snapshots.png', dpi=120)
plt.close()

# also plot geometry time series for this run
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
fig.suptitle('Geometry time series  (N=150, single run)', fontsize=10)
axes[0].plot(t3, rg3, color='steelblue'); axes[0].set_xlabel('Time'); axes[0].set_ylabel('Rg'); axes[0].set_title('Radius of gyration')
axes[1].plot(t3, ar3, color='seagreen');  axes[1].set_xlabel('Time'); axes[1].set_ylabel('AR'); axes[1].set_title('Aspect ratio')
axes[2].plot(t3, phi3, color='darkorange'); axes[2].set_xlabel('Time'); axes[2].set_ylabel('Phi'); axes[2].set_title('Order parameter'); axes[2].set_ylim(0,1)
axes[3].plot(t3, dist3, color='crimson'); axes[3].set_xlabel('Time'); axes[3].set_ylabel('Pred-prey dist'); axes[3].set_title('Predator proximity')
plt.tight_layout()
plt.savefig('figures/geometry_3_timeseries.png', dpi=120)
plt.close()
print('  --> figures/geometry_3_snapshots.png')
print('  --> figures/geometry_3_timeseries.png')

print('\nGeometry analysis complete.')
