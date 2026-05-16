# fragmentation.py -- What does the flock actually look like when Phi drops to 0.77?
#
# Finding 14 showed encirclement with n_pred=6 gives Phi=0.769 with high variance.
# Phi is a single number -- it can't distinguish between:
#   (a) one struggling-but-intact flock
#   (b) two or more sub-flocks moving in different directions
#   (c) total dissolution into random walkers
#
# This script detects and characterizes sub-flocks using spatial clustering:
# two agents are in the same cluster if periodic distance <= 4*r0. Connected
# components via union-find on a vectorized adjacency matrix.
#
# Experiments:
#   1. Fragmentation time series: naive vs encirclement at n_pred=6
#   2. Cluster statistics vs n_pred: how do fragment count and largest-cluster
#      fraction change as we scale up encircling predators?
#   3. Snapshot gallery: spatial maps at several time points, colored by cluster

import os
import numpy as np
import matplotlib.pyplot as plt
from flocking import params, buffer, force, order_parameter
from predator import PREY_DEFAULT, PRED_DEFAULT

os.makedirs('figures', exist_ok=True)
N_SEEDS = 8
R_ENC   = 0.15


# =============================================================================
# SIMULATION RUNNERS (self-contained -- no imports from experiment scripts)
# =============================================================================
def _run_predators(n_pred, angles_fn, prey_overrides, pred_overrides, n_frames, seed):
    """
    Generic multi-predator runner. angles_fn(ip, cx, cy) -> (tx_target, ty_target).
    Used by both naive and encirclement runners below.
    """
    if seed is not None:
        np.random.seed(seed)

    pp = PREY_DEFAULT.copy()
    if prey_overrides:
        pp.update(prey_overrides)
    pd = PRED_DEFAULT.copy()
    if pred_overrides:
        pd.update(pred_overrides)

    p  = params(pp)
    N  = p['N'];  dt = p['dt']
    frame_every = max(1, p['n_iter'] // n_frames)

    x   = np.zeros(2*N)
    x[:N] = np.random.uniform(0., 1., N)
    x[N:] = np.random.uniform(0., 1., N)
    vx  = np.random.uniform(-1., 1., N) * p['v0']
    vy  = np.random.uniform(-1., 1., N) * p['v0']

    pred_x  = np.random.uniform(0., 1., n_pred)
    pred_y  = np.random.uniform(0., 1., n_pred)
    pred_vx = np.zeros(n_pred)
    pred_vy = np.zeros(n_pred)

    frames = []
    for i in range(p['n_iter']):
        rb = max(p['r0'], p['rf'])
        nb, xb, yb, vxb, vyb = buffer(rb, x, vx, vy, N)
        fx, fy = force(nb, xb, yb, vxb, vyb, x, vx, vy, p)

        for ip in range(n_pred):
            for j in range(N):
                ddx = pred_x[ip] - x[j];   ddx -= round(ddx)
                ddy = pred_y[ip] - x[N+j]; ddy -= round(ddy)
                d = np.sqrt(ddx**2 + ddy**2)
                if 0 < d <= pd['r0_pred']:
                    s = pd['eps_pred'] * (1. - d/pd['r0_pred'])**1.5 / d
                    fx[j] -= s * ddx
                    fy[j] -= s * ddy

        cx = np.arctan2(np.sin(2*np.pi*x[:N]).mean(),
                        np.cos(2*np.pi*x[:N]).mean()) / (2*np.pi) % 1.
        cy = np.arctan2(np.sin(2*np.pi*x[N:]).mean(),
                        np.cos(2*np.pi*x[N:]).mean()) / (2*np.pi) % 1.

        for ip in range(n_pred):
            tx_t, ty_t = angles_fn(ip, cx, cy)
            tx = tx_t - pred_x[ip]; tx -= round(tx)
            ty = ty_t - pred_y[ip]; ty -= round(ty)
            dist = np.sqrt(tx**2 + ty**2)
            if dist > 0: tx /= dist; ty /= dist
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


def run_naive(n_pred=2, prey_overrides=None, pred_overrides=None,
              n_frames=200, seed=None):
    def target(ip, cx, cy): return cx, cy
    return _run_predators(n_pred, target, prey_overrides, pred_overrides, n_frames, seed)


def run_encircle(n_pred=3, R_enc=0.15, prey_overrides=None, pred_overrides=None,
                 n_frames=200, seed=None):
    angles = np.array([2*np.pi*k/n_pred for k in range(n_pred)])
    def target(ip, cx, cy):
        return ((cx + R_enc*np.cos(angles[ip])) % 1.,
                (cy + R_enc*np.sin(angles[ip])) % 1.)
    return _run_predators(n_pred, target, prey_overrides, pred_overrides, n_frames, seed)


def geom_stats(frames):
    Phi = np.array([order_parameter(vx, vy) for _, _, vx, vy, _, _ in frames])
    return Phi


def mean_min_dist(frames):
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
# CLUSTERING  (vectorized union-find on periodic distance matrix)
# =============================================================================
def find_clusters(px, py, r_cluster=None):
    if r_cluster is None:
        r_cluster = 4 * PREY_DEFAULT['r0']
    N = len(px)

    # vectorized periodic distance matrix
    dx = px[:, np.newaxis] - px[np.newaxis, :]
    dy = py[:, np.newaxis] - py[np.newaxis, :]
    dx -= np.round(dx); dy -= np.round(dy)
    adj = (dx**2 + dy**2) <= r_cluster**2
    np.fill_diagonal(adj, False)

    # union-find
    parent = np.arange(N)
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(N):
        for j in np.where(adj[i])[0]:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[ri] = rj

    roots = np.array([find(i) for i in range(N)])
    _, label = np.unique(roots, return_inverse=True)
    return label, len(np.unique(label))


def cluster_stats(px, py, vx, vy):
    label, n_cl = find_clusters(px, py)
    N = len(px)
    sizes = np.bincount(label, minlength=n_cl)
    largest = np.argmax(sizes)
    largest_frac = sizes[largest] / N
    phi_w = sum(order_parameter(vx[label==c], vy[label==c]) * sizes[c]/N
                for c in range(n_cl))
    largest_phi = order_parameter(vx[label==largest], vy[label==largest])
    return dict(n_clusters=n_cl, largest_frac=largest_frac,
                mean_phi=phi_w, largest_phi=largest_phi, labels=label)


# =============================================================================
# EXP 1: FRAGMENTATION TIME SERIES  (naive vs encirclement, n_pred=6)
# =============================================================================
print('Exp 1: Fragmentation time series  (naive vs encirclement, n_pred=6, %d seeds)' % N_SEEDS)
n_it = 5000; n_fr = 250
pp_e1 = PREY_DEFAULT.copy(); pp_e1['n_iter'] = n_it

def run_and_cluster(frames_list):
    nc_t=[]; lf_t=[]; lphi_t=[]; phi_t=[]
    for px, py, vx, vy, _, _ in frames_list:
        cs = cluster_stats(px, py, vx, vy)
        nc_t.append(cs['n_clusters']); lf_t.append(cs['largest_frac'])
        lphi_t.append(cs['largest_phi']); phi_t.append(order_parameter(vx, vy))
    return nc_t, lf_t, lphi_t, phi_t

modes = {tag: dict(nc=[], lf=[], lphi=[], phi=[]) for tag in ['naive','encircle']}

for s in range(N_SEEDS):
    fn = run_naive(n_pred=6, prey_overrides=pp_e1, n_frames=n_fr, seed=s)
    fe = run_encircle(n_pred=6, R_enc=R_ENC, prey_overrides=pp_e1, n_frames=n_fr, seed=s)
    for tag, frames in [('naive', fn), ('encircle', fe)]:
        nc_t, lf_t, lphi_t, phi_t = run_and_cluster(frames)
        modes[tag]['nc'].append(nc_t); modes[tag]['lf'].append(lf_t)
        modes[tag]['lphi'].append(lphi_t); modes[tag]['phi'].append(phi_t)

for tag in modes:
    for k in modes[tag]:
        modes[tag][k] = np.array(modes[tag][k])

print('  Steady-state (last 20%% of frames):')
for tag in ['naive', 'encircle']:
    nc_ss   = modes[tag]['nc'][:,-50:].mean()
    lf_ss   = modes[tag]['lf'][:,-50:].mean()
    phi_ss  = modes[tag]['phi'][:,-50:].mean()
    lphi_ss = modes[tag]['lphi'][:,-50:].mean()
    print('  %s:  n_clusters=%.1f  largest_frac=%.3f  '
          'global_phi=%.3f  largest_phi=%.3f' % (
          tag, nc_ss, lf_ss, phi_ss, lphi_ss))

dt_val = PREY_DEFAULT['dt']
fs = max(1, n_it // n_fr)
t = np.arange(n_fr) * fs * dt_val

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle('Flock fragmentation: naive vs encirclement  (n_pred=6, %d seeds)' % N_SEEDS,
             fontsize=11)
colors = {'naive': 'steelblue', 'encircle': 'darkorange'}
labels = {'naive': 'Naive (CoM chase)', 'encircle': 'Encirclement (R=0.15)'}
for tag in ['naive', 'encircle']:
    c = colors[tag]; lbl = labels[tag]
    for ax, key, ylabel, title, ylim in zip(
            axes.flat,
            ['phi','nc','lf','lphi'],
            ['Global Phi','Number of clusters','Largest cluster fraction','Largest cluster Phi'],
            ['Global flock coherence','Fragment count',
             'Largest fragment size','Largest fragment coherence'],
            [(0,1),None,(0,1),(0,1)]):
        m = modes[tag][key].mean(0); s_ = modes[tag][key].std(0)
        ax.plot(t, m, color=c, lw=2, label=lbl)
        ax.fill_between(t, m-s_, m+s_, color=c, alpha=0.15)
        ax.set_xlabel('Time'); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(fontsize=8)
        if ylim: ax.set_ylim(*ylim)
plt.tight_layout()
plt.savefig('figures/frag_1_timeseries.png', dpi=120)
plt.close()
print('  --> figures/frag_1_timeseries.png')


# =============================================================================
# EXP 2: CLUSTER STATISTICS vs n_pred  (encirclement, steady state)
# =============================================================================
print('\nExp 2: Cluster stats vs n_pred  (encirclement, %d seeds)' % N_SEEDS)
n_pred_vals = [1, 2, 3, 4, 6, 8]
pp_e2 = PREY_DEFAULT.copy(); pp_e2['n_iter'] = n_it

res2 = {}
for n_pred in n_pred_vals:
    nc_s=[]; lf_s=[]; phi_s=[]; lphi_s=[]
    for s in range(N_SEEDS):
        f = run_encircle(n_pred=n_pred, R_enc=R_ENC,
                         prey_overrides=pp_e2, n_frames=n_fr, seed=s)
        nc_t=[]; lf_t=[]; lphi_t=[]; phi_t=[]
        for px, py, vx, vy, _, _ in f:
            cs = cluster_stats(px, py, vx, vy)
            nc_t.append(cs['n_clusters']); lf_t.append(cs['largest_frac'])
            lphi_t.append(cs['largest_phi']); phi_t.append(order_parameter(vx, vy))
        nc_s.append(np.mean(nc_t[-50:])); lf_s.append(np.mean(lf_t[-50:]))
        lphi_s.append(np.mean(lphi_t[-50:])); phi_s.append(np.mean(phi_t[-50:]))
    res2[n_pred] = dict(nc=np.mean(nc_s), nc_std=np.std(nc_s),
                        lf=np.mean(lf_s), lf_std=np.std(lf_s),
                        phi=np.mean(phi_s), lphi=np.mean(lphi_s))
    print('  n_pred=%2d  n_clusters=%.1f  largest_frac=%.3f  '
          'global_phi=%.3f  largest_phi=%.3f' % (
          n_pred, res2[n_pred]['nc'], res2[n_pred]['lf'],
          res2[n_pred]['phi'], res2[n_pred]['lphi']))

fig, axes = plt.subplots(1, 3, figsize=(14, 5))
fig.suptitle('Fragment structure vs encircling predator count  (%d seeds)' % N_SEEDS, fontsize=11)
x = n_pred_vals
axes[0].errorbar(x, [res2[n]['nc'] for n in x], yerr=[res2[n]['nc_std'] for n in x],
                 fmt='o-', color='steelblue', lw=2, capsize=5)
axes[0].set_xlabel('Number of predators'); axes[0].set_ylabel('Mean cluster count')
axes[0].set_title('Fragment count')
axes[1].errorbar(x, [res2[n]['lf'] for n in x], yerr=[res2[n]['lf_std'] for n in x],
                 fmt='s-', color='seagreen', lw=2, capsize=5)
axes[1].set_xlabel('Number of predators'); axes[1].set_ylabel('Fraction of agents')
axes[1].set_title('Largest fragment size'); axes[1].set_ylim(0, 1)
axes[2].plot(x, [res2[n]['phi']  for n in x], 'o-', color='crimson', lw=2, label='Global Phi')
axes[2].plot(x, [res2[n]['lphi'] for n in x], 's--', color='darkorange', lw=2,
             label='Largest cluster Phi')
axes[2].set_xlabel('Number of predators'); axes[2].set_ylabel('Order parameter')
axes[2].set_title('Global vs within-cluster coherence'); axes[2].set_ylim(0, 1)
axes[2].legend(fontsize=8)
plt.tight_layout()
plt.savefig('figures/frag_2_cluster_stats.png', dpi=120)
plt.close()
print('  --> figures/frag_2_cluster_stats.png')


# =============================================================================
# EXP 3: SNAPSHOT GALLERY  (n_pred=6, encirclement, N=200)
# =============================================================================
print('\nExp 3: Snapshot gallery  (n_pred=6, encirclement, N=200, seed=3)')
pp_e3 = PREY_DEFAULT.copy(); pp_e3['n_iter'] = 6000; pp_e3['N'] = 200
f3 = run_encircle(n_pred=6, R_enc=R_ENC, prey_overrides=pp_e3, n_frames=300, seed=3)
fs3 = max(1, 6000 // 300)

snap_times = [0., 5., 10., 15., 20., 30., 40., 59.]
fig, axes = plt.subplots(2, 4, figsize=(15, 7))
fig.suptitle('Flock fragmentation snapshots  (N=200, n_pred=6, encirclement)', fontsize=11)
cluster_colors = plt.cm.tab10.colors
for ax, ts in zip(axes.flat, snap_times):
    fi = min(int(ts / (fs3 * PREY_DEFAULT['dt'])), len(f3)-1)
    px, py, vx, vy, pxs, pys = f3[fi]
    label, n_cl = find_clusters(px, py)
    sizes = np.bincount(label, minlength=n_cl)
    for cid in range(n_cl):
        mask = label == cid
        col = 'lightgray' if sizes[cid] == 1 else cluster_colors[cid % len(cluster_colors)]
        ax.scatter(px[mask], py[mask], s=5 if sizes[cid]>1 else 2, color=col, zorder=3)
    ax.scatter(pxs, pys, s=180, color='red', marker='*', zorder=5)
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    cs = cluster_stats(px, py, vx, vy)
    ax.set_title('t=%.0f  nc=%d  lf=%.2f\nPhi=%.2f' % (
        fi*fs3*PREY_DEFAULT['dt'], cs['n_clusters'],
        cs['largest_frac'], order_parameter(vx,vy)), fontsize=8)
plt.tight_layout()
plt.savefig('figures/frag_3_snapshots.png', dpi=120)
plt.close()
print('  --> figures/frag_3_snapshots.png')

print('\nFragmentation analysis complete.')
