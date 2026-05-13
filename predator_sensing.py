# predator_sensing.py -- Effect of limited predator sensing range
#
# So far predators have perfect, infinite knowledge of prey position (they always
# know the exact flock CoM). This is biologically unrealistic. Real predators can
# only see/smell/hear prey within some range.
#
# Model: predator k has sensing radius r_sense. Within r_sense, it locks on to
# the nearest prey agent. Outside r_sense, it executes a biased random walk
# (slow drift in current direction + noise). This creates qualitatively different
# hunting phases: search, approach, attack.
#
# Experiments:
#   1. Sensing radius sweep (r_sense=0.05..inf, single predator) -- how does
#      min pred-prey distance and coherence change with sensing range?
#   2. Time series: does limited sensing create boom-bust hunting cycles?
#      (predator loses the flock, wanders, relocates, attacks, flock recovers)
#   3. Multi-predator with limited sensing vs perfect sensing (n_pred=3,6)
#   4. Comparison table: naive-perfect, naive-limited, encirclement-limited

import os
import numpy as np
import matplotlib.pyplot as plt
from flocking import params, buffer, force, order_parameter
from predator import PREY_DEFAULT, PRED_DEFAULT

os.makedirs('figures', exist_ok=True)
N_SEEDS = 8
INF_SENSE = 1e9   # sentinel for "perfect sensing"


def run_sensing(n_pred=1, r_sense=0.3, encircle=False, R_enc=0.15,
                prey_overrides=None, pred_overrides=None,
                n_frames=200, seed=None):
    """
    Multi-predator simulation with limited sensing radius.

    Within r_sense of the nearest prey agent, predator locks on to prey CoM.
    Outside r_sense, predator does a biased random walk (keeps current heading,
    adds noise) at reduced speed -- searching, not chasing.

    encircle=True: use angular offset targets (like encirclement.py) when locked on.
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

    pred_x  = np.random.uniform(0., 1., n_pred)
    pred_y  = np.random.uniform(0., 1., n_pred)
    pred_vx = np.zeros(n_pred)
    pred_vy = np.zeros(n_pred)

    frames = []
    locked = np.zeros(n_pred, dtype=bool)   # track lock-on state per predator

    for i in range(p['n_iter']):
        rb = max(p['r0'], p['rf'])
        nb, xb, yb, vxb, vyb = buffer(rb, x, vx, vy, N)
        fx, fy = force(nb, xb, yb, vxb, vyb, x, vx, vy, p)

        # repulsion from each predator onto prey (only if locked on)
        for ip in range(n_pred):
            if not locked[ip]:
                continue
            for j in range(N):
                ddx = pred_x[ip] - x[j];   ddx -= round(ddx)
                ddy = pred_y[ip] - x[N+j]; ddy -= round(ddy)
                d = np.sqrt(ddx**2 + ddy**2)
                if 0 < d <= pd['r0_pred']:
                    s = pd['eps_pred'] * (1. - d/pd['r0_pred'])**1.5 / d
                    fx[j] -= s * ddx
                    fy[j] -= s * ddy

        # prey CoM
        cx = np.arctan2(np.sin(2*np.pi*x[:N]).mean(),
                        np.cos(2*np.pi*x[:N]).mean()) / (2*np.pi) % 1.
        cy = np.arctan2(np.sin(2*np.pi*x[N:]).mean(),
                        np.cos(2*np.pi*x[N:]).mean()) / (2*np.pi) % 1.

        for ip in range(n_pred):
            # check if any prey is within sensing radius
            ddx_all = x[:N] - pred_x[ip]; ddx_all -= np.round(ddx_all)
            ddy_all = x[N:] - pred_y[ip]; ddy_all -= np.round(ddy_all)
            d_all = np.sqrt(ddx_all**2 + ddy_all**2)
            locked[ip] = d_all.min() <= r_sense

            if locked[ip]:
                # locked on: chase CoM (or offset for encirclement)
                if encircle and n_pred > 1:
                    tx_tgt = (cx + R_enc * np.cos(angles[ip])) % 1.
                    ty_tgt = (cy + R_enc * np.sin(angles[ip])) % 1.
                else:
                    tx_tgt = cx
                    ty_tgt = cy
                tx = tx_tgt - pred_x[ip]; tx -= round(tx)
                ty = ty_tgt - pred_y[ip]; ty -= round(ty)
                alpha_eff = pd['alpha_pred']
            else:
                # searching: slow random walk in current direction
                speed = np.sqrt(pred_vx[ip]**2 + pred_vy[ip]**2)
                if speed > 0:
                    tx = pred_vx[ip] / speed
                    ty = pred_vy[ip] / speed
                else:
                    tx = np.random.uniform(-1., 1.)
                    ty = np.random.uniform(-1., 1.)
                    dn = np.sqrt(tx**2+ty**2)
                    if dn > 0: tx /= dn; ty /= dn
                alpha_eff = pd['alpha_pred'] * 0.3   # slower while searching

            dist = np.sqrt(tx**2 + ty**2)
            if dist > 0:
                tx /= dist; ty /= dist

            sp = np.sqrt(pred_vx[ip]**2 + pred_vy[ip]**2)
            v0_eff = pd['v0_pred'] if locked[ip] else pd['v0_pred'] * 0.4
            pfx = alpha_eff * tx
            pfy = alpha_eff * ty
            if sp > 0:
                pfx += pd['mu_pred'] * (v0_eff - sp) * pred_vx[ip]/sp
                pfy += pd['mu_pred'] * (v0_eff - sp) * pred_vy[ip]/sp
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
                           pred_x.copy(), pred_y.copy(),
                           locked.copy()))

    return frames


def phi_series(frames):
    return np.array([order_parameter(vx, vy) for _, _, vx, vy, _, _, _ in frames])

def dist_series(frames):
    dists = []
    for px, py, _, _, pxs, pys, locked in frames:
        locked_preds = np.where(locked)[0]
        if len(locked_preds) == 0:
            dists.append(np.nan)
            continue
        min_d = np.inf
        for ip in locked_preds:
            ddx = pxs[ip] - px; ddx -= np.round(ddx)
            ddy = pys[ip] - py; ddy -= np.round(ddy)
            d = np.sqrt(ddx**2+ddy**2).min()
            min_d = min(min_d, d)
        dists.append(min_d)
    return np.array(dists)

def lock_fraction(frames):
    """Fraction of time at least one predator is locked on."""
    return np.array([f[-1].any() for f in frames], dtype=float)


# =============================================================================
# EXP 1: SENSING RADIUS SWEEP  (single predator)
# =============================================================================
print('Exp 1: Sensing radius sweep  (n_pred=1, %d seeds)' % N_SEEDS)
r_sense_vals = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50, INF_SENSE]
r_labels     = ['0.05', '0.10', '0.15', '0.20', '0.30', '0.50', 'inf']
n_it = 5000; n_fr = 250
pp_e1 = PREY_DEFAULT.copy(); pp_e1['n_iter'] = n_it

res1 = {}
for r_sense, rlbl in zip(r_sense_vals, r_labels):
    phi_ss=[]; dist_ss=[]; lock_ss=[]
    for s in range(N_SEEDS):
        f = run_sensing(n_pred=1, r_sense=r_sense, prey_overrides=pp_e1,
                        n_frames=n_fr, seed=s)
        phi_ss.append(phi_series(f)[-50:].mean())
        d = dist_series(f)[-50:]
        dist_ss.append(np.nanmean(d))
        lock_ss.append(lock_fraction(f).mean())
    res1[rlbl] = dict(phi=np.mean(phi_ss), phi_std=np.std(phi_ss),
                      dist=np.nanmean(dist_ss), lock=np.mean(lock_ss))
    print('  r_sense=%s  Phi=%.3f +/- %.3f  dist=%.3f  lock_frac=%.2f' % (
        rlbl, res1[rlbl]['phi'], res1[rlbl]['phi_std'],
        res1[rlbl]['dist'], res1[rlbl]['lock']))


# =============================================================================
# EXP 2: TIME SERIES -- does limited sensing create hunting cycles?
# =============================================================================
print('\nExp 2: Hunting cycles time series  (r_sense=0.20 vs inf, %d seeds)' % N_SEEDS)
ts_results = {}
for r_sense, lbl in [(0.20, '0.20'), (INF_SENSE, 'inf')]:
    phi_runs=[]; lock_runs=[]; dist_runs=[]
    for s in range(N_SEEDS):
        f = run_sensing(n_pred=1, r_sense=r_sense, prey_overrides=pp_e1,
                        n_frames=n_fr, seed=s)
        phi_runs.append(phi_series(f))
        lock_runs.append(lock_fraction(f))
        dist_runs.append(dist_series(f))
    ts_results[lbl] = dict(phi=np.array(phi_runs),
                            lock=np.array(lock_runs),
                            dist=np.array(dist_runs))

fs1 = max(1, n_it // n_fr)
t1 = np.arange(n_fr) * fs1 * PREY_DEFAULT['dt']


# =============================================================================
# EXP 3: MULTI-PREDATOR WITH LIMITED SENSING  (n_pred=3 and 6)
# =============================================================================
print('\nExp 3: Multi-predator with limited sensing  (%d seeds)' % N_SEEDS)
compare = [(1,'inf',INF_SENSE,False),
           (1,'0.20',0.20,False),
           (3,'inf',INF_SENSE,False),
           (3,'0.20',0.20,False),
           (6,'inf',INF_SENSE,True),    # encirclement with perfect sensing
           (6,'0.20',0.20,True)]         # encirclement with limited sensing

res3 = {}
for n_pred, rlbl, r_sense, enc in compare:
    phi_ss=[]; dist_ss=[]
    for s in range(N_SEEDS):
        f = run_sensing(n_pred=n_pred, r_sense=r_sense, encircle=enc,
                        prey_overrides=pp_e1, n_frames=n_fr, seed=s)
        phi_ss.append(phi_series(f)[-50:].mean())
        d = dist_series(f)[-50:]
        dist_ss.append(np.nanmean(d))
    key = (n_pred, rlbl, r_sense, enc)
    res3[key] = dict(phi=np.mean(phi_ss), phi_std=np.std(phi_ss),
                     dist=np.nanmean(dist_ss))
    print('  n_pred=%d  r_sense=%s  enc=%s  Phi=%.3f +/- %.3f  dist=%.3f' % (
        n_pred, rlbl, enc, res3[key]['phi'], res3[key]['phi_std'], res3[key]['dist']))


# =============================================================================
# FIGURES
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Predator sensing radius effects  (%d seeds)' % N_SEEDS, fontsize=11)

# Exp 1: phi and lock fraction vs r_sense
r_x = list(range(len(r_labels)))
axes[0].errorbar(r_x, [res1[l]['phi'] for l in r_labels],
                 yerr=[res1[l]['phi_std'] for l in r_labels],
                 fmt='o-', color='steelblue', lw=2, capsize=5, label='Phi')
ax2 = axes[0].twinx()
ax2.plot(r_x, [res1[l]['lock'] for l in r_labels],
         's--', color='crimson', lw=2, label='Lock fraction')
ax2.set_ylabel('Lock-on fraction', color='crimson')
axes[0].set_xticks(r_x); axes[0].set_xticklabels(r_labels)
axes[0].set_xlabel('Sensing radius'); axes[0].set_ylabel('Phi')
axes[0].set_title('Coherence and lock-on vs sensing range'); axes[0].set_ylim(0,1)

# Exp 2: time series
colors_ts = {'0.20': 'darkorange', 'inf': 'steelblue'}
for lbl, arr in ts_results.items():
    phi_m = np.nanmean(arr['phi'], axis=0)
    phi_s = np.nanstd(arr['phi'], axis=0)
    lock_m = arr['lock'].mean(axis=0)
    axes[1].plot(t1, phi_m, color=colors_ts[lbl], lw=2, label='r=%s Phi' % lbl)
    axes[1].fill_between(t1, phi_m-phi_s, phi_m+phi_s, color=colors_ts[lbl], alpha=0.15)
axes[1].set_xlabel('Time'); axes[1].set_ylabel('Phi')
axes[1].set_title('Coherence time series (n_pred=1)'); axes[1].set_ylim(0,1)
axes[1].legend(fontsize=8)

# Exp 3: grouped bar chart
labels3 = ['1-inf', '1-0.20', '3-inf', '3-0.20', '6-enc-inf', '6-enc-0.20']
phi_vals = [res3[k]['phi'] for k in compare]
phi_errs = [res3[k]['phi_std'] for k in compare]
bar_colors = ['steelblue','cornflowerblue','seagreen','mediumseagreen','crimson','lightsalmon']
x3 = np.arange(len(labels3))
axes[2].bar(x3, phi_vals, yerr=phi_errs, color=bar_colors, capsize=5, width=0.6)
axes[2].set_xticks(x3); axes[2].set_xticklabels(labels3, rotation=30, ha='right', fontsize=8)
axes[2].set_ylabel('Phi'); axes[2].set_title('n_pred × sensing × strategy comparison')
axes[2].set_ylim(0,1)
axes[2].axhline(0.77, color='gray', ls='--', lw=0.8, label='Encircle-perfect baseline')
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig('figures/sensing_1_summary.png', dpi=120)
plt.close()
print('  --> figures/sensing_1_summary.png')

# Time series detail figure
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.suptitle('Hunting cycles: limited (r=0.20) vs perfect sensing  (%d seeds)' % N_SEEDS,
             fontsize=11)
for lbl in ['0.20', 'inf']:
    arr = ts_results[lbl]
    phi_m = np.nanmean(arr['phi'], axis=0)
    phi_s = np.nanstd(arr['phi'], axis=0)
    lock_m = arr['lock'].mean(axis=0)
    axes[0].plot(t1, phi_m, color=colors_ts[lbl], lw=2, label='r_sense=%s' % lbl)
    axes[0].fill_between(t1, phi_m-phi_s, phi_m+phi_s, color=colors_ts[lbl], alpha=0.15)
    if lbl == '0.20':
        axes[1].plot(t1, lock_m, color=colors_ts[lbl], lw=2, label='Lock-on fraction')
axes[0].set_ylabel('Order parameter Phi'); axes[0].set_ylim(0,1); axes[0].legend(fontsize=8)
axes[0].set_title('Flock coherence')
axes[1].set_xlabel('Time'); axes[1].set_ylabel('Lock-on fraction (r=0.20)')
axes[1].set_title('Predator lock-on status over time')
axes[1].set_ylim(0,1)
plt.tight_layout()
plt.savefig('figures/sensing_2_cycles.png', dpi=120)
plt.close()
print('  --> figures/sensing_2_cycles.png')

print('\nPredator sensing analysis complete.')
