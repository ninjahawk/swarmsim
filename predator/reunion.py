# reunion.py -- Sub-flock reunion after predator removal (Finding 16 follow-up)
#
# Finding 16 showed that encirclement (n_pred=6) divides the flock into
# coherent sub-flocks moving in different directions.  That experiment ended
# with the predators still active.  The open question: if predators are then
# REMOVED, do the sub-flocks reunite or do they remain permanently divided?
#
# Method
# ------
# Three-phase simulation:
#   Phase 1 (steps 0..n_warmup):    pure flock, no predators, settles to Phi~1
#   Phase 2 (n_warmup..n_attack):   add 6 encircling predators, fragment the flock
#   Phase 3 (n_attack..n_total):    remove all predators, observe recovery
#
# Metrics over time:
#   - Phi (global order parameter)
#   - n_clusters (DBSCAN-style: connected components within rf)
#   - largest_frac (fraction of agents in the largest cluster)
#
# Hypotheses:
#   H1: Sub-flocks reunite quickly (Phi -> 1 within a few thousand steps).
#       This would mean encirclement causes only TRANSIENT division.
#   H2: Sub-flocks remain permanently separated on the periodic torus,
#       each cruising in its own direction.  Would mean encirclement causes
#       PERMANENT division -- the topological state of the flock has changed.
#   H3: Partial recovery -- some sub-flocks merge but not all.

import os
import numpy as np
import matplotlib.pyplot as plt
from model import Flock, Predator, _periodic_disp
from flocking import order_parameter

os.makedirs('figures', exist_ok=True)

N_SEEDS = 6

# Phase timings (in steps)
# Use n_pred=10 (Finding 8 floor) and a longer attack to force genuine
# fragmentation before testing reunion.
N_PRED   = 10
N_WARMUP = 1500
N_ATTACK = 5500   # 4000 steps of attack
N_TOTAL  = 12000  # 6500 steps of recovery


def find_clusters(px, py, rf):
    """Connected-components clustering: two agents in same cluster iff
       periodic distance <= rf.  Returns label array of length N."""
    N = len(px)
    dx = px[np.newaxis, :] - px[:, np.newaxis]
    dy = py[np.newaxis, :] - py[:, np.newaxis]
    dx -= np.round(dx); dy -= np.round(dy)
    d2 = dx**2 + dy**2
    adj = (d2 <= rf**2) & (d2 > 0)

    labels = -np.ones(N, dtype=int)
    cur = 0
    for i in range(N):
        if labels[i] >= 0:
            continue
        stack = [i]
        while stack:
            j = stack.pop()
            if labels[j] >= 0:
                continue
            labels[j] = cur
            for k in np.where(adj[j])[0]:
                if labels[k] < 0:
                    stack.append(k)
        cur += 1
    return labels, cur


def run_one(seed):
    """Run one reunion experiment.  Returns dict of timeseries."""
    np.random.seed(seed)
    # Use the predator-regime prey defaults (matches encirclement.py / Finding 14-16):
    # slow prey (v0=0.02) so the v0=0.05 predator can actually pursue.
    flock = Flock(seed=seed, N=350, v0=0.02, ramp=0.1)
    preds = []
    record_every = 50

    phi_ts = []; n_clust_ts = []; lfrac_ts = []; t_ts = []
    for step in range(N_TOTAL):
        # Phase transitions
        if step == N_WARMUP:
            preds = [Predator(strategy='encircle', angle=k*360.0/N_PRED, enc_radius=0.15)
                     for k in range(N_PRED)]
        elif step == N_ATTACK:
            preds = []  # remove predators

        flock.evolve(predators=preds)

        if step % record_every == 0:
            phi_ts.append(flock.phi)
            labels, n_clust = find_clusters(flock.px, flock.py, flock.rf)
            sizes = np.bincount(labels)
            lfrac_ts.append(sizes.max() / flock.N)
            n_clust_ts.append(n_clust)
            t_ts.append(step * flock.dt)

    return dict(
        t=np.array(t_ts), phi=np.array(phi_ts),
        n_clust=np.array(n_clust_ts), lfrac=np.array(lfrac_ts),
    )


print('Reunion experiment: %d encircling predators applied steps %d..%d, removed for steps %d..%d'
      % (N_PRED, N_WARMUP, N_ATTACK, N_ATTACK, N_TOTAL))
print('  (%d seeds)' % N_SEEDS)

runs = []
for s in range(N_SEEDS):
    print('  seed %d ...' % s, flush=True)
    runs.append(run_one(s))

# Stack
t = runs[0]['t']
phi_arr   = np.array([r['phi'] for r in runs])
clust_arr = np.array([r['n_clust'] for r in runs])
lfrac_arr = np.array([r['lfrac'] for r in runs])

dt_val = 0.01
t_warmup_end = N_WARMUP * dt_val
t_attack_end = N_ATTACK * dt_val

# Steady-state windows:
#   pre-attack:    last 500 steps before t_warmup_end
#   end-of-attack: last 500 steps before t_attack_end
#   post-attack:   last 1000 steps of run
def window_mean(arr, t0, t1):
    mask = (t >= t0) & (t < t1)
    return arr[:, mask].mean(axis=1)

phi_pre    = window_mean(phi_arr,   t_warmup_end - 5.0, t_warmup_end)
phi_during = window_mean(phi_arr,   t_attack_end - 5.0, t_attack_end)
phi_post   = window_mean(phi_arr,   t[-1] - 10.0, t[-1] + 1)
clust_pre    = window_mean(clust_arr, t_warmup_end - 5.0, t_warmup_end)
clust_during = window_mean(clust_arr, t_attack_end - 5.0, t_attack_end)
clust_post   = window_mean(clust_arr, t[-1] - 10.0, t[-1] + 1)
lfrac_pre    = window_mean(lfrac_arr, t_warmup_end - 5.0, t_warmup_end)
lfrac_during = window_mean(lfrac_arr, t_attack_end - 5.0, t_attack_end)
lfrac_post   = window_mean(lfrac_arr, t[-1] - 10.0, t[-1] + 1)

print('\n=== Phase summary (mean +/- std across %d seeds) ===' % N_SEEDS)
print('Phi:        pre=%.3f +/- %.3f   during=%.3f +/- %.3f   post=%.3f +/- %.3f' % (
    phi_pre.mean(), phi_pre.std(), phi_during.mean(), phi_during.std(),
    phi_post.mean(), phi_post.std()))
print('n_clusters: pre=%.1f   during=%.1f   post=%.1f' % (
    clust_pre.mean(), clust_during.mean(), clust_post.mean()))
print('largest_frac: pre=%.3f   during=%.3f   post=%.3f' % (
    lfrac_pre.mean(), lfrac_during.mean(), lfrac_post.mean()))

# Recovery: how long after t_attack_end does Phi cross e.g. 0.95?
phi_mean = phi_arr.mean(axis=0)
post_idx = np.where(t > t_attack_end)[0]
rec_times = []
for r in runs:
    post = r['phi'][post_idx]
    crossings = np.where(post >= 0.95)[0]
    if crossings.size:
        rec_times.append((post_idx[crossings[0]] - post_idx[0]) * (t[1]-t[0]))
    else:
        rec_times.append(np.nan)
print('Recovery time to Phi=0.95: %s' % rec_times)
rec_clean = [r for r in rec_times if not np.isnan(r)]
if rec_clean:
    print('  mean recovery time: %.1f time units (%d / %d seeds recovered)' %
          (np.mean(rec_clean), len(rec_clean), N_SEEDS))
else:
    print('  no seed recovered to Phi=0.95 within the simulation window')

# Plot timeseries
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
for ax in axes:
    ax.axvspan(t_warmup_end, t_attack_end, color='red', alpha=0.10,
               label='predator attack')

ax = axes[0]
for r in runs:
    ax.plot(r['t'], r['phi'], color='steelblue', alpha=0.25, lw=0.8)
ax.plot(t, phi_arr.mean(0), color='navy', lw=2, label='mean')
ax.set_ylabel('Global Phi'); ax.set_ylim(0, 1.05)
ax.set_title('Sub-flock reunion after predator removal (%d seeds)' % N_SEEDS)
ax.legend(fontsize=8, loc='lower right')

ax = axes[1]
for r in runs:
    ax.plot(r['t'], r['n_clust'], color='seagreen', alpha=0.25, lw=0.8)
ax.plot(t, clust_arr.mean(0), color='darkgreen', lw=2, label='mean')
ax.set_ylabel('n_clusters')
ax.legend(fontsize=8, loc='upper right')

ax = axes[2]
for r in runs:
    ax.plot(r['t'], r['lfrac'], color='darkorange', alpha=0.25, lw=0.8)
ax.plot(t, lfrac_arr.mean(0), color='chocolate', lw=2, label='mean')
ax.set_ylabel('largest cluster fraction')
ax.set_xlabel('Time')
ax.set_ylim(0, 1.05)
ax.legend(fontsize=8, loc='lower right')

plt.tight_layout()
plt.savefig('figures/reunion_1_timeseries.png', dpi=120)
plt.close()
print('  --> figures/reunion_1_timeseries.png')

# Bar summary
fig, ax = plt.subplots(figsize=(8, 5))
phases = ['pre-attack', 'during', 'post-attack']
phi_m  = [phi_pre.mean(), phi_during.mean(), phi_post.mean()]
phi_s  = [phi_pre.std(),  phi_during.std(),  phi_post.std()]
lf_m   = [lfrac_pre.mean(), lfrac_during.mean(), lfrac_post.mean()]
lf_s   = [lfrac_pre.std(),  lfrac_during.std(),  lfrac_post.std()]
x = np.arange(3)
ax.bar(x - 0.18, phi_m, 0.34, yerr=phi_s, color='steelblue', label='Global Phi', capsize=5)
ax.bar(x + 0.18, lf_m, 0.34, yerr=lf_s, color='darkorange', label='Largest cluster frac', capsize=5)
ax.set_xticks(x); ax.set_xticklabels(phases)
ax.set_ylim(0, 1.05); ax.set_ylabel('value')
ax.set_title('Phase comparison: does the flock recover?')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig('figures/reunion_2_summary.png', dpi=120)
plt.close()
print('  --> figures/reunion_2_summary.png')

print('\nReunion analysis complete.')
