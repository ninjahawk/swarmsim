# decision_time.py -- HOW FAST does the flock decide? (time-resolved leadership)
#
# F72-F74 all measured only the steady-state heading, averaged over a late window. They say
# nothing about the DYNAMICS of the decision -- how quickly the flock turns to follow its
# leaders, and whether a hard decision takes longer to resolve. This experiment records the
# full heading time series and extracts a response/commitment time.
#
# Part 1 -- RESPONSE TIME vs informed fraction (single leader toward +x). How long until the
#           flock heading settles onto the goal direction, as a function of rho? Speed-accuracy:
#           do more leaders make the flock both faster AND more accurate (no tradeoff), and how
#           does noise (ramp) affect speed vs accuracy?
# Part 2 -- COMMITMENT TIME vs conflict angle (two equal opposed subgroups, F73 geometry).
#           Near the compromise->consensus boundary (~90-120 deg, F73) the flock must break a
#           symmetry to commit. Does the decision exhibit CRITICAL SLOWING -- a peak in the time
#           to settle -- near that boundary, as a bistable system does near its bifurcation?
#
# Timing metric: settle_time = first time (in tu) the heading enters and STAYS within a
# tolerance band (TOL_DEG) of its final value (mean heading over the last SETTLE_WIN steps),
# for the remainder of the run. Measured from t=0 (no warmup discarded for timing).

import os
import numpy as np
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from flocking import buffer as buf_fn, order_parameter

os.makedirs('figures', exist_ok=True)
N_SEEDS = 4

BASE = dict(N=350, r0=0.005, eps=0.1, rf=0.1,
            alpha=1.0, v0=1.0, mu=10.0, ramp=0.5, dt=0.01)
N_ITER    = 5000
TOL_DEG   = 15.0     # heading must settle within this band of its final value
SETTLE_WIN = 800     # final window (steps) used to define the settled heading


def ang_diff(a, b):
    """Smallest absolute angular difference (deg) between two headings."""
    d = (a - b + 180.) % 360. - 180.
    return abs(d)


def run(theta_deg, rho1, rho2, w_lead, seed, ramp=None):
    """Two subgroups: rho1 -> +x, rho2 -> rotated by theta. Returns heading time series."""
    rng = np.random.RandomState(seed)
    p = BASE.copy()
    if ramp is not None:
        p['ramp'] = ramp
    N = p['N']; dt = p['dt']
    r0, eps, rf = p['r0'], p['eps'], p['rf']
    v0, mu = p['v0'], p['mu']

    g1 = np.array([1.0, 0.0])
    th = np.radians(theta_deg)
    g2 = np.array([np.cos(th), np.sin(th)])

    n1 = int(round(rho1 * N)); n2 = int(round(rho2 * N))
    perm = rng.permutation(N)
    grp1 = np.zeros(N, dtype=bool); grp2 = np.zeros(N, dtype=bool)
    grp1[perm[:n1]] = True
    grp2[perm[n1:n1 + n2]] = True

    x  = np.zeros(2 * N)
    x[:N] = rng.uniform(0., 1., N); x[N:] = rng.uniform(0., 1., N)
    vx = rng.uniform(-1., 1., N) * v0
    vy = rng.uniform(-1., 1., N) * v0

    rb = max(r0, rf)
    head = np.zeros(N_ITER)
    for i in range(N_ITER):
        nb, xb, yb, vxb, vyb = buf_fn(rb, x, vx, vy, N)
        dx = xb[:nb][np.newaxis, :] - x[:N][:, np.newaxis]
        dy = yb[:nb][np.newaxis, :] - x[N:][:, np.newaxis]
        d2 = dx**2 + dy**2
        not_self = np.ones((N, nb), dtype=bool)
        idx = np.arange(min(N, nb)); not_self[idx, idx] = False

        flock_mask = (d2 <= rf**2) & not_self
        flx = np.where(flock_mask, vxb[:nb], 0.).sum(axis=1)
        fly = np.where(flock_mask, vyb[:nb], 0.).sum(axis=1)
        nfl = flock_mask.sum(axis=1)
        nrm = np.sqrt(flx**2 + fly**2); nrm[nfl == 0] = 1.
        flockx = p['alpha'] * flx / nrm; flocky = p['alpha'] * fly / nrm

        rep_mask = (d2 <= (2 * r0)**2) & not_self & (d2 > 0)
        d_safe   = np.where(rep_mask, np.sqrt(d2), 1.)
        base_r   = np.where(rep_mask, 1. - d_safe / (2. * r0), 0.)
        strength = np.where(rep_mask, eps * base_r**1.5 / d_safe, 0.)
        repx = (-strength * dx).sum(axis=1); repy = (-strength * dy).sum(axis=1)

        vnorm = np.sqrt(vx**2 + vy**2)
        vnorms = np.where(vnorm == 0, 1., vnorm)
        fpropx = mu * (v0 - vnorm) * vx / vnorms
        fpropy = mu * (v0 - vnorm) * vy / vnorms

        frandx = p['ramp'] * rng.uniform(-1., 1., N)
        frandy = p['ramp'] * rng.uniform(-1., 1., N)

        fx = flockx + repx + fpropx + frandx
        fy = flocky + repy + fpropy + frandy

        if n1 > 0: fx[grp1] += w_lead * g1[0]; fy[grp1] += w_lead * g1[1]
        if n2 > 0: fx[grp2] += w_lead * g2[0]; fy[grp2] += w_lead * g2[1]

        vx += fx * dt; vy += fy * dt
        x[:N] = (x[:N] + vx * dt) % 1.; x[N:] = (x[N:] + vy * dt) % 1.

        head[i] = np.degrees(np.arctan2(vy.mean(), vx.mean()))

    final = np.degrees(np.arctan2(
        np.mean(np.sin(np.radians(head[-SETTLE_WIN:]))),
        np.mean(np.cos(np.radians(head[-SETTLE_WIN:])))))
    # settle time: last time heading is OUTSIDE the tolerance band, +1 step
    outside = np.array([ang_diff(h, final) > TOL_DEG for h in head])
    if outside.any():
        settle_step = np.max(np.where(outside)[0]) + 1
    else:
        settle_step = 0
    settle_tu = settle_step * BASE['dt']
    acc = np.cos(np.radians(final))   # accuracy toward g1 (+x)
    return settle_tu, acc, final


print('Time-resolved leadership decisions: how fast does the flock commit?')
print('  N=%d  pure-flock  %d seeds  settle tol=%.0f deg\n' % (BASE['N'], N_SEEDS, TOL_DEG))

# Part 1 -- response time vs informed fraction (single leader)
print('== Part 1: RESPONSE TIME vs informed fraction (single leader, w=1.0) ==')
RHO_VALS = [0.02, 0.05, 0.10, 0.20, 0.50]
part1 = {}
for rho in RHO_VALS:
    ts = []; accs = []
    for s in range(N_SEEDS):
        t, a, _ = run(0.0, rho, 0.0, 1.0, s)
        ts.append(t); accs.append(a)
    part1[rho] = (np.mean(ts), np.std(ts), np.mean(accs))
    print('   rho=%.2f (%3d informed)  settle_time=%5.2f +/- %4.2f tu  accuracy=%+.3f'
          % (rho, int(round(rho * BASE['N'])), np.mean(ts), np.std(ts), np.mean(accs)))

# Part 1b -- speed vs accuracy under noise (fixed rho=0.10)
print('\n== Part 1b: noise dependence (single leader, rho=0.10, w=1.0) ==')
RAMP_VALS = [0.5, 2.0, 5.0, 10.0]
part1b = {}
for ramp in RAMP_VALS:
    ts = []; accs = []
    for s in range(N_SEEDS):
        t, a, _ = run(0.0, 0.10, 0.0, 1.0, s, ramp=ramp)
        ts.append(t); accs.append(a)
    part1b[ramp] = (np.mean(ts), np.std(ts), np.mean(accs))
    print('   ramp=%4.1f  settle_time=%5.2f +/- %4.2f tu  accuracy=%+.3f'
          % (ramp, np.mean(ts), np.std(ts), np.mean(accs)))

# Part 2 -- commitment time vs conflict angle (two equal opposed subgroups)
print('\n== Part 2: COMMITMENT TIME vs conflict angle (equal 5%+5%, w=1.0) ==')
print('   critical slowing test: peak settle_time near the F73 compromise->consensus boundary?')
THETA_VALS = [30, 60, 90, 120, 150, 180]
rho_e = 0.05
part2 = {}
for theta in THETA_VALS:
    ts = []; accs = []
    for s in range(N_SEEDS):
        t, a, _ = run(theta, rho_e, rho_e, 1.0, s)
        ts.append(t); accs.append(a)
    part2[theta] = (np.mean(ts), np.std(ts), np.mean(accs))
    print('   theta=%3d  settle_time=%5.2f +/- %4.2f tu  final_acc_to_g1=%+.3f'
          % (theta, np.mean(ts), np.std(ts), np.mean(accs)))

# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle('Time-resolved leadership decisions (N=%d, %d seeds)' % (BASE['N'], N_SEEDS),
             fontsize=12)

ax = axes[0]
t1 = [part1[r][0] for r in RHO_VALS]; e1 = [part1[r][1] for r in RHO_VALS]
a1 = [part1[r][2] for r in RHO_VALS]
ax.errorbar(RHO_VALS, t1, yerr=e1, marker='o', capsize=4, lw=2, color='steelblue',
            label='settle time (tu)')
ax2 = ax.twinx()
ax2.plot(RHO_VALS, a1, 's--', color='crimson', lw=2, label='accuracy')
ax.set_xlabel('informed fraction rho'); ax.set_ylabel('settle time (tu)', color='steelblue')
ax2.set_ylabel('final accuracy', color='crimson'); ax2.set_ylim(0, 1.05)
ax.set_title('Part 1: response time & accuracy vs rho'); ax.grid(alpha=0.3)

ax = axes[1]
tb = [part1b[r][0] for r in RAMP_VALS]; eb = [part1b[r][1] for r in RAMP_VALS]
ab = [part1b[r][2] for r in RAMP_VALS]
ax.errorbar(RAMP_VALS, tb, yerr=eb, marker='o', capsize=4, lw=2, color='steelblue',
            label='settle time (tu)')
ax2 = ax.twinx()
ax2.plot(RAMP_VALS, ab, 's--', color='crimson', lw=2, label='accuracy')
ax.set_xlabel('noise ramp'); ax.set_ylabel('settle time (tu)', color='steelblue')
ax2.set_ylabel('final accuracy', color='crimson'); ax2.set_ylim(0, 1.05)
ax.set_title('Part 1b: speed-accuracy vs noise (rho=0.10)'); ax.grid(alpha=0.3)

ax = axes[2]
t2 = [part2[t][0] for t in THETA_VALS]; e2 = [part2[t][1] for t in THETA_VALS]
ax.errorbar(THETA_VALS, t2, yerr=e2, marker='o', capsize=4, lw=2, color='purple')
ax.set_xlabel('conflict angle theta (deg)'); ax.set_ylabel('commitment time (tu)')
ax.set_title('Part 2: critical slowing near decision boundary?'); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/decision_time_1.png', dpi=120)
plt.close()
print('\n  --> figures/decision_time_1.png')
print('\nTime-resolved decision analysis complete.')
