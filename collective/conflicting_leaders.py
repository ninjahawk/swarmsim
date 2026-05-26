# conflicting_leaders.py -- two informed subgroups pulling in DIFFERENT directions.
#
# F72 (leadership.py) showed a single informed minority steers the whole flock toward
# one shared direction. The classic Couzin et al. (2005) follow-up is CONFLICT: split the
# informed agents into two subgroups that prefer DIFFERENT directions and ask whether the
# group (a) COMPROMISES -- travels the average direction, (b) reaches CONSENSUS -- picks one
# of the two and commits, or (c) SPLITS -- the flock fragments into two sub-flocks.
#
# Couzin's prediction: for a small angular difference the group compromises (averages); past
# a critical angular difference it switches to consensus (selects one direction at random),
# because averaging two widely separated directions is not a viable heading. With unequal
# subgroup sizes the larger subgroup wins more often. We reproduce both effects here in the
# Charbonneau/Silverberg flocking model (where leadership was never previously studied).
#
# Two experiments:
#   Exp 1 -- ANGLE sweep at equal subgroup sizes (rho1 = rho2 = 0.05). g1 = +x; g2 rotated
#            by theta from +x. Sweep theta 0..180 deg. Measure where the flock actually
#            travels and whether it is intermediate (compromise) or committed to one side.
#   Exp 2 -- SIZE-RATIO sweep at fixed large angle (theta = 180 deg, direct opposition).
#            Vary n1:n2 with n1+n2 fixed. Does the majority subgroup win?
#
# Metrics (over the measurement window):
#   heading_deg = direction of the flock mean velocity, degrees from +x
#   |compromise gap| = how far the heading sits from the nearer of the two goals vs the
#                      midpoint, used to classify compromise vs consensus
#   bimodality = std of per-seed heading: low = all seeds agree (compromise),
#                high = seeds split between the two goals (consensus, random pick)
#   Phi = order parameter (does the flock stay coherent or split?)
#   split_frac = fraction of agents in the smaller of two velocity-direction clusters
#                (~0 = one flock, ~0.5 = clean two-way split)

import os
import numpy as np
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from flocking import buffer as buf_fn, order_parameter

os.makedirs('figures', exist_ok=True)
N_SEEDS = 6

BASE = dict(N=350, r0=0.005, eps=0.1, rf=0.1,
            alpha=1.0, v0=1.0, mu=10.0, ramp=0.5, dt=0.01)
N_WARMUP = 1000
N_ITER   = 4000
W_LEAD   = 1.0          # leader bias strength (F72: w=1 gives crisp steering)
RHO_EACH = 0.05         # each subgroup is 5% of the flock (F72: enough to steer alone)


def run(theta_deg, rho1, rho2, w_lead, seed):
    """Two informed subgroups: subgroup 1 -> +x, subgroup 2 -> rotated by theta."""
    rng = np.random.RandomState(seed)
    p = BASE.copy()
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
    hx_rec = []; hy_rec = []; phi_rec = []; split_rec = []
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

        # leadership bias on each informed subgroup
        fx[grp1] += w_lead * g1[0]; fy[grp1] += w_lead * g1[1]
        fx[grp2] += w_lead * g2[0]; fy[grp2] += w_lead * g2[1]

        vx += fx * dt; vy += fy * dt
        x[:N] = (x[:N] + vx * dt) % 1.; x[N:] = (x[N:] + vy * dt) % 1.

        if i >= N_WARMUP:
            mvx = vx.mean(); mvy = vy.mean()
            hx_rec.append(mvx); hy_rec.append(mvy)
            phi_rec.append(order_parameter(vx, vy))
            # split metric: fraction of agents whose velocity is closer to g2 than g1
            # (only meaningful when the flock has actually split)
            proj1 = vx * g1[0] + vy * g1[1]
            proj2 = vx * g2[0] + vy * g2[1]
            frac2 = np.mean(proj2 > proj1)
            split_rec.append(min(frac2, 1 - frac2))   # smaller cluster fraction

    mhx = np.mean(hx_rec); mhy = np.mean(hy_rec)
    heading = np.degrees(np.arctan2(mhy, mhx))
    return heading, float(np.mean(phi_rec)), float(np.mean(split_rec))


# ---------------------------------------------------------------------------
print('Conflicting leaders: two informed subgroups pulling in different directions')
print('  N=%d  pure-flock (v0=1.0)  %d seeds  w_lead=%.1f\n' % (BASE['N'], N_SEEDS, W_LEAD))

# Exp 1 -- angle sweep, equal subgroups
print('== Exp 1: ANGLE sweep (equal subgroups, rho=%.2f each) ==' % RHO_EACH)
print('   g1 = +x (0 deg); g2 rotated by theta. Midpoint heading = theta/2.')
THETA_VALS = [0, 30, 60, 90, 120, 150, 180]
exp1 = {}
for theta in THETA_VALS:
    hs = []; phis = []; splits = []
    for s in range(N_SEEDS):
        h, ph, sp = run(theta, RHO_EACH, RHO_EACH, W_LEAD, s)
        # fold heading into [0,180] measured from g1, signed toward g2 side
        hs.append(h); phis.append(ph); splits.append(sp)
    hs = np.array(hs)
    # circular-ish summary: report mean |heading| and its spread across seeds
    mean_h = np.mean(hs); std_h = np.std(hs)
    midpoint = theta / 2.0
    exp1[theta] = (mean_h, std_h, np.mean(phis), np.mean(splits), midpoint)
    print('   theta=%3d  midpoint=%5.1f  heading=%+6.1f +/- %4.1f  Phi=%.3f  split_frac=%.3f'
          % (theta, midpoint, mean_h, std_h, np.mean(phis), np.mean(splits)))

# Exp 2 -- size-ratio sweep at direct opposition (theta=180)
print('\n== Exp 2: SIZE-RATIO sweep (theta=180 deg, direct opposition) ==')
print('   subgroup 1 -> +x, subgroup 2 -> -x. Total informed fixed at 0.10.')
RATIO_VALS = [(0.05, 0.05), (0.06, 0.04), (0.07, 0.03), (0.08, 0.02), (0.10, 0.00)]
exp2 = {}
for rho1, rho2 in RATIO_VALS:
    hs = []; phis = []; splits = []
    for s in range(N_SEEDS):
        h, ph, sp = run(180, rho1, rho2, W_LEAD, s)
        # accuracy toward g1 (+x): cos of heading
        hs.append(np.cos(np.radians(h)))
        phis.append(ph); splits.append(sp)
    exp2[(rho1, rho2)] = (np.mean(hs), np.std(hs), np.mean(phis), np.mean(splits))
    print('   rho1=%.2f rho2=%.2f (n1=%3d n2=%3d)  accuracy_to_g1=%+.3f +/- %.3f  Phi=%.3f  split=%.3f'
          % (rho1, rho2, int(round(rho1 * BASE['N'])), int(round(rho2 * BASE['N'])),
             np.mean(hs), np.std(hs), np.mean(phis), np.mean(splits)))

# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle('Conflicting leaders: compromise vs consensus (N=%d, %d seeds, w_lead=%.1f)'
             % (BASE['N'], N_SEEDS, W_LEAD), fontsize=12)

ax = axes[0]
thetas = THETA_VALS
headings = [exp1[t][0] for t in thetas]
herr = [exp1[t][1] for t in thetas]
mids = [exp1[t][4] for t in thetas]
ax.plot(thetas, mids, 'k--', lw=1.5, label='compromise (midpoint = theta/2)')
ax.errorbar(thetas, headings, yerr=herr, marker='o', capsize=4, lw=2,
            color='purple', label='actual flock heading')
ax.set_xlabel('angular conflict theta (deg)'); ax.set_ylabel('flock heading (deg from g1)')
ax.set_title('Exp 1: heading vs conflict angle')
ax.grid(alpha=0.3); ax.legend(fontsize=9)

ax = axes[1]
hstd = [exp1[t][1] for t in thetas]
split1 = [exp1[t][3] for t in thetas]
ax.plot(thetas, hstd, 'o-', color='crimson', lw=2, label='cross-seed heading std (deg)')
ax2 = ax.twinx()
ax2.plot(thetas, split1, 's--', color='teal', lw=2, label='split_frac (smaller cluster)')
ax.set_xlabel('angular conflict theta (deg)')
ax.set_ylabel('cross-seed heading std (deg)', color='crimson')
ax2.set_ylabel('split_frac', color='teal')
ax.set_title('Exp 1: consensus randomness & splitting')
ax.grid(alpha=0.3)

ax = axes[2]
ratios = [r1 - r2 for (r1, r2) in RATIO_VALS]   # majority margin
acc = [exp2[k][0] for k in RATIO_VALS]
aerr = [exp2[k][1] for k in RATIO_VALS]
ax.axhline(0, ls=':', color='gray')
ax.errorbar(ratios, acc, yerr=aerr, marker='o', capsize=4, lw=2, color='darkgreen')
ax.set_xlabel('majority margin (rho1 - rho2)')
ax.set_ylabel('accuracy toward majority goal (+x)')
ax.set_title('Exp 2: does the larger subgroup win? (theta=180)')
ax.set_ylim(-1.05, 1.05); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/conflicting_leaders_1.png', dpi=120)
plt.close()
print('\n  --> figures/conflicting_leaders_1.png')
print('\nConflicting-leaders analysis complete.')
