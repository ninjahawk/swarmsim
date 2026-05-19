# flocking3d_segregation.py -- Alpha-contrast segregation in 3D (Finding 51)
#
# Finding 27 showed that in 2D, two populations with different alignment-strength
# alpha do spatially segregate: a local-purity diagnostic (fraction of an agent's
# r_f neighbors that share its type) rose from 0.50 (well mixed) to 0.73 at maximum
# alpha contrast. The along-heading bulk segregation index missed this -- the
# segregation is local clustering, not bulk sorting. (Finding 24 had shown that
# v0 contrast alone produces no segregation; only alpha contrast does.)
#
# This experiment asks whether alpha-contrast segregation transfers to 3D. The
# question is not obvious. The extra spatial dimension gives the two populations
# more room to slide past one another, which could either dilute the local
# clustering (segregation weaker in 3D) or be irrelevant if segregation is driven
# by the alignment coupling rather than by geometric packing. Findings 46 and 47
# established that kinematic mixing -- the process that destroys imposed structure
# -- is dimension-independent; this tests whether self-organized structure is too.
#
# Design. 3D flocking model (N=350, [0,1]^3 torus, fast-prey regime v0=1.0,
# ramp=0.5 matching the 2D segregation experiments). Two populations: active
# (alpha=1.0) and passive (alpha swept from 1.0 down to 0.0), f_active=0.5.
# Metric: 3D local purity = mean fraction of r_f neighbors sharing an agent's type.
# Purity = 0.5 means well mixed; purity > 0.5 means spatial segregation.
#
# Run with:  python 3d/flocking3d_segregation.py

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import os
import time

os.makedirs('figures', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# ---------------------------------------------------------------------------
# Parameters (3D flocking, fast-prey segregation regime)
# ---------------------------------------------------------------------------
N        = 350
N_SEEDS  = 5
N_ITER   = 4000
N_RECORD = 30          # average the metric over the last N_RECORD steps

R0_3D = 0.02
RF_3D = 0.20
V0    = 1.0
MU    = 10.0
RAMP  = 0.5
EPS   = 0.1
EXP_N = 1.5
RB    = 2.0 * R0_3D

ALPHA_ACTIVE   = 1.0
ALPHA_PASSIVE  = [1.0, 0.7, 0.5, 0.3, 0.1, 0.0]
F_ACTIVE       = 0.5


def order_param3d(vel):
    spd = np.maximum(np.sqrt((vel**2).sum(axis=0)), 1e-10)
    vhat = vel / spd[np.newaxis, :]
    return float(np.sqrt((vhat.mean(axis=1)**2).sum()))


def run_3d(alpha_passive, seed):
    """Run a 3D mixed-alpha flock. Return (purity_active, purity_passive, Phi)."""
    np.random.seed(seed)
    pos = np.random.uniform(0., 1., (3, N))
    raw = np.random.randn(3, N)
    raw /= np.sqrt((raw**2).sum(axis=0))
    vel = V0 * raw

    n_active = round(F_ACTIVE * N)
    is_active = np.zeros(N, dtype=bool)
    is_active[np.random.choice(N, size=n_active, replace=False)] = True
    alpha_arr = np.where(is_active, ALPHA_ACTIVE, alpha_passive)

    not_self = ~np.eye(N, dtype=bool)
    pur_a, pur_p, phi_v = [], [], []

    for step in range(N_ITER):
        dx = pos[0, np.newaxis, :] - pos[0, :, np.newaxis]
        dy = pos[1, np.newaxis, :] - pos[1, :, np.newaxis]
        dz = pos[2, np.newaxis, :] - pos[2, :, np.newaxis]
        dx -= np.round(dx); dy -= np.round(dy); dz -= np.round(dz)
        d2 = dx**2 + dy**2 + dz**2

        rep_mask = (d2 <= RB**2) & not_self & (d2 > 0)
        d_safe   = np.where(rep_mask, np.sqrt(d2), 1.0)
        base_r   = np.maximum(np.where(rep_mask, 1.0 - d_safe/RB, 0.0), 0.0)
        strength = np.where(rep_mask, EPS * base_r**EXP_N / d_safe, 0.0)
        fx = (-strength * dx).sum(axis=1)
        fy = (-strength * dy).sum(axis=1)
        fz = (-strength * dz).sum(axis=1)

        flock_mask = (d2 <= RF_3D**2) & not_self
        svx = (vel[0] * flock_mask).sum(axis=1)
        svy = (vel[1] * flock_mask).sum(axis=1)
        svz = (vel[2] * flock_mask).sum(axis=1)
        vbar = np.sqrt(svx**2 + svy**2 + svz**2)
        has = flock_mask.sum(axis=1) > 0
        safe = np.where(has, vbar, 1.0)
        fx += np.where(has, alpha_arr * svx/safe, 0.0)
        fy += np.where(has, alpha_arr * svy/safe, 0.0)
        fz += np.where(has, alpha_arr * svz/safe, 0.0)

        spd = np.maximum(np.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2), 1e-10)
        prop = MU * (V0 - spd) / spd
        fx += prop*vel[0]; fy += prop*vel[1]; fz += prop*vel[2]

        fx += RAMP * np.random.uniform(-1., 1., N)
        fy += RAMP * np.random.uniform(-1., 1., N)
        fz += RAMP * np.random.uniform(-1., 1., N)

        vel[0] += fx*0.01; vel[1] += fy*0.01; vel[2] += fz*0.01
        pos = (pos + vel*0.01) % 1.0

        if step >= N_ITER - N_RECORD:
            adj = (d2 <= RF_3D**2) & not_self
            deg = adj.sum(axis=1)
            deg_safe = np.where(deg == 0, 1, deg)
            same_a = adj @ is_active.astype(np.int32)
            same_p = adj @ (~is_active).astype(np.int32)
            pur_a.append((same_a[is_active] / deg_safe[is_active]).mean())
            pur_p.append((same_p[~is_active] / deg_safe[~is_active]).mean())
            phi_v.append(order_param3d(vel))

    return float(np.mean(pur_a)), float(np.mean(pur_p)), float(np.mean(phi_v))


if __name__ == '__main__':
    print('Finding 51 -- alpha-contrast segregation in 3D')
    print('  N=%d  seeds=%d  iter=%d  f_active=%.2f' % (
          N, N_SEEDS, N_ITER, F_ACTIVE))
    print('  alpha_active=%.1f  alpha_passive sweep: %s' % (
          ALPHA_ACTIVE, ALPHA_PASSIVE))
    print('  purity 0.5 = well mixed; purity > 0.5 = spatial segregation')
    print()
    t0 = time.time()

    results = {}
    for ap in ALPHA_PASSIVE:
        pa, pp, ph = [], [], []
        ts = time.time()
        for s in range(N_SEEDS):
            a, p, phi = run_3d(ap, s)
            pa.append(a); pp.append(p); ph.append(phi)
        results[ap] = (np.mean(pa), np.std(pa), np.mean(pp), np.std(pp),
                       np.mean(ph))
        print('  alpha_passive=%.2f  purity_active=%.3f+/-%.3f  '
              'purity_passive=%.3f+/-%.3f  Phi=%.3f  [%.0fs]' % (
              ap, results[ap][0], results[ap][1], results[ap][2],
              results[ap][3], results[ap][4], time.time() - ts), flush=True)

    print('\nTotal runtime: %.1f min' % ((time.time() - t0)/60.0))

    contrast = np.array([1.0 - ap for ap in ALPHA_PASSIVE])
    pur_a = np.array([results[ap][0] for ap in ALPHA_PASSIVE])
    pur_a_e = np.array([results[ap][1] for ap in ALPHA_PASSIVE])
    pur_p = np.array([results[ap][2] for ap in ALPHA_PASSIVE])
    pur_p_e = np.array([results[ap][3] for ap in ALPHA_PASSIVE])

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.errorbar(contrast, pur_a, yerr=pur_a_e, marker='o', capsize=4,
                color='crimson', label='active purity (alpha=1.0)')
    ax.errorbar(contrast, pur_p, yerr=pur_p_e, marker='s', capsize=4,
                color='steelblue', label='passive purity')
    ax.axhline(0.5, ls='--', color='gray', label='well mixed (purity=0.5)')
    ax.set_xlabel('Alpha contrast (1 - alpha_passive)')
    ax.set_ylabel('Local purity (same-type fraction of r_f neighbors)')
    ax.set_title('Finding 51 -- alpha-contrast segregation in 3D')
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig('figures/finding51_3d_segregation.png', dpi=140)
    print('  -> figures/finding51_3d_segregation.png')

    with open('outputs/finding51_3d_segregation.txt', 'w') as f:
        f.write('Finding 51 -- alpha-contrast segregation in 3D\n')
        f.write('N=%d seeds=%d iter=%d f_active=%.2f\n\n' % (
                N, N_SEEDS, N_ITER, F_ACTIVE))
        f.write('alpha_passive  contrast  purity_active  purity_passive  Phi\n')
        for ap in ALPHA_PASSIVE:
            r = results[ap]
            f.write('%.2f           %.2f      %.4f         %.4f          %.4f\n' % (
                    ap, 1.0-ap, r[0], r[2], r[4]))
    print('  -> outputs/finding51_3d_segregation.txt')
