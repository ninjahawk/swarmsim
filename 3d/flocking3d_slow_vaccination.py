# flocking3d_slow_vaccination.py -- 3D slow-recoverer vaccination (Finding 58)
#
# F56 showed that in a 2D flock with bimodal recovery rates, targeting the slow
# half of the population beats random vaccination by 2-3x, and is the only
# targeting strategy in this study that does.  F57 confirmed the advantage is
# internal (per-agent rate), not spatial.
#
# F46 had reported a 3D vaccination NULL: random, spatial, and degree-targeted
# vaccination are statistically identical in 3D, because (a) the contact graph
# in 3D is even more homogeneous than 2D (F46), and (b) kinematic mixing
# transfers to 3D unchanged (F52 says 3D actually mixes 1.8x slower at matched
# degree, but slowly enough that coverage still erodes).  The F56 mechanism
# (per-agent rate hubs invariant under mixing) is dimension-independent in
# principle.  Prediction: 3D slow-targeting reproduces the 2D advantage.
#
# This script tests it.  Same 3D harness as F46/F47; add bimodal per-agent
# gamma_i, then compare random / spatial / slow vaccination at matched p_immune.

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import os
import time

os.makedirs('figures', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

N         = 350
N_SEEDS   = 3
N_WARMUP  = 3000
N_ITER    = 8000
DT        = 0.01
RECORD_EVERY = 50

R0_3D = 0.02
RF_3D = 0.20
ALPHA = 1.0
V0    = 0.02
MU    = 10.0
RAMP  = 0.1
EPS   = 0.1
EXP_N = 1.5
RB_3D = 2.0 * R0_3D

PANIC_ALPHA = 0.1
PANIC_RAMP  = 10.0
R_CONT      = 0.155
BETA        = 1.5     # tuned for 3D so the homogeneous threshold is just below
GAMMA_MEAN  = 2.0
SPREAD      = 1.6     # bimodal {0.4, 3.6}; mirror of F54 "strong" at gamma_mean=2.0
F0_FRAC     = 0.05

P_IMMUNE_LIST = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50]


def order_param3d(vel):
    spd = np.maximum(np.sqrt((vel**2).sum(axis=0)), 1e-10)
    vhat = vel / spd[np.newaxis, :]
    return float(np.sqrt((vhat.mean(axis=1)**2).sum()))


def flock_forces(pos, vel):
    dx = pos[0, np.newaxis, :] - pos[0, :, np.newaxis]
    dy = pos[1, np.newaxis, :] - pos[1, :, np.newaxis]
    dz = pos[2, np.newaxis, :] - pos[2, :, np.newaxis]
    dx -= np.round(dx); dy -= np.round(dy); dz -= np.round(dz)
    d2 = dx**2 + dy**2 + dz**2
    not_self = ~np.eye(N, dtype=bool)
    rep_mask = (d2 <= RB_3D**2) & not_self & (d2 > 0)
    d_safe = np.where(rep_mask, np.sqrt(d2), 1.0)
    base_r = np.maximum(np.where(rep_mask, 1.0 - d_safe/RB_3D, 0.0), 0.0)
    strength = np.where(rep_mask, EPS * base_r**EXP_N / d_safe, 0.0)
    fx = (-strength * dx).sum(axis=1)
    fy = (-strength * dy).sum(axis=1)
    fz = (-strength * dz).sum(axis=1)
    return fx, fy, fz, d2, not_self


def step_flock(pos, vel, alpha_arr, ramp_arr, rng):
    fx, fy, fz, d2, not_self = flock_forces(pos, vel)
    flock_mask = (d2 <= RF_3D**2) & not_self
    svx = (vel[0] * flock_mask).sum(axis=1)
    svy = (vel[1] * flock_mask).sum(axis=1)
    svz = (vel[2] * flock_mask).sum(axis=1)
    vbar = np.sqrt(svx**2 + svy**2 + svz**2)
    has = (flock_mask.sum(axis=1) > 0)
    safe = np.where(has, vbar, 1.0)
    fx += np.where(has, alpha_arr * svx / safe, 0.0)
    fy += np.where(has, alpha_arr * svy / safe, 0.0)
    fz += np.where(has, alpha_arr * svz / safe, 0.0)
    spd = np.maximum(np.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2), 1e-10)
    prop = MU * (V0 - spd) / spd
    fx += prop * vel[0]; fy += prop * vel[1]; fz += prop * vel[2]
    fx += ramp_arr * rng.uniform(-1., 1., N)
    fy += ramp_arr * rng.uniform(-1., 1., N)
    fz += ramp_arr * rng.uniform(-1., 1., N)
    vel[0] += fx * DT; vel[1] += fy * DT; vel[2] += fz * DT
    pos[0] = (pos[0] + vel[0]*DT) % 1.0
    pos[1] = (pos[1] + vel[1]*DT) % 1.0
    pos[2] = (pos[2] + vel[2]*DT) % 1.0
    return pos, vel


def warmup(seed):
    np.random.seed(seed)
    pos = np.random.uniform(0., 1., (3, N))
    raw = np.random.randn(3, N)
    raw /= np.sqrt((raw**2).sum(axis=0))
    vel = V0 * raw
    rng = np.random.default_rng(seed * 13 + 1)
    alpha_arr = np.full(N, ALPHA)
    ramp_arr  = np.full(N, RAMP)
    for _ in range(N_WARMUP):
        pos, vel = step_flock(pos, vel, alpha_arr, ramp_arr, rng)
    return pos, vel


def contact_within(pos):
    dx = pos[0, np.newaxis, :] - pos[0, :, np.newaxis]
    dy = pos[1, np.newaxis, :] - pos[1, :, np.newaxis]
    dz = pos[2, np.newaxis, :] - pos[2, :, np.newaxis]
    dx -= np.round(dx); dy -= np.round(dy); dz -= np.round(dz)
    rd2 = dx**2 + dy**2 + dz**2
    return (rd2 <= R_CONT**2) & (rd2 > 0)


def spatial_select(pos, n_im):
    if n_im == 0:
        return np.array([], dtype=int)
    c = pos.mean(axis=1)
    d0 = pos - c[:, np.newaxis]
    d0 -= np.round(d0)
    first = int(np.argmin((d0**2).sum(axis=0)))
    selected = [first]
    d = pos - pos[:, first:first+1]
    d -= np.round(d)
    min_dist = np.sqrt((d**2).sum(axis=0))
    min_dist[first] = -1.0
    while len(selected) < n_im:
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        min_dist[nxt] = -1.0
        d = pos - pos[:, nxt:nxt+1]
        d -= np.round(d)
        new_dist = np.sqrt((d**2).sum(axis=0))
        min_dist = np.where(min_dist >= 0, np.minimum(min_dist, new_dist), min_dist)
    return np.array(selected, dtype=int)


def make_gamma(rng, N_):
    gamma = np.full(N_, GAMMA_MEAN, dtype=float)
    is_slow = np.zeros(N_, dtype=bool)
    if SPREAD > 0:
        idx = rng.choice(N_, size=N_ // 2, replace=False)
        is_slow[idx] = True
        gamma[is_slow]  = GAMMA_MEAN - SPREAD
        gamma[~is_slow] = GAMMA_MEAN + SPREAD
    return gamma, is_slow


def run_sis_het(pos0, vel0, rng_sis, gamma_arr, is_immune):
    pos = pos0.copy(); vel = vel0.copy()
    is_panicked = np.zeros(N, dtype=bool)
    n0 = max(1, round(F0_FRAC * N))
    susc = np.where(~is_immune)[0]
    if susc.size >= n0:
        idx0 = rng_sis.choice(susc, size=n0, replace=False)
    else:
        idx0 = susc
    is_panicked[idx0] = True
    p_recover_per_step = 1.0 - np.exp(-gamma_arr * DT)
    last_window_start = N_ITER - int(20.0 / DT)
    f_series = []
    for i in range(N_ITER):
        alpha_arr = np.where(is_panicked, PANIC_ALPHA, ALPHA)
        ramp_arr  = np.where(is_panicked, PANIC_RAMP,  RAMP)
        pos, vel = step_flock(pos, vel, alpha_arr, ramp_arr, rng_sis)
        if is_panicked.any() and (~is_panicked & ~is_immune).any():
            within = contact_within(pos)
            k_arr = within @ is_panicked.astype(np.int32)
            calm_sus = np.where(~is_panicked & ~is_immune)[0]
            if calm_sus.size:
                p_trans = 1.0 - np.exp(-BETA * k_arr[calm_sus] * DT)
                r = rng_sis.uniform(0., 1., calm_sus.size)
                flipped = calm_sus[r < p_trans]
                if flipped.size:
                    is_panicked[flipped] = True
        if is_panicked.any():
            pidx = np.where(is_panicked)[0]
            r = rng_sis.uniform(0., 1., pidx.size)
            recovered = pidx[r < p_recover_per_step[pidx]]
            if recovered.size:
                is_panicked[recovered] = False
        if i >= last_window_start and i % RECORD_EVERY == 0:
            f_series.append(is_panicked.mean())
    return float(np.mean(f_series)) if f_series else 0.0


if __name__ == '__main__':
    print('F58 -- 3D slow-recoverer vaccination')
    print('  beta=%.2f gamma_mean=%.2f spread=%.2f  N=%d seeds=%d' % (
          BETA, GAMMA_MEAN, SPREAD, N, N_SEEDS))
    print('  bimodal gamma {%.2f, %.2f}' % (GAMMA_MEAN - SPREAD, GAMMA_MEAN + SPREAD))
    print('  p_immune: %s' % P_IMMUNE_LIST)
    print()

    results = {p: {'random': [], 'spatial': [], 'slow': []}
               for p in P_IMMUNE_LIST}
    t0 = time.time()

    for s in range(N_SEEDS):
        print('  seed %d: warmup...' % s, flush=True)
        pos0, vel0 = warmup(s)
        phi_w = order_param3d(vel0)
        within = contact_within(pos0)
        deg = within.sum(axis=1)
        print('    Phi=%.3f  mean_k=%.2f  max_k=%d' % (
              phi_w, deg.mean(), deg.max()), flush=True)

        rng_het = np.random.default_rng(seed=10000 + s)
        gamma_arr, is_slow = make_gamma(rng_het, N)
        slow_idx_sorted = np.argsort(gamma_arr +
                                     rng_het.uniform(0., 1e-9, N))

        for p_im in P_IMMUNE_LIST:
            n_im = int(round(p_im * N))

            rng_r = np.random.default_rng(seed=s * 1000 + int(p_im * 1000))
            is_im_r = np.zeros(N, dtype=bool)
            if n_im > 0:
                is_im_r[rng_r.choice(N, size=n_im, replace=False)] = True

            sp_idx = spatial_select(pos0, n_im)
            is_im_s = np.zeros(N, dtype=bool)
            if sp_idx.size > 0:
                is_im_s[sp_idx] = True

            is_im_sl = np.zeros(N, dtype=bool)
            if n_im > 0:
                is_im_sl[slow_idx_sorted[:n_im]] = True

            f_r = run_sis_het(pos0, vel0,
                              np.random.default_rng(s*2000 + int(p_im*1000)),
                              gamma_arr, is_im_r)
            f_s = run_sis_het(pos0, vel0,
                              np.random.default_rng(s*5000 + int(p_im*1000)),
                              gamma_arr, is_im_s)
            f_sl = run_sis_het(pos0, vel0,
                               np.random.default_rng(s*7000 + int(p_im*1000)),
                               gamma_arr, is_im_sl)
            results[p_im]['random'].append(f_r)
            results[p_im]['spatial'].append(f_s)
            results[p_im]['slow'].append(f_sl)
            print('    p=%.2f: random=%.3f spatial=%.3f slow=%.3f' % (
                  p_im, f_r, f_s, f_sl), flush=True)

    print('\nRuntime: %.1f min' % ((time.time() - t0)/60.0))

    print('\n=== Results ===')
    print('%8s  %10s  %10s  %10s' % ('p_immune', 'random', 'spatial', 'slow'))
    rm_a, sm_a, slm_a, re_a, se_a, sle_a = [], [], [], [], [], []
    for p in P_IMMUNE_LIST:
        rm, rs = np.mean(results[p]['random']),  np.std(results[p]['random'])
        sm, ss = np.mean(results[p]['spatial']), np.std(results[p]['spatial'])
        sl, sle = np.mean(results[p]['slow']),   np.std(results[p]['slow'])
        rm_a.append(rm); sm_a.append(sm); slm_a.append(sl)
        re_a.append(rs); se_a.append(ss); sle_a.append(sle)
        print('  %5.2f     %.3f+/-%.3f  %.3f+/-%.3f  %.3f+/-%.3f' % (
              p, rm, rs, sm, ss, sl, sle))

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.errorbar(P_IMMUNE_LIST, rm_a, yerr=re_a, fmt='o-', color='gray',
                lw=2, capsize=3, label='random')
    ax.errorbar(P_IMMUNE_LIST, sm_a, yerr=se_a, fmt='s-', color='steelblue',
                lw=2, capsize=3, label='spatial')
    ax.errorbar(P_IMMUNE_LIST, slm_a, yerr=sle_a, fmt='^-', color='crimson',
                lw=2, capsize=3, label='slow')
    ax.set_xlabel('p_immune'); ax.set_ylabel('steady-state panic frac')
    ax.set_title('3D heterogeneous-recovery vaccination (beta=%.2f, N=%d, %d seeds)'
                 % (BETA, N, N_SEEDS))
    ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig('figures/flocking3d_slow_vaccination_1.png', dpi=120)
    plt.close()
    print('  --> figures/flocking3d_slow_vaccination_1.png')
    print('\nF58 complete.')
