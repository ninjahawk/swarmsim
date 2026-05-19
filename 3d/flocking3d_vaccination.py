# flocking3d_vaccination.py -- 3D spatial vs random vs degree-targeted vaccination
#                              (Finding 46)
#
# Motivation.  The report's Section 5 synthesis identifies ALIGNMENT-DRIVEN
# KINEMATIC MIXING as the unifying mechanism behind several null results:
#   F36  degree-targeted vaccination fails -- mixing restores hub positions
#   F37  spatial (farthest-point) vaccination fails -- mixing scrambles the
#        immune agents' spatial coverage before the epidemic runs
# Both were established in 2D.  This experiment asks whether the mechanism
# survives the move to 3D.
#
# A priori the result is not obvious.  In 3D the flock has an extra degree of
# freedom for agents to slide past one another, so kinematic mixing could be
# either FASTER (more room to rearrange -> spatial vaccination fails even harder)
# or SLOWER (lower local density -> coverage persists longer -> spatial helps).
# Finding 43 already showed one 2D result (R_enc/Rg ~ 0.5 optimum) does NOT
# transfer to 3D, so transfer cannot be assumed.
#
# Design.  3D flocking model (N=350, [0,1]^3 torus, params from flocking3d.py).
# SIS contagion on the 3D contact network.  Three vaccination strategies:
#   random   -- immune agents uniform at random
#   spatial  -- farthest-point (maxmin) sampling on the 3D torus
#   targeted -- top contact-degree agents
# Sweep p_immune; metric f_ss = mean panic fraction over the last 20 time units.
#
# The contact radius R_CONT is chosen to give a mean contact degree comparable
# to the 2D experiments (~9); the script prints the measured value so the
# epidemic ratio R0 can be interpreted.
#
# Run with:  python 3d/flocking3d_vaccination.py

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import os
import time

os.makedirs('figures', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
N         = 350
N_SEEDS   = 5
N_WARMUP  = 3000
N_ITER    = 10000          # 100 time units of SIS dynamics
DT        = 0.01
RECORD_EVERY = 50

# 3D prey physics (matched to flocking3d.py)
R0_3D = 0.02
RF_3D = 0.20
ALPHA = 1.0
V0    = 0.02
MU    = 10.0
RAMP  = 0.1
EPS   = 0.1
EXP_N = 1.5
RB_3D = 2.0 * R0_3D

# Contagion
PANIC_ALPHA = 0.1
PANIC_RAMP  = 10.0
R_CONT      = 0.155        # 3D contact radius: tuned so mean_k ~ 9 (matches the 2D
                           # vaccination experiments). The 3D flock is far more dilute
                           # than the 2D one, so a larger contact radius is required;
                           # R_CONT < rf=0.20 keeps it a sub-network of the alignment graph.
BETA        = 2.5
GAMMA       = 2.0
F0_FRAC     = 0.05

P_IMMUNE_LIST = [0.00, 0.10, 0.20, 0.30, 0.40, 0.46, 0.50, 0.60]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def order_param3d(vel):
    spd = np.maximum(np.sqrt((vel**2).sum(axis=0)), 1e-10)
    vhat = vel / spd[np.newaxis, :]
    return float(np.sqrt((vhat.mean(axis=1)**2).sum()))


def flock_forces(pos, vel):
    """Vectorized 3D prey forces. pos,vel are (3,N). Returns (3,N) force array."""
    dx = pos[0, np.newaxis, :] - pos[0, :, np.newaxis]
    dy = pos[1, np.newaxis, :] - pos[1, :, np.newaxis]
    dz = pos[2, np.newaxis, :] - pos[2, :, np.newaxis]
    dx -= np.round(dx); dy -= np.round(dy); dz -= np.round(dz)
    d2 = dx**2 + dy**2 + dz**2
    not_self = ~np.eye(N, dtype=bool)

    rep_mask = (d2 <= RB_3D**2) & not_self & (d2 > 0)
    d_safe   = np.where(rep_mask, np.sqrt(d2), 1.0)
    base_r   = np.maximum(np.where(rep_mask, 1.0 - d_safe/RB_3D, 0.0), 0.0)
    strength = np.where(rep_mask, EPS * base_r**EXP_N / d_safe, 0.0)
    fx = (-strength * dx).sum(axis=1)
    fy = (-strength * dy).sum(axis=1)
    fz = (-strength * dz).sum(axis=1)
    return fx, fy, fz, d2, not_self


def step_flock(pos, vel, alpha_arr, ramp_arr, rng):
    """Advance the 3D flock one step with per-agent alpha and ramp."""
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
    """Boolean (N,N) contact matrix on 3D torus, radius R_CONT, no self."""
    dx = pos[0, np.newaxis, :] - pos[0, :, np.newaxis]
    dy = pos[1, np.newaxis, :] - pos[1, :, np.newaxis]
    dz = pos[2, np.newaxis, :] - pos[2, :, np.newaxis]
    dx -= np.round(dx); dy -= np.round(dy); dz -= np.round(dz)
    rd2 = dx**2 + dy**2 + dz**2
    return (rd2 <= R_CONT**2) & (rd2 > 0)


def measure_degree(pos):
    return contact_within(pos).sum(axis=1)


def spatial_select(pos, n_im):
    """Farthest-point (maxmin) sampling on the 3D torus."""
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


def run_sis(pos0, vel0, rng, is_immune):
    """Run SIS on the 3D flock. Return f_ss (mean panic fraction, last 20 tu)."""
    pos = pos0.copy(); vel = vel0.copy()
    is_panicked = np.zeros(N, dtype=bool)

    n0 = max(1, round(F0_FRAC * N))
    susceptible = np.where(~is_immune)[0]
    if susceptible.size >= n0:
        idx0 = rng.choice(susceptible, size=n0, replace=False)
    else:
        idx0 = susceptible
    is_panicked[idx0] = True

    p_recover = 1.0 - np.exp(-GAMMA * DT)
    last_window_start = N_ITER - int(20.0 / DT)
    f_series = []

    for i in range(N_ITER):
        alpha_arr = np.where(is_panicked, PANIC_ALPHA, ALPHA)
        ramp_arr  = np.where(is_panicked, PANIC_RAMP,  RAMP)
        pos, vel = step_flock(pos, vel, alpha_arr, ramp_arr, rng)

        if is_panicked.any() and (~is_panicked & ~is_immune).any():
            within = contact_within(pos)
            k_arr = within @ is_panicked.astype(np.int32)
            calm_sus = np.where(~is_panicked & ~is_immune)[0]
            if calm_sus.size:
                p_trans = 1.0 - np.exp(-BETA * k_arr[calm_sus] * DT)
                r = rng.uniform(0., 1., calm_sus.size)
                flipped = calm_sus[r < p_trans]
                if flipped.size:
                    is_panicked[flipped] = True

        if is_panicked.any():
            panic_idx = np.where(is_panicked)[0]
            r = rng.uniform(0., 1., panic_idx.size)
            recovered = panic_idx[r < p_recover]
            if recovered.size:
                is_panicked[recovered] = False

        if i >= last_window_start and i % RECORD_EVERY == 0:
            f_series.append(is_panicked.mean())

    return float(np.mean(f_series)) if f_series else 0.0


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print('Finding 46 -- 3D spatial vs random vs targeted vaccination')
    print('  beta=%.2f gamma=%.2f R0=%.2f  N=%d seeds=%d' % (
          BETA, GAMMA, BETA/GAMMA, N, N_SEEDS))
    print('  p_immune: %s' % P_IMMUNE_LIST)
    print()

    results = {p: {'random': [], 'spatial': [], 'targeted': []}
               for p in P_IMMUNE_LIST}
    degree_all = []
    t0 = time.time()

    for s in range(N_SEEDS):
        print('  seed %d: warmup...' % s, flush=True)
        pos0, vel0 = warmup(s)
        phi_w = order_param3d(vel0)
        deg = measure_degree(pos0)
        degree_all.append(deg)
        sorted_deg_idx = np.argsort(-deg)
        print('    Phi=%.3f  mean_k=%.2f  max_k=%d' % (
              phi_w, deg.mean(), deg.max()), flush=True)

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

            is_im_t = np.zeros(N, dtype=bool)
            if n_im > 0:
                is_im_t[sorted_deg_idx[:n_im]] = True

            f_r = run_sis(pos0, vel0,
                          np.random.default_rng(s*2000 + int(p_im*1000)), is_im_r)
            f_s = run_sis(pos0, vel0,
                          np.random.default_rng(s*5000 + int(p_im*1000)), is_im_s)
            f_t = run_sis(pos0, vel0,
                          np.random.default_rng(s*3000 + int(p_im*1000)), is_im_t)
            results[p_im]['random'].append(f_r)
            results[p_im]['spatial'].append(f_s)
            results[p_im]['targeted'].append(f_t)
            print('    p=%.2f: random=%.3f  spatial=%.3f  targeted=%.3f' % (
                  p_im, f_r, f_s, f_t), flush=True)

    print('\nTotal runtime: %.1f min' % ((time.time() - t0)/60.0))

    print('\n=== Results ===')
    print('%8s  %14s  %14s  %14s' % (
          'p_immune', 'f_ss(random)', 'f_ss(spatial)', 'f_ss(target)'))
    r_m, s_m, t_m, r_e, s_e, t_e = [], [], [], [], [], []
    for p in P_IMMUNE_LIST:
        rm, rs = np.mean(results[p]['random']),  np.std(results[p]['random'])
        sm, ss = np.mean(results[p]['spatial']), np.std(results[p]['spatial'])
        tm, ts = np.mean(results[p]['targeted']),np.std(results[p]['targeted'])
        r_m.append(rm); s_m.append(sm); t_m.append(tm)
        r_e.append(rs); s_e.append(ss); t_e.append(ts)
        print('%8.2f  %5.3f+/-%.3f  %5.3f+/-%.3f  %5.3f+/-%.3f' % (
              p, rm, rs, sm, ss, tm, ts))

    all_deg = np.concatenate(degree_all)
    print('\nDegree distribution: mean=%.2f  median=%.0f  std=%.2f  max=%d  CV=%.2f' % (
          all_deg.mean(), np.median(all_deg), all_deg.std(), all_deg.max(),
          all_deg.std()/max(all_deg.mean(), 1e-9)))

    p_arr = np.array(P_IMMUNE_LIST)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.errorbar(p_arr, r_m, yerr=r_e, marker='o', label='random',
                color='steelblue', capsize=4)
    ax.errorbar(p_arr, s_m, yerr=s_e, marker='^', label='spatial (farthest-point)',
                color='seagreen', capsize=4)
    ax.errorbar(p_arr, t_m, yerr=t_e, marker='s', label='degree-targeted',
                color='crimson', capsize=4, alpha=0.7, ls='--')
    ax.axhline(0.1, ls='--', color='gray', lw=1, label='quench threshold')
    ax.set_xlabel('Immune fraction p_immune')
    ax.set_ylabel('Steady-state panic fraction f_ss')
    ax.set_title('3D vaccination strategies (Finding 46)\nbeta=2.5 gamma=2.0 R0=1.25')
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(-0.03, 0.65)
    fig.tight_layout()
    fig.savefig('figures/finding46_3d_vaccination.png', dpi=140)
    print('\n  -> figures/finding46_3d_vaccination.png')

    with open('outputs/finding46_3d_vaccination.txt', 'w') as f:
        f.write('Finding 46 -- 3D spatial vs random vs targeted vaccination\n')
        f.write('beta=%.2f gamma=%.2f R0=%.2f N=%d seeds=%d R_cont=%.3f\n' % (
                BETA, GAMMA, BETA/GAMMA, N, N_SEEDS, R_CONT))
        f.write('mean_k=%.2f max_k=%d CV=%.2f\n\n' % (
                all_deg.mean(), all_deg.max(),
                all_deg.std()/max(all_deg.mean(), 1e-9)))
        f.write('p_immune  random  spatial  targeted\n')
        for j, p in enumerate(P_IMMUNE_LIST):
            f.write('%.2f      %.4f  %.4f   %.4f\n' % (
                    p, r_m[j], s_m[j], t_m[j]))
    print('  -> outputs/finding46_3d_vaccination.txt')
