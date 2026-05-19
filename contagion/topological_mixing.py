# topological_mixing.py -- Topological vs metric alignment: does k-NN alignment
#                           slow kinematic mixing and rescue targeted vaccination?
#                           (Finding 47)
#
# The report's Section 5 synthesis identifies alignment-driven kinematic mixing as
# the mechanism behind the vaccination null results (F36 degree-targeted, F37
# spatial, F46 the 3D confirmation). It closes with an explicit, pre-registered
# FALSIFIABLE PREDICTION:
#
#   "A flocking model in which the alignment force is replaced by a topological
#    (k-nearest-neighbor) interaction rather than a metric one should exhibit
#    weaker mixing -- the neighbor graph would be more stable because k-nearest is
#    a permutation-stable structure. Under such a modification, targeted
#    vaccination should partially recover its advantage over random."
#
# This script tests that prediction directly. It implements 2D flocking with a
# switchable alignment rule:
#   metric : align to all neighbors within rf (the standard Charbonneau model)
#   topo   : align to the k nearest neighbors regardless of distance, with k set
#            to the mean metric alignment-degree so the two rules are comparable.
#
# Two diagnostics:
#   A. Mixing rate. On a pure flock (no contagion), record each agent's contact
#      neighbor set (within R_CONT) at fixed intervals and measure the mean
#      Jaccard dissimilarity between successive snapshots -- a direct measure of
#      how fast the neighbor graph rewires. Prediction: topo mixes slower.
#   B. Vaccination. Random vs degree-targeted immunization at supercritical SIS,
#      under each alignment rule. Prediction: targeted ~ random under metric
#      (reproducing F36), but targeted < random under topo.
#
# If targeted vaccination does NOT recover under topological alignment, the
# Section 5 mechanism is wrong or incomplete -- this is a genuine test of the
# report's central claim.
#
# Run with:  python contagion/topological_mixing.py

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import os
import time

os.makedirs('figures', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# ---------------------------------------------------------------------------
# Parameters (matched to the 2D vaccination experiments, F36/F37)
# ---------------------------------------------------------------------------
N         = 350
N_SEEDS   = 5
DT        = 0.01

R0_REP = 0.005
RF     = 0.10
ALPHA  = 1.0
V0     = 0.02
MU     = 10.0
RAMP   = 0.1
EPS    = 0.1
RB_REP = 2.0 * R0_REP

# Contagion
PANIC_ALPHA = 0.1
PANIC_RAMP  = 10.0
R_CONT      = 0.05
BETA        = 2.5
GAMMA       = 2.0
F0_FRAC     = 0.05

N_WARMUP   = 1500
N_ITER     = 10000          # 100 time units of SIS
P_IMMUNE_LIST = [0.20, 0.30, 0.40, 0.50]

# Mixing diagnostic
MIX_STEPS    = 5000         # length of the pure-flock mixing run
MIX_INTERVAL = 200          # snapshot every 2 time units


# ---------------------------------------------------------------------------
# Geometry helpers (2D minimum-image torus)
# ---------------------------------------------------------------------------
def pair_d2(x, y):
    dx = x[np.newaxis, :] - x[:, np.newaxis]
    dy = y[np.newaxis, :] - y[:, np.newaxis]
    dx -= np.round(dx); dy -= np.round(dy)
    return dx, dy, dx**2 + dy**2


def order_param(vx, vy):
    spd = np.maximum(np.sqrt(vx**2 + vy**2), 1e-12)
    return float(np.sqrt((vx/spd).mean()**2 + (vy/spd).mean()**2))


def alignment_force(d2, vx, vy, mode, k, not_self):
    """Return (ax, ay): unit-normalized mean-neighbor-velocity direction.

    mode='metric': neighbors within RF.  mode='topo': k nearest neighbors.
    """
    if mode == 'metric':
        mask = (d2 <= RF**2) & not_self
        svx = (vx[np.newaxis, :] * mask).sum(axis=1)
        svy = (vy[np.newaxis, :] * mask).sum(axis=1)
        has = mask.sum(axis=1) > 0
    else:
        d2_self = np.where(not_self, d2, np.inf)
        nn = np.argpartition(d2_self, k, axis=1)[:, :k]
        svx = vx[nn].sum(axis=1)
        svy = vy[nn].sum(axis=1)
        has = np.ones(d2.shape[0], dtype=bool)
    nrm = np.sqrt(svx**2 + svy**2)
    safe = np.where(nrm > 0, nrm, 1.0)
    return np.where(has, svx/safe, 0.0), np.where(has, svy/safe, 0.0)


def step(x, y, vx, vy, mode, k, rng, alpha_arr, ramp_arr):
    """Advance the 2D flock one step under the chosen alignment rule."""
    N_ = x.size
    not_self = ~np.eye(N_, dtype=bool)
    dx, dy, d2 = pair_d2(x, y)

    rep_mask = (d2 <= RB_REP**2) & not_self & (d2 > 0)
    d_safe   = np.where(rep_mask, np.sqrt(d2), 1.0)
    base_r   = np.maximum(np.where(rep_mask, 1.0 - d_safe/RB_REP, 0.0), 0.0)
    strength = np.where(rep_mask, EPS * base_r**1.5 / d_safe, 0.0)
    fx = (-strength * dx).sum(axis=1)
    fy = (-strength * dy).sum(axis=1)

    ax, ay = alignment_force(d2, vx, vy, mode, k, not_self)
    fx += alpha_arr * ax
    fy += alpha_arr * ay

    spd = np.maximum(np.sqrt(vx**2 + vy**2), 1e-12)
    prop = MU * (V0 - spd) / spd
    fx += prop * vx; fy += prop * vy

    fx += ramp_arr * rng.uniform(-1., 1., N_)
    fy += ramp_arr * rng.uniform(-1., 1., N_)

    vx = vx + fx * DT; vy = vy + fy * DT
    x = (x + vx * DT) % 1.0
    y = (y + vy * DT) % 1.0
    return x, y, vx, vy, d2


def warmup(seed, mode, k):
    np.random.seed(seed)
    x = np.random.uniform(0., 1., N)
    y = np.random.uniform(0., 1., N)
    vx = np.random.uniform(-1., 1., N) * V0
    vy = np.random.uniform(-1., 1., N) * V0
    rng = np.random.default_rng(seed * 17 + 3)
    alpha_arr = np.full(N, ALPHA)
    ramp_arr  = np.full(N, RAMP)
    for _ in range(N_WARMUP):
        x, y, vx, vy, _ = step(x, y, vx, vy, mode, k, rng, alpha_arr, ramp_arr)
    return x, y, vx, vy


def contact_sets(x, y):
    """List of frozensets: contact neighbors (within R_CONT) of each agent."""
    dx = x[np.newaxis, :] - x[:, np.newaxis]
    dy = y[np.newaxis, :] - y[:, np.newaxis]
    dx -= np.round(dx); dy -= np.round(dy)
    rd2 = dx**2 + dy**2
    within = (rd2 <= R_CONT**2) & (rd2 > 0)
    return [frozenset(np.where(row)[0]) for row in within], within.sum(axis=1)


def metric_degree_k(seed):
    """Settle a metric flock and return its mean alignment-neighbor count."""
    x, y, vx, vy = warmup(seed, 'metric', 0)
    _, _, d2 = pair_d2(x, y)
    not_self = ~np.eye(N, dtype=bool)
    deg = ((d2 <= RF**2) & not_self).sum(axis=1)
    return float(deg.mean())


# ---------------------------------------------------------------------------
# Diagnostic A -- neighbor-graph mixing rate
# ---------------------------------------------------------------------------
def measure_mixing(seed, mode, k):
    """Mean Jaccard dissimilarity of contact sets between successive snapshots."""
    x, y, vx, vy = warmup(seed, mode, k)
    rng = np.random.default_rng(seed * 29 + 7)
    alpha_arr = np.full(N, ALPHA)
    ramp_arr  = np.full(N, RAMP)

    prev = None
    diss = []
    deg_cv = []
    for i in range(MIX_STEPS):
        x, y, vx, vy, _ = step(x, y, vx, vy, mode, k, rng, alpha_arr, ramp_arr)
        if i % MIX_INTERVAL == 0:
            sets, deg = contact_sets(x, y)
            deg_cv.append(deg.std() / max(deg.mean(), 1e-9))
            if prev is not None:
                vals = []
                for a, b in zip(prev, sets):
                    u = len(a | b)
                    vals.append(1.0 - (len(a & b) / u if u else 1.0))
                diss.append(np.mean(vals))
            prev = sets
    return float(np.mean(diss)), float(np.mean(deg_cv))


# ---------------------------------------------------------------------------
# Diagnostic B -- SIS with random vs degree-targeted vaccination
# ---------------------------------------------------------------------------
def run_sis(x0, y0, vx0, vy0, mode, k, rng, is_immune):
    x = x0.copy(); y = y0.copy(); vx = vx0.copy(); vy = vy0.copy()
    is_panicked = np.zeros(N, dtype=bool)

    n0 = max(1, round(F0_FRAC * N))
    susceptible = np.where(~is_immune)[0]
    idx0 = (rng.choice(susceptible, size=n0, replace=False)
            if susceptible.size >= n0 else susceptible)
    is_panicked[idx0] = True

    p_recover = 1.0 - np.exp(-GAMMA * DT)
    last_window = N_ITER - int(20.0 / DT)
    f_series = []

    for i in range(N_ITER):
        alpha_arr = np.where(is_panicked, PANIC_ALPHA, ALPHA)
        ramp_arr  = np.where(is_panicked, PANIC_RAMP,  RAMP)
        x, y, vx, vy, _ = step(x, y, vx, vy, mode, k, rng, alpha_arr, ramp_arr)

        if is_panicked.any() and (~is_panicked & ~is_immune).any():
            dx = x[np.newaxis, :] - x[:, np.newaxis]
            dy = y[np.newaxis, :] - y[:, np.newaxis]
            dx -= np.round(dx); dy -= np.round(dy)
            within = (dx**2 + dy**2 <= R_CONT**2) & (dx**2 + dy**2 > 0)
            kcnt = within @ is_panicked.astype(np.int32)
            cs = np.where(~is_panicked & ~is_immune)[0]
            p_trans = 1.0 - np.exp(-BETA * kcnt[cs] * DT)
            flipped = cs[rng.uniform(0., 1., cs.size) < p_trans]
            if flipped.size:
                is_panicked[flipped] = True

        if is_panicked.any():
            pidx = np.where(is_panicked)[0]
            rec = pidx[rng.uniform(0., 1., pidx.size) < p_recover]
            if rec.size:
                is_panicked[rec] = False

        if i >= last_window and i % 50 == 0:
            f_series.append(is_panicked.mean())

    return float(np.mean(f_series)) if f_series else 0.0


def degree_order(x, y):
    dx = x[np.newaxis, :] - x[:, np.newaxis]
    dy = y[np.newaxis, :] - y[:, np.newaxis]
    dx -= np.round(dx); dy -= np.round(dy)
    rd2 = dx**2 + dy**2
    deg = ((rd2 <= R_CONT**2) & (rd2 > 0)).sum(axis=1)
    return np.argsort(-deg)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print('Finding 47 -- topological vs metric alignment')
    print('  testing the Section 5 prediction: k-NN alignment slows mixing')
    print('  and lets targeted vaccination recover an advantage')
    print()
    t0 = time.time()

    # Calibrate k = mean metric alignment degree
    k_samples = [metric_degree_k(s) for s in range(N_SEEDS)]
    K = int(round(np.mean(k_samples)))
    print('  calibrated k (mean metric alignment degree) = %d' % K)
    print()

    # --- Diagnostic A: mixing rate ---
    print('Diagnostic A -- neighbor-graph mixing rate')
    mix = {}
    for mode in ('metric', 'topo'):
        diss_list, cv_list = [], []
        for s in range(N_SEEDS):
            d, cv = measure_mixing(s, mode, K)
            diss_list.append(d); cv_list.append(cv)
        mix[mode] = (np.mean(diss_list), np.std(diss_list), np.mean(cv_list))
        print('  %-7s  Jaccard dissimilarity / 2tu = %.4f +/- %.4f   degree CV = %.3f' % (
              mode, mix[mode][0], mix[mode][1], mix[mode][2]))
    print()

    # --- Diagnostic B: vaccination ---
    print('Diagnostic B -- random vs degree-targeted vaccination')
    results = {}   # (mode, strategy) -> {p: [f_ss per seed]}
    for mode in ('metric', 'topo'):
        for strat in ('random', 'targeted'):
            results[(mode, strat)] = {p: [] for p in P_IMMUNE_LIST}

    for mode in ('metric', 'topo'):
        tm = time.time()
        for s in range(N_SEEDS):
            x0, y0, vx0, vy0 = warmup(s, mode, K)
            deg_sorted = degree_order(x0, y0)
            for p_im in P_IMMUNE_LIST:
                n_im = int(round(p_im * N))
                rng_r = np.random.default_rng(s * 1000 + int(p_im * 1000))
                im_r = np.zeros(N, dtype=bool)
                im_r[rng_r.choice(N, size=n_im, replace=False)] = True
                im_t = np.zeros(N, dtype=bool)
                im_t[deg_sorted[:n_im]] = True

                f_r = run_sis(x0, y0, vx0, vy0, mode, K,
                              np.random.default_rng(s*2000 + int(p_im*1000)), im_r)
                f_t = run_sis(x0, y0, vx0, vy0, mode, K,
                              np.random.default_rng(s*3000 + int(p_im*1000)), im_t)
                results[(mode, 'random')][p_im].append(f_r)
                results[(mode, 'targeted')][p_im].append(f_t)
        print('  %s mode done [%.0fs]' % (mode, time.time() - tm), flush=True)

    print()
    print('=== Diagnostic B results (f_ss: mean panic fraction) ===')
    summary = {}
    for mode in ('metric', 'topo'):
        print('  %s alignment:' % mode)
        print('    %8s  %16s  %16s  %10s' % (
              'p_immune', 'random', 'targeted', 'advantage'))
        for p in P_IMMUNE_LIST:
            r = np.array(results[(mode, 'random')][p])
            t = np.array(results[(mode, 'targeted')][p])
            adv = r.mean() - t.mean()   # positive => targeted helps
            summary[(mode, p)] = (r.mean(), r.std(), t.mean(), t.std(), adv)
            print('    %8.2f  %7.3f +/- %.3f  %7.3f +/- %.3f  %+9.3f' % (
                  p, r.mean(), r.std(), t.mean(), t.std(), adv))
    print()
    print('Total runtime: %.1f min' % ((time.time() - t0)/60.0))

    # --- Figure ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    p_arr = np.array(P_IMMUNE_LIST)

    for ax, mode in zip(axes, ('metric', 'topo')):
        r_m = [summary[(mode, p)][0] for p in P_IMMUNE_LIST]
        r_e = [summary[(mode, p)][1] for p in P_IMMUNE_LIST]
        t_m = [summary[(mode, p)][2] for p in P_IMMUNE_LIST]
        t_e = [summary[(mode, p)][3] for p in P_IMMUNE_LIST]
        ax.errorbar(p_arr, r_m, yerr=r_e, marker='o', capsize=4,
                    color='steelblue', label='random')
        ax.errorbar(p_arr, t_m, yerr=t_e, marker='s', capsize=4,
                    color='crimson', label='degree-targeted')
        ax.set_xlabel('Immune fraction p_immune')
        ax.set_ylabel('Steady-state panic fraction f_ss')
        ax.set_title('%s alignment\nmixing = %.3f Jaccard/2tu' % (
                     mode, mix[mode][0]))
        ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_ylim(-0.03, 0.7)
    fig.suptitle('Finding 47 -- topological alignment vs kinematic mixing')
    fig.tight_layout()
    fig.savefig('figures/finding47_topological_mixing.png', dpi=140)
    print('  -> figures/finding47_topological_mixing.png')

    with open('outputs/finding47_topological_mixing.txt', 'w') as f:
        f.write('Finding 47 -- topological vs metric alignment\n')
        f.write('N=%d seeds=%d  calibrated k=%d\n\n' % (N, N_SEEDS, K))
        f.write('Diagnostic A -- mixing rate (Jaccard dissimilarity per 2 tu):\n')
        for mode in ('metric', 'topo'):
            f.write('  %-7s  %.4f +/- %.4f   degree CV = %.3f\n' % (
                    mode, mix[mode][0], mix[mode][1], mix[mode][2]))
        f.write('\nDiagnostic B -- vaccination (f_ss):\n')
        for mode in ('metric', 'topo'):
            f.write('  %s alignment:\n' % mode)
            f.write('    p_immune  random  targeted  advantage(rand-targ)\n')
            for p in P_IMMUNE_LIST:
                rm, rs, tm_, ts, adv = summary[(mode, p)]
                f.write('    %.2f      %.4f  %.4f    %+.4f\n' % (p, rm, tm_, adv))
    print('  -> outputs/finding47_topological_mixing.txt')
