# infectiousness_heterogeneity.py -- SIS contagion with heterogeneous per-agent
# transmission rates beta_i.
#
# F54 established that heterogeneous RECOVERY rates (per-agent gamma_i, fixed
# arithmetic mean) lower the SIS epidemic threshold: slow recoverers act as
# reservoirs.  This script asks the dual question for the source side of the
# transmission: what if every agent recovers at the same gamma, but each agent
# carries its own transmission rate beta_i?  The arithmetic mean of beta is
# held fixed; what changes is the SPREAD.
#
# Mean-field expectation differs from F54.  For SIS on a well-mixed graph the
# endemic state depends on the average product beta*<k>/gamma.  At fixed mean
# beta and homogeneous gamma there is no straightforward Jensen-style inequality
# pushing the threshold down -- a high-beta agent contributes more, a low-beta
# agent less, and the leading effect cancels.  But on a SPATIAL contact graph
# the high-beta minority can act as "super-spreaders": once panicked they seed
# their entire neighbourhood faster than recovery can prune it, even if their
# population fraction is small.  Whether this lowers beta_c or leaves it
# unchanged is the question.
#
# Conditions (all share arithmetic-mean beta = beta_target):
#   homog   : every beta_i = beta_target
#   mild    : bimodal {beta - 0.5, beta + 0.5}  (50/50)
#   strong  : bimodal {beta - 0.8, beta + 0.8}  (50/50)
#   extreme : bimodal {0.05*beta, 1.95*beta}    (50/50; multiplicative)
#
# Experiments:
#   1. Beta sweep for each condition -- threshold shift + f_ss elevation
#   2. Source check: panic-source share contributed by high-beta vs low-beta
#      agents (panic-frame ownership)
#   3. Heterogeneity sweep at fixed near-threshold mean beta

import os
import numpy as np
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from flocking import buffer as buf_fn, order_parameter

os.makedirs('figures', exist_ok=True)

N_SEEDS = 4

BASE = dict(
    N=350, r0=0.005, eps=0.1, rf=0.1,
    alpha=1.0, v0=1.0, mu=10.0, ramp=0.5, dt=0.01,
    n_iter=4000,
)
PANIC_ALPHA = 0.1
PANIC_RAMP  = 10.0
R_CONT      = 0.05
GAMMA       = 1.0


def make_beta(N, beta_mean, spread_kind, rng):
    """Per-agent transmission rates with arithmetic mean = beta_mean.
    spread_kind in {'homog', 'mild', 'strong', 'extreme'}.
    Returns (beta_arr, is_super) where is_super flags the high-beta half."""
    beta_arr = np.full(N, beta_mean, dtype=float)
    is_super = np.zeros(N, dtype=bool)
    if spread_kind == 'homog':
        return beta_arr, is_super
    super_idx = rng.choice(N, size=N // 2, replace=False)
    is_super[super_idx] = True
    if spread_kind == 'mild':
        delta = 0.5
        beta_arr[is_super]  = max(0., beta_mean + delta)
        beta_arr[~is_super] = max(0., beta_mean - delta)
    elif spread_kind == 'strong':
        delta = 0.8
        beta_arr[is_super]  = max(0., beta_mean + delta)
        beta_arr[~is_super] = max(0., beta_mean - delta)
    elif spread_kind == 'extreme':
        beta_arr[is_super]  = 1.95 * beta_mean
        beta_arr[~is_super] = 0.05 * beta_mean
    # renormalise to exact arithmetic mean
    beta_arr *= beta_mean / max(beta_arr.mean(), 1e-12)
    return beta_arr, is_super


def run_sis_beta_het(beta_mean=1.0, spread_kind='homog', f0=0.05,
                    n_frames=200, seed=None):
    """SIS contagion with per-agent transmission beta_i, homogeneous gamma.
    Returns frames where each frame carries (x, y, vx, vy, is_panicked,
    is_super, k_panic_neighbors_per_calm) -- the last is for source-attribution."""
    rng = np.random.RandomState(seed)

    p = BASE.copy()
    N      = p['N']
    dt     = p['dt']
    n_iter = p['n_iter']
    r0, eps, rf = p['r0'], p['eps'], p['rf']
    v0, mu       = p['v0'], p['mu']
    frame_every  = max(1, n_iter // n_frames)

    beta_arr, is_super = make_beta(N, beta_mean, spread_kind, rng)
    p_recover_per_step = 1. - np.exp(-GAMMA * dt)

    n0 = max(1, round(f0 * N)) if f0 > 0 else 0
    is_panicked = np.zeros(N, dtype=bool)
    if n0 > 0:
        seed_idx = rng.choice(N, size=n0, replace=False)
        is_panicked[seed_idx] = True

    x  = np.zeros(2*N)
    x[:N] = rng.uniform(0., 1., N)
    x[N:] = rng.uniform(0., 1., N)
    vx = rng.uniform(-1., 1., N) * v0
    vy = rng.uniform(-1., 1., N) * v0

    rb = max(r0, rf, R_CONT)
    frames = []

    # cumulative attributions: how many transmissions did super vs normal agents source
    source_super = 0
    source_normal = 0

    for i in range(n_iter):
        nb, xb, yb, vxb, vyb = buf_fn(rb, x, vx, vy, N)

        dx = xb[:nb][np.newaxis, :] - x[:N][:, np.newaxis]
        dy = yb[:nb][np.newaxis, :] - x[N:][:, np.newaxis]
        d2 = dx**2 + dy**2

        not_self = np.ones((N, nb), dtype=bool)
        idx = np.arange(min(N, nb))
        not_self[idx, idx] = False

        alpha_arr = np.where(is_panicked, PANIC_ALPHA, p['alpha'])
        ramp_arr  = np.where(is_panicked, PANIC_RAMP,  p['ramp'])

        flock_mask = (d2 <= rf**2) & not_self
        flx = np.where(flock_mask, vxb[:nb], 0.).sum(axis=1)
        fly = np.where(flock_mask, vyb[:nb], 0.).sum(axis=1)
        nfl = flock_mask.sum(axis=1)
        nrm = np.sqrt(flx**2 + fly**2)
        nrm[nfl == 0] = 1.
        flockx = alpha_arr * flx / nrm
        flocky = alpha_arr * fly / nrm

        rep_mask = (d2 <= (2*r0)**2) & not_self & (d2 > 0)
        d_safe   = np.where(rep_mask, np.sqrt(d2), 1.)
        base     = np.where(rep_mask, 1. - d_safe/(2.*r0), 0.)
        strength = np.where(rep_mask, eps * base**1.5 / d_safe, 0.)
        repx = (-strength * dx).sum(axis=1)
        repy = (-strength * dy).sum(axis=1)

        vnorm  = np.sqrt(vx**2 + vy**2)
        vnorms = np.where(vnorm == 0, 1., vnorm)
        fpropx = mu * (v0 - vnorm) * vx / vnorms
        fpropy = mu * (v0 - vnorm) * vy / vnorms

        frandx = ramp_arr * rng.uniform(-1., 1., N)
        frandy = ramp_arr * rng.uniform(-1., 1., N)

        fx = flockx + repx + fpropx + frandx
        fy = flocky + repy + fpropy + frandy

        vx += fx * dt
        vy += fy * dt
        x[:N] = (x[:N] + vx*dt) % 1.
        x[N:] = (x[N:] + vy*dt) % 1.

        # contagion using per-source beta_i. A calm j gets infected with prob
        # 1 - exp(-sum_i beta_i * dt) over panicked neighbours i.
        if is_panicked.any() and (~is_panicked).any():
            real_dx = x[:N][np.newaxis, :] - x[:N][:, np.newaxis]
            real_dy = x[N:][np.newaxis, :] - x[N:][:, np.newaxis]
            real_dx -= np.round(real_dx); real_dy -= np.round(real_dy)
            rd2 = real_dx**2 + real_dy**2
            within = (rd2 <= R_CONT**2) & (rd2 > 0)
            # contribution_ij = within_ij * is_panicked_j * beta_j
            beta_contrib = within * (is_panicked.astype(float) * beta_arr)[np.newaxis, :]
            lam = beta_contrib.sum(axis=1)  # per-calm hazard rate
            calm_idx = np.where(~is_panicked)[0]
            if calm_idx.size:
                lam_calm = lam[calm_idx]
                p_trans = 1. - np.exp(-lam_calm * dt)
                r = rng.uniform(0., 1., calm_idx.size)
                flipped = calm_idx[r < p_trans]
                if flipped.size:
                    # attribute source: pick a panicked neighbour weighted by beta
                    for j in flipped:
                        contribs = beta_contrib[j]
                        tot = contribs.sum()
                        if tot > 0:
                            pick = rng.choice(N, p=contribs / tot)
                            if is_super[pick]: source_super += 1
                            else:              source_normal += 1
                    is_panicked[flipped] = True

        if is_panicked.any():
            panic_idx = np.where(is_panicked)[0]
            r = rng.uniform(0., 1., panic_idx.size)
            recovered = panic_idx[r < p_recover_per_step]
            if recovered.size:
                is_panicked[recovered] = False

        if i % frame_every == 0:
            frames.append((x[:N].copy(), x[N:].copy(),
                           vx.copy(), vy.copy(),
                           is_panicked.copy(), is_super.copy()))

    attribution = (source_super, source_normal)
    return frames, attribution


def summarize(frames, ss_tail=30):
    last = frames[-ss_tail:]
    f_ss = np.mean([m.mean() for _, _, _, _, m, _ in last])
    phi_ss = np.mean([order_parameter(vx, vy) for _, _, vx, vy, _, _ in last])
    super_f, normal_f = [], []
    for _, _, _, _, m, sup in last:
        if sup.any():
            super_f.append(m[sup].mean())
            normal_f.append(m[~sup].mean())
    super_f = np.mean(super_f) if super_f else float('nan')
    normal_f = np.mean(normal_f) if normal_f else float('nan')
    return f_ss, phi_ss, super_f, normal_f


CONDITIONS = ['homog', 'mild', 'strong', 'extreme']

# =============================================================================
# EXP 1: BETA SWEEP for each heterogeneity condition
# =============================================================================
print('Exp 1: beta sweep, gamma=1.0, %d seeds' % N_SEEDS)
print('  (per-agent beta_i; arithmetic mean held fixed at sweep value)')
beta_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
res1 = {}
for kind in CONDITIONS:
    res1[kind] = {}
    print('  -- %s --' % kind)
    for beta in beta_vals:
        fs = []
        for s in range(N_SEEDS):
            frames, _ = run_sis_beta_het(beta_mean=beta, spread_kind=kind,
                                         f0=0.05, seed=s)
            f_ss, _, _, _ = summarize(frames)
            fs.append(f_ss)
        res1[kind][beta] = (np.mean(fs), np.std(fs))
        print('     beta=%4.1f  f_ss=%.3f +/- %.3f' % (beta, np.mean(fs), np.std(fs)))


def threshold_cross(betas, fvals, level=0.15):
    for j in range(1, len(betas)):
        if fvals[j-1] < level <= fvals[j]:
            t = (level - fvals[j-1]) / (fvals[j] - fvals[j-1])
            return betas[j-1] + t * (betas[j] - betas[j-1])
    return float('nan')

print('\n  Threshold beta (f_ss crosses 0.15):')
for kind in CONDITIONS:
    betas = beta_vals
    fvals = [res1[kind][b][0] for b in betas]
    bc = threshold_cross(betas, fvals)
    print('     %-8s beta_c = %.3f' % (kind, bc))

# =============================================================================
# EXP 2: SOURCE ATTRIBUTION -- who does the spreading
# =============================================================================
print('\nExp 2: source attribution at beta_mean=1.0')
res2 = {}
attr_totals = {}
for kind in CONDITIONS:
    if kind == 'homog':
        continue
    super_v, normal_v = [], []
    src_sup, src_nor = 0, 0
    for s in range(N_SEEDS):
        frames, attr = run_sis_beta_het(beta_mean=1.0, spread_kind=kind,
                                        f0=0.05, seed=s)
        _, _, super_f, normal_f = summarize(frames)
        super_v.append(super_f); normal_v.append(normal_f)
        src_sup += attr[0]; src_nor += attr[1]
    res2[kind] = (np.mean(super_v), np.mean(normal_v))
    attr_totals[kind] = (src_sup, src_nor)
    tot = max(1, src_sup + src_nor)
    print('  %-8s super-agent f=%.3f  normal-agent f=%.3f  '
          'source share super=%.1f%%' % (
        kind, np.mean(super_v), np.mean(normal_v), 100. * src_sup / tot))

# =============================================================================
# EXP 3: HETEROGENEITY SWEEP at fixed sub-threshold mean beta
# =============================================================================
print('\nExp 3: heterogeneity sweep at beta_mean=0.5 (near-threshold)')
beta_het = 0.5
delta_vals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5]

def make_beta_delta(N, beta_mean, delta, rng):
    beta_arr = np.full(N, beta_mean, dtype=float)
    is_super = np.zeros(N, dtype=bool)
    if delta > 0:
        idx = rng.choice(N, size=N//2, replace=False)
        is_super[idx] = True
        beta_arr[is_super]  = max(0., beta_mean + delta)
        beta_arr[~is_super] = max(0., beta_mean - delta)
        beta_arr *= beta_mean / max(beta_arr.mean(), 1e-12)
    return beta_arr, is_super


def run_sis_beta_delta(beta_mean, delta, f0=0.05, n_frames=200, seed=None):
    rng = np.random.RandomState(seed)
    p = BASE.copy()
    N      = p['N']; dt = p['dt']; n_iter = p['n_iter']
    r0, eps, rf = p['r0'], p['eps'], p['rf']
    v0, mu = p['v0'], p['mu']
    frame_every = max(1, n_iter // n_frames)
    beta_arr, is_super = make_beta_delta(N, beta_mean, delta, rng)
    p_recover_per_step = 1. - np.exp(-GAMMA * dt)
    n0 = max(1, round(f0 * N)) if f0 > 0 else 0
    is_panicked = np.zeros(N, dtype=bool)
    if n0 > 0:
        is_panicked[rng.choice(N, size=n0, replace=False)] = True
    x  = np.zeros(2*N)
    x[:N] = rng.uniform(0., 1., N); x[N:] = rng.uniform(0., 1., N)
    vx = rng.uniform(-1., 1., N) * v0
    vy = rng.uniform(-1., 1., N) * v0
    rb = max(r0, rf, R_CONT)
    frames = []
    for i in range(n_iter):
        nb, xb, yb, vxb, vyb = buf_fn(rb, x, vx, vy, N)
        dx = xb[:nb][np.newaxis, :] - x[:N][:, np.newaxis]
        dy = yb[:nb][np.newaxis, :] - x[N:][:, np.newaxis]
        d2 = dx**2 + dy**2
        not_self = np.ones((N, nb), dtype=bool)
        idx0 = np.arange(min(N, nb)); not_self[idx0, idx0] = False
        alpha_arr = np.where(is_panicked, PANIC_ALPHA, p['alpha'])
        ramp_arr  = np.where(is_panicked, PANIC_RAMP,  p['ramp'])
        flock_mask = (d2 <= rf**2) & not_self
        flx = np.where(flock_mask, vxb[:nb], 0.).sum(axis=1)
        fly = np.where(flock_mask, vyb[:nb], 0.).sum(axis=1)
        nfl = flock_mask.sum(axis=1)
        nrm = np.sqrt(flx**2 + fly**2); nrm[nfl == 0] = 1.
        flockx = alpha_arr * flx / nrm; flocky = alpha_arr * fly / nrm
        rep_mask = (d2 <= (2*r0)**2) & not_self & (d2 > 0)
        d_safe = np.where(rep_mask, np.sqrt(d2), 1.)
        base = np.where(rep_mask, 1. - d_safe/(2.*r0), 0.)
        strength = np.where(rep_mask, eps * base**1.5 / d_safe, 0.)
        repx = (-strength * dx).sum(axis=1)
        repy = (-strength * dy).sum(axis=1)
        vnorm = np.sqrt(vx**2 + vy**2)
        vnorms = np.where(vnorm == 0, 1., vnorm)
        fpropx = mu * (v0 - vnorm) * vx / vnorms
        fpropy = mu * (v0 - vnorm) * vy / vnorms
        frandx = ramp_arr * rng.uniform(-1., 1., N)
        frandy = ramp_arr * rng.uniform(-1., 1., N)
        fx = flockx + repx + fpropx + frandx
        fy = flocky + repy + fpropy + frandy
        vx += fx * dt; vy += fy * dt
        x[:N] = (x[:N] + vx*dt) % 1.; x[N:] = (x[N:] + vy*dt) % 1.
        if is_panicked.any() and (~is_panicked).any():
            rdx = x[:N][np.newaxis, :] - x[:N][:, np.newaxis]
            rdy = x[N:][np.newaxis, :] - x[N:][:, np.newaxis]
            rdx -= np.round(rdx); rdy -= np.round(rdy)
            rd2 = rdx**2 + rdy**2
            within = (rd2 <= R_CONT**2) & (rd2 > 0)
            lam = (within * (is_panicked.astype(float) * beta_arr)[np.newaxis, :]).sum(axis=1)
            calm_idx = np.where(~is_panicked)[0]
            if calm_idx.size:
                p_trans = 1. - np.exp(-lam[calm_idx] * dt)
                r = rng.uniform(0., 1., calm_idx.size)
                flipped = calm_idx[r < p_trans]
                if flipped.size:
                    is_panicked[flipped] = True
        if is_panicked.any():
            panic_idx = np.where(is_panicked)[0]
            r = rng.uniform(0., 1., panic_idx.size)
            recovered = panic_idx[r < p_recover_per_step]
            if recovered.size:
                is_panicked[recovered] = False
        if i % frame_every == 0:
            frames.append((x[:N].copy(), x[N:].copy(),
                           vx.copy(), vy.copy(),
                           is_panicked.copy(), is_super.copy()))
    return frames


res3 = {}
for delta in delta_vals:
    fs = []
    for s in range(N_SEEDS):
        frames = run_sis_beta_delta(beta_mean=beta_het, delta=delta, f0=0.05, seed=s)
        f_ss, _, _, _ = summarize(frames)
        fs.append(f_ss)
    res3[delta] = (np.mean(fs), np.std(fs))
    print('  delta=%.2f  f_ss=%.3f +/- %.3f' % (delta, np.mean(fs), np.std(fs)))

# =============================================================================
# FIGURES
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('SIS with heterogeneous infectiousness (N=350, gamma=1.0, %d seeds)'
             % N_SEEDS, fontsize=11)

ax = axes[0]
colors = {'homog': 'black', 'mild': 'goldenrod',
          'strong': 'darkorange', 'extreme': 'crimson'}
for kind in CONDITIONS:
    fvals = [res1[kind][b][0] for b in beta_vals]
    fstd  = [res1[kind][b][1] for b in beta_vals]
    ax.errorbar(beta_vals, fvals, yerr=fstd, fmt='o-', lw=2, capsize=3,
                color=colors[kind], label=kind)
ax.axhline(0.15, ls=':', color='gray', lw=1)
ax.set_xlabel('Mean transmission rate beta'); ax.set_ylabel('steady-state panic frac')
ax.set_title('Beta sweep by heterogeneity')
ax.set_ylim(-0.05, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = axes[1]
kinds = list(res2.keys())
super_v = [res2[n][0] for n in kinds]
normal_v = [res2[n][1] for n in kinds]
share_sup = []
for k in kinds:
    s, n = attr_totals[k]
    share_sup.append(100. * s / max(1, s + n))
xp = np.arange(len(kinds))
ax.bar(xp - 0.2, super_v, 0.4, color='crimson', label='super f')
ax.bar(xp + 0.2, normal_v, 0.4, color='steelblue', label='normal f')
for j, sh in enumerate(share_sup):
    ax.text(j, 0.95, 'src %.0f%%' % sh, ha='center', fontsize=8)
ax.set_xticks(xp); ax.set_xticklabels(kinds)
ax.set_ylabel('panic fraction'); ax.set_title('Source attribution (beta_mean=1.0)')
ax.set_ylim(0, 1.0); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

ax = axes[2]
fv  = [res3[d][0] for d in delta_vals]
fvs = [res3[d][1] for d in delta_vals]
ax.errorbar(delta_vals, fv, yerr=fvs, fmt='o-', color='purple', lw=2, capsize=3)
ax.set_xlabel('beta spread (bimodal half-width)')
ax.set_ylabel('steady-state panic frac')
ax.set_title('Heterogeneity sweep (beta_mean=0.5)')
ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/infectiousness_heterogeneity_1.png', dpi=120)
plt.close()
print('\n  --> figures/infectiousness_heterogeneity_1.png')
print('\nHeterogeneous-infectiousness SIS analysis complete.')
