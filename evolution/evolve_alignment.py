"""
evolve_alignment.py -- co-adaptation thread, F93 candidate. A DIFFERENT heritable
trait: the alignment strength alpha, under capture/removal predation.

F87-F92 evolved the collective-escape weight. This evolves the flocking force
itself: each prey carries a heritable alignment strength alpha_i (the weight on the
velocity-alignment force), with no escape force at all (w = 0). The predators are
the F66 predictive encircler (lead 2, six of them, slow-prey regime), and selection
is capture/removal exactly as in F87: a prey within a kill radius of a predator is
captured at a fixed hazard and replaced by a mutated clone of a random survivor,
which inherits alpha + a small mutation; N is held fixed.

QUESTION. Does predation drive the flock toward MORE alignment (tighter flocking --
safety in numbers, coherent collective motion) or LESS (scatter, so that fewer prey
are near any one predator at a time -- the dilution effect of F7/F11)? The two
pressures are opposed: a tight flock moves coherently but presents many prey at once
to a predator that reaches it, while a loose swarm dilutes each predator's local
catch but loses the coherence that the alignment force provides. Whichever wins is
the evolutionary preference, and it is not obvious a priori -- hence the experiment.
Initial alpha0 is swept to see where the mean alignment strength converges.

Reuses the validated per-step physics; alpha is now a per-agent array.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from flocking import buffer as buf_fn, order_parameter
from model import _periodic_com, _periodic_disp
from vectorized_predator import predator_force

os.makedirs('figures', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

BASE = dict(N=150, r0=0.005, eps=0.1, rf=0.1, v0=0.02, mu=10.0, ramp=0.1, dt=0.01)
PRED = dict(v0=0.05, mu=10.0, alpha=5.0, r0=0.1, eps=2.0, ramp=1.0, enc_radius=0.15)
N_PRED, LEAD = 6, 2.0
N_WARMUP, N_EVOLVE = 800, 20000
R_KILL, CAPTURE_RATE = 0.03, 3.0
MUT_SIGMA, A_MIN, A_MAX = 0.10, 0.0, 4.0
JITTER, RECORD_EVERY = 0.004, 200

ALPHA0_VALUES = [0.2, 0.5, 1.0, 2.0]
N_SEEDS = 3


def run(alpha0, seed, alpha_init=None, mut_sigma=None, hi_thresh=1.0):
    """Evolve heritable alignment strength under capture/removal predation.

    Additive options (defaults reproduce the original sweep exactly):
      alpha_init -- seeded per-agent alpha array (overrides the uniform alpha0 start),
                    used by the invasion experiment (cf F88).
      mut_sigma  -- per-capture mutation step override (default MUT_SIGMA).
      hi_thresh  -- an agent counts as 'high-alignment' if alpha > this; the returned
                    frac_hi tracks that fraction (the alpha-analog of F88's escaper frac).
    """
    rng = np.random.RandomState(seed)
    p = BASE
    N, dt = p['N'], p['dt']
    r0, eps, rf = p['r0'], p['eps'], p['rf']
    v0, mu, ramp = p['v0'], p['mu'], p['ramp']
    ms = MUT_SIGMA if mut_sigma is None else mut_sigma

    x = np.zeros(2 * N)
    x[:N] = rng.uniform(0., 1., N); x[N:] = rng.uniform(0., 1., N)
    vx = rng.uniform(-1., 1., N) * v0; vy = rng.uniform(-1., 1., N) * v0
    if alpha_init is not None:
        alpha = np.asarray(alpha_init, dtype=float).copy()   # seeded minority start
    else:
        alpha = np.full(N, float(alpha0))          # heritable alignment strength

    pred_x = rng.uniform(0., 1., N_PRED); pred_y = rng.uniform(0., 1., N_PRED)
    pred_vx = np.zeros(N_PRED); pred_vy = np.zeros(N_PRED)
    ang = np.radians(np.arange(N_PRED) * 360.0 / N_PRED)
    cos_a, sin_a = np.cos(ang), np.sin(ang)

    rb = max(r0, rf); p_capture = CAPTURE_RATE * dt
    n_total = N_WARMUP + N_EVOLVE
    t_rec, a_mean, a_std, phi_rec, cap_rec, fhi_rec = [], [], [], [], [], []
    cum = 0

    for i in range(n_total):
        on = (i >= N_WARMUP)
        nb, xb, yb, vxb, vyb = buf_fn(rb, x, vx, vy, N)
        dx = xb[:nb][np.newaxis, :] - x[:N][:, np.newaxis]
        dy = yb[:nb][np.newaxis, :] - x[N:][:, np.newaxis]
        d2 = dx**2 + dy**2
        not_self = np.ones((N, nb), dtype=bool)
        idx = np.arange(min(N, nb)); not_self[idx, idx] = False

        flock_mask = (d2 <= rf**2) & not_self
        flx = np.where(flock_mask, vxb[:nb], 0.).sum(axis=1)
        fly = np.where(flock_mask, vyb[:nb], 0.).sum(axis=1)
        nfl = flock_mask.sum(axis=1); nrm = np.sqrt(flx**2 + fly**2); nrm[nfl == 0] = 1.
        flockx = alpha * flx / nrm; flocky = alpha * fly / nrm     # PER-AGENT alpha

        rep_mask = (d2 <= (2*r0)**2) & not_self & (d2 > 0)
        d_safe = np.where(rep_mask, np.sqrt(d2), 1.)
        base_r = np.where(rep_mask, 1. - d_safe/(2.*r0), 0.)
        strength = np.where(rep_mask, eps * base_r**1.5 / d_safe, 0.)
        repx = (-strength * dx).sum(axis=1); repy = (-strength * dy).sum(axis=1)

        vnorm = np.sqrt(vx**2 + vy**2); vns = np.where(vnorm == 0, 1., vnorm)
        fpropx = mu * (v0 - vnorm) * vx / vns; fpropy = mu * (v0 - vnorm) * vy / vns
        frandx = ramp * rng.uniform(-1., 1., N); frandy = ramp * rng.uniform(-1., 1., N)
        fx_total = flockx + repx + fpropx + frandx
        fy_total = flocky + repy + fpropy + frandy

        if on:
            pred_xy = np.column_stack((pred_x, pred_y))
            pfx, pfy = predator_force(pred_xy, x[:N], x[N:], PRED['r0'], PRED['eps'])
            fx_total += pfx; fy_total += pfy

        vx += fx_total * dt; vy += fy_total * dt
        x[:N] = (x[:N] + vx*dt) % 1.; x[N:] = (x[N:] + vy*dt) % 1.

        if on:
            cx, cy = _periodic_com(x[:N]), _periodic_com(x[N:])
            vmx, vmy = vx.mean(), vy.mean()
            tx = (cx + PRED['enc_radius']*cos_a + LEAD*vmx) % 1.
            ty = (cy + PRED['enc_radius']*sin_a + LEAD*vmy) % 1.
            ddx = _periodic_disp(tx, pred_x); ddy = _periodic_disp(ty, pred_y)
            dist = np.sqrt(ddx**2 + ddy**2); nz = dist > 0
            ddx = np.where(nz, ddx/np.where(nz, dist, 1.), ddx)
            ddy = np.where(nz, ddy/np.where(nz, dist, 1.), ddy)
            sp = np.sqrt(pred_vx**2 + pred_vy**2); sps = np.where(sp > 0, sp, 1.)
            drive = np.where(sp > 0, PRED['mu']*(PRED['v0']-sp)/sps, 0.)
            pfx2 = PRED['alpha']*ddx + drive*pred_vx; pfy2 = PRED['alpha']*ddy + drive*pred_vy
            pn = rng.uniform(-1., 1., (N_PRED, 2))
            pfx2 += PRED['ramp']*pn[:, 0]; pfy2 += PRED['ramp']*pn[:, 1]
            pred_vx += pfx2*dt; pred_vy += pfy2*dt
            pred_x = (pred_x + pred_vx*dt) % 1.; pred_y = (pred_y + pred_vy*dt) % 1.

            ddxp = _periodic_disp(pred_x[:, None], x[:N][None, :])
            ddyp = _periodic_disp(pred_y[:, None], x[N:][None, :])
            min_pred = np.sqrt(ddxp**2 + ddyp**2).min(axis=0)
            at_risk = min_pred < R_KILL
            captured = at_risk & (rng.uniform(0., 1., N) < p_capture)
            cap_idx = np.where(captured)[0]; surv_idx = np.where(~captured)[0]
            if cap_idx.size > 0 and surv_idx.size > 0:
                parents = rng.choice(surv_idx, size=cap_idx.size, replace=True)
                alpha[cap_idx] = np.clip(alpha[parents] + rng.normal(0., ms, cap_idx.size), A_MIN, A_MAX)
                x[cap_idx]   = (x[parents] + rng.uniform(-JITTER, JITTER, cap_idx.size)) % 1.
                x[N+cap_idx] = (x[N+parents] + rng.uniform(-JITTER, JITTER, cap_idx.size)) % 1.
                vx[cap_idx] = vx[parents]; vy[cap_idx] = vy[parents]
                cum += cap_idx.size

            if (i - N_WARMUP) % RECORD_EVERY == 0:
                t_rec.append((i - N_WARMUP) * dt)
                a_mean.append(alpha.mean()); a_std.append(alpha.std())
                phi_rec.append(order_parameter(vx, vy)); cap_rec.append(cum)
                fhi_rec.append(float(np.mean(alpha > hi_thresh)))

    return dict(t=np.array(t_rec), a_mean=np.array(a_mean), a_std=np.array(a_std),
                phi=np.array(phi_rec), cum_cap=np.array(cap_rec),
                frac_hi=np.array(fhi_rec),
                a_final=alpha.copy(), alpha0=alpha0, seed=seed)


def steady(a):
    return float(np.mean(a[-len(a)//3:]))


def main():
    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('Evolving the ALIGNMENT strength alpha under capture/removal predation (predictive predator)')
    out('  N=%d  n_pred=%d  lead=%.1f  %d tu  %d seeds  no escape force (w=0); alpha is the heritable trait'
        % (BASE['N'], N_PRED, LEAD, N_EVOLVE*BASE['dt'], N_SEEDS))
    out('  Question: does predation select for MORE alignment (tight flock) or LESS (scatter/dilution)?\n')

    res = {}
    for a0 in ALPHA0_VALUES:
        runs = [run(a0, s) for s in range(N_SEEDS)]
        res[a0] = runs
        aS = np.mean([steady(r['a_mean']) for r in runs])
        aS_sd = np.std([steady(r['a_mean']) for r in runs])
        phi = np.mean([steady(r['phi']) for r in runs])
        cap = np.mean([r['cum_cap'][-1] for r in runs])
        out('  alpha0=%.2f -> steady mean alpha=%.2f (+/-%.2f)  Phi=%.2f  captures=%.0f'
            % (a0, aS, aS_sd, phi, cap))

    out('')
    finals = {a0: np.mean([steady(r['a_mean']) for r in res[a0]]) for a0 in ALPHA0_VALUES}
    lo, hi = finals[ALPHA0_VALUES[0]], finals[ALPHA0_VALUES[-1]]
    conv = abs(hi - lo) < 0.4
    if conv:
        ess = 0.5 * (lo + hi)
        direction = ('UP toward tighter flocking' if ess > 1.1 else
                     'DOWN toward scatter' if ess < 0.9 else 'to an intermediate value')
        out('  -> Alignment strength converges to a common ESS ~%.2f from both low and high starts (%.2f, %.2f):'
            % (ess, lo, hi))
        out('     predation selects %s. Convergence from both sides indicates a genuine attractor.' % direction)
    else:
        out('  -> Outcome depends on the start (low->%.2f, high->%.2f): history-dependent, like the escape'
            % (lo, hi))
        out('     weight (F87 hysteresis). See the trajectory figure.')

    # ---------------------------------------------------------------- figure
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    colors = plt.cm.plasma(np.linspace(0, 0.85, len(ALPHA0_VALUES)))
    for a0, c in zip(ALPHA0_VALUES, colors):
        for r in res[a0]:
            ax[0].plot(r['t'], r['a_mean'], color=c, alpha=0.8,
                       label='alpha0=%.2f' % a0 if r['seed'] == 0 else None)
    ax[0].axhline(1.0, ls='--', color='gray', alpha=0.6, label='alpha=1 (default)')
    ax[0].set_xlabel('time (tu)'); ax[0].set_ylabel('population mean alignment strength alpha')
    ax[0].set_title('Evolution of alignment strength under predation'); ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    a0s = np.array(ALPHA0_VALUES)
    aE = np.array([np.mean([steady(r['a_mean']) for r in res[a0]]) for a0 in ALPHA0_VALUES])
    pE = np.array([np.mean([steady(r['phi']) for r in res[a0]]) for a0 in ALPHA0_VALUES])
    ax[1].plot(a0s, aE, 'o-', lw=2, color='teal', label='steady mean alpha')
    ax[1].plot(a0s, a0s, ls=':', color='gray', alpha=0.5, label='no change (y=x)')
    ax[1].plot(a0s, pE, 's--', lw=1.5, color='orange', alpha=0.7, label='steady Phi')
    ax[1].set_xlabel('initial alpha0'); ax[1].set_ylabel('steady value')
    ax[1].set_title('Initial vs evolved alignment strength'); ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    fig.suptitle('Heritable alignment strength under predation (N=%d, %d seeds)' % (BASE['N'], N_SEEDS), fontsize=10)
    plt.tight_layout()
    plt.savefig('figures/evolve_alignment_1.png', dpi=120)
    plt.close()
    out('\n  --> figures/evolve_alignment_1.png')
    with open('outputs/evolve_alignment.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    out('  --> outputs/evolve_alignment.txt')
    out('\nAlignment-evolution experiment complete.')


if __name__ == '__main__':
    main()
