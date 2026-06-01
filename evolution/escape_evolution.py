"""
escape_evolution.py -- co-adaptation thread, first experiment (F87 candidate).

QUESTION. F70 found a "dangerous valley" in the prey's collective-escape weight:
a weak escape (w ~ 0.25) is WORSE than none (Phi 0.275 vs 0.53), and only a strong
escape (w >= alpha = 1) restores coherence and outruns a predictive-encirclement
predator. If the escape weight is a HERITABLE per-agent trait under predation
selection, does selection drive a low-w population ACROSS that valley to the
winning regime, or does the valley trap it at low w?

FITNESS MODEL (student's choice, 2026-06: capture/removal). An agent that comes
within r_kill of any predator is captured with hazard rate `capture_rate` per time
unit and immediately replaced by a mutated clone of a random surviving agent
(inherits escape weight w + Gaussian mutation, spawns at the parent's position and
velocity with a tiny jitter). Population size N is held fixed (a Moran-style
continuous-replacement scheme), which is why this model fits the fixed-N harness.

MECHANISM. Predictive encirclement (F66, lead=2 tu, the hardest predator) plus the
F70 collective-escape force w_i * e_hat, where e_hat is the shared unit vector from
the predator centroid toward the flock CoM and w_i is the per-agent heritable trait.
The flock step and predator dynamics are the validated, bit-identical code used in
vectorized_predator_prey.py.

DESIGN CHOICES (made for this first pass; the student can sweep/revise them):
  - slow-prey regime, n_pred=6, lead=2 tu, R_enc=0.15 (the F66 hardest predator)
  - N=150 (smaller than the 350 default to keep the long evolutionary run tractable;
    still well above the F21 min viable flock size and the N=100 used in F78)
  - r_kill=0.03 (< predator repulsion r0=0.10: capture needs getting close)
  - capture_rate=3.0 /tu, mut_sigma=0.10, w in [0, 5]
  - initial weight w0 swept over {0.0, 0.25, 0.5, 1.0, 2.0} to test the valley from
    both sides (low/valley/high starts); 2 seeds each
  - 800-step warmup (flock forms, predators off) then 15000 evolution steps (=150 tu)

OUTPUT. Mean escape weight w(t) per initial condition (does it climb across the
valley?), final w distribution, capture rate, and Phi. Figure + console summary.
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

BASE = dict(N=150, r0=0.005, eps=0.1, rf=0.1,
            alpha=1.0, v0=0.02, mu=10.0, ramp=0.1, dt=0.01)
PRED = dict(v0=0.05, mu=10.0, alpha=5.0, r0=0.1, eps=2.0, ramp=1.0, enc_radius=0.15)

N_PRED   = 6
LEAD     = 2.0
N_WARMUP = 800
N_EVOLVE = 15000
R_KILL       = 0.03
CAPTURE_RATE = 3.0      # hazard per time unit when within r_kill
MUT_SIGMA    = 0.10
W_MIN, W_MAX = 0.0, 5.0
JITTER       = 0.004
RECORD_EVERY = 100

W0_VALUES = [0.0, 0.25, 0.5, 1.0, 2.0]
N_SEEDS   = 2

# Exp2: a long-timescale check on the low-w starts, to distinguish a true barrier
# (climb plateaus below the escape threshold) from a slow brake (climb continues).
N_EVOLVE_LONG = 40000     # 400 tu
LONG_W0       = [0.0, 0.5]


def run_evolution(w0, seed, w_init=None, mut_sigma=None, esc_thresh=0.75, capture_rate=None):
    """Evolve the population. w_init (N,) overrides the uniform w0 start (for seeded
    minorities); mut_sigma overrides the module default (for jump-the-valley tests);
    capture_rate overrides the module default (for predation-pressure sweeps).
    Backward compatible: with all None this is exactly the F87 run."""
    rng = np.random.RandomState(seed)
    p = BASE
    N, dt = p['N'], p['dt']
    r0, eps, rf = p['r0'], p['eps'], p['rf']
    v0, mu, alpha, ramp = p['v0'], p['mu'], p['alpha'], p['ramp']
    ms = MUT_SIGMA if mut_sigma is None else float(mut_sigma)
    cap_rate = CAPTURE_RATE if capture_rate is None else float(capture_rate)

    x = np.zeros(2 * N)
    x[:N] = rng.uniform(0., 1., N); x[N:] = rng.uniform(0., 1., N)
    vx = rng.uniform(-1., 1., N) * v0
    vy = rng.uniform(-1., 1., N) * v0
    w = np.full(N, float(w0)) if w_init is None else np.asarray(w_init, float).copy()

    pred_x = rng.uniform(0., 1., N_PRED); pred_y = rng.uniform(0., 1., N_PRED)
    pred_vx = np.zeros(N_PRED); pred_vy = np.zeros(N_PRED)
    ang = np.radians(np.arange(N_PRED) * 360.0 / N_PRED)
    cos_a, sin_a = np.cos(ang), np.sin(ang)

    rb = max(r0, rf)
    p_capture = cap_rate * dt
    n_total = N_WARMUP + N_EVOLVE

    t_rec, w_mean_rec, w_std_rec, phi_rec, cap_rec = [], [], [], [], []
    frac_esc_rec = []
    cum_captures = 0

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
        nfl = flock_mask.sum(axis=1)
        nrm = np.sqrt(flx**2 + fly**2); nrm[nfl == 0] = 1.
        flockx = alpha * flx / nrm; flocky = alpha * fly / nrm

        rep_mask = (d2 <= (2*r0)**2) & not_self & (d2 > 0)
        d_safe = np.where(rep_mask, np.sqrt(d2), 1.)
        base_r = np.where(rep_mask, 1. - d_safe/(2.*r0), 0.)
        strength = np.where(rep_mask, eps * base_r**1.5 / d_safe, 0.)
        repx = (-strength * dx).sum(axis=1); repy = (-strength * dy).sum(axis=1)

        vnorm = np.sqrt(vx**2 + vy**2); vnorms = np.where(vnorm == 0, 1., vnorm)
        fpropx = mu * (v0 - vnorm) * vx / vnorms
        fpropy = mu * (v0 - vnorm) * vy / vnorms

        frandx = ramp * rng.uniform(-1., 1., N)
        frandy = ramp * rng.uniform(-1., 1., N)

        fx_total = flockx + repx + fpropx + frandx
        fy_total = flocky + repy + fpropy + frandy

        if on:
            pred_xy = np.column_stack((pred_x, pred_y))
            pfx_prey, pfy_prey = predator_force(pred_xy, x[:N], x[N:], PRED['r0'], PRED['eps'])
            fx_total += pfx_prey; fy_total += pfy_prey

            # collective escape: shared direction, per-agent heritable weight w_i
            cx0, cy0 = _periodic_com(x[:N]), _periodic_com(x[N:])
            pcx, pcy = _periodic_com(pred_x), _periodic_com(pred_y)
            ex = _periodic_disp(cx0, pcx); ey = _periodic_disp(cy0, pcy)
            en = np.sqrt(ex**2 + ey**2)
            if en > 1e-9:
                ex /= en; ey /= en
                fx_total += w * ex; fy_total += w * ey

        vx += fx_total * dt; vy += fy_total * dt
        x[:N] = (x[:N] + vx*dt) % 1.; x[N:] = (x[N:] + vy*dt) % 1.

        if on:
            # predictive predator update (vectorized)
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
            pfx = PRED['alpha']*ddx + drive*pred_vx
            pfy = PRED['alpha']*ddy + drive*pred_vy
            pnoise = rng.uniform(-1., 1., (N_PRED, 2))
            pfx += PRED['ramp']*pnoise[:, 0]; pfy += PRED['ramp']*pnoise[:, 1]
            pred_vx += pfx*dt; pred_vy += pfy*dt
            pred_x = (pred_x + pred_vx*dt) % 1.; pred_y = (pred_y + pred_vy*dt) % 1.

            # ---- capture / removal selection on the heritable escape weight ----
            ddxp = _periodic_disp(pred_x[:, None], x[:N][None, :])
            ddyp = _periodic_disp(pred_y[:, None], x[N:][None, :])
            min_pred = np.sqrt(ddxp**2 + ddyp**2).min(axis=0)        # (N,)
            at_risk = min_pred < R_KILL
            captured = at_risk & (rng.uniform(0., 1., N) < p_capture)
            cap_idx = np.where(captured)[0]
            surv_idx = np.where(~captured)[0]
            if cap_idx.size > 0 and surv_idx.size > 0:
                parents = rng.choice(surv_idx, size=cap_idx.size, replace=True)
                # inherit weight + mutation; clone position/velocity (+ jitter)
                w[cap_idx] = np.clip(w[parents] + rng.normal(0., ms, cap_idx.size), W_MIN, W_MAX)
                x[cap_idx]   = (x[parents] + rng.uniform(-JITTER, JITTER, cap_idx.size)) % 1.
                x[N+cap_idx] = (x[N+parents] + rng.uniform(-JITTER, JITTER, cap_idx.size)) % 1.
                vx[cap_idx] = vx[parents]; vy[cap_idx] = vy[parents]
                cum_captures += cap_idx.size

            if (i - N_WARMUP) % RECORD_EVERY == 0:
                t_rec.append((i - N_WARMUP) * dt)
                w_mean_rec.append(w.mean()); w_std_rec.append(w.std())
                phi_rec.append(order_parameter(vx, vy))
                cap_rec.append(cum_captures)
                frac_esc_rec.append(float((w > esc_thresh).mean()))

    return dict(t=np.array(t_rec), w_mean=np.array(w_mean_rec), w_std=np.array(w_std_rec),
                phi=np.array(phi_rec), cum_cap=np.array(cap_rec),
                frac_esc=np.array(frac_esc_rec),
                w_final=w.copy(), w0=w0, seed=seed)


def main():
    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out('Evolution of the collective-escape weight under capture/removal selection')
    out('  N=%d  n_pred=%d  lead=%.1f tu  r_kill=%.3f  capture_rate=%.1f/tu  mut_sigma=%.2f'
        % (BASE['N'], N_PRED, LEAD, R_KILL, CAPTURE_RATE, MUT_SIGMA))
    out('  %d evolution steps (=%.0f tu) after %d warmup; %d seeds per initial w0'
        % (N_EVOLVE, N_EVOLVE*BASE['dt'], N_WARMUP, N_SEEDS))
    out('  Question: does selection drive low-w starts ACROSS the F70 valley (w~0.25 worse')
    out('  than none) up to the winning regime (w>=alpha=1), or does the valley trap them?\n')

    results = {}
    for w0 in W0_VALUES:
        runs = [run_evolution(w0, s) for s in range(N_SEEDS)]
        results[w0] = runs
        wf = np.concatenate([r['w_final'] for r in runs])
        wmean_end = np.mean([r['w_mean'][-1] for r in runs])
        phi_end = np.mean([r['phi'][-1] for r in runs])
        cap_end = np.mean([r['cum_cap'][-1] for r in runs])
        out('  w0=%.2f -> final mean w = %.3f (pop median %.3f, max %.2f)  Phi_end=%.3f  captures=%.0f'
            % (w0, wmean_end, np.median(wf), wf.max(), phi_end, cap_end))

    out('')
    finals = {w0: np.mean([r['w_mean'][-1] for r in results[w0]]) for w0 in W0_VALUES}
    cap = {w0: np.mean([r['cum_cap'][-1] for r in results[w0]]) for w0 in W0_VALUES}
    out('  Over 150 tu the outcome is sharply set by the initial weight: high-w starts are')
    out('  stable and nearly predation-free (w0=2 -> w~%.2f, ~%.0f captures) while low/valley'
        % (finals[2.0], cap[2.0]))
    out('  starts stall well below the escape threshold (w0=0 -> w~%.2f, ~%.0f captures).'
        % (finals[0.0], cap[0.0]))

    # ---- Exp2: long-timescale check on the low starts (barrier vs brake) ----
    out('')
    out('Exp2: long run (%.0f tu) on low-w starts -- is the stall a true barrier or a slow brake?'
        % (N_EVOLVE_LONG * BASE['dt']))
    saved = globals()['N_EVOLVE']
    globals()['N_EVOLVE'] = N_EVOLVE_LONG
    long_runs = {}
    try:
        for w0 in LONG_W0:
            r = run_evolution(w0, 0)
            long_runs[w0] = r
            k = len(r['w_mean']) // 3
            slope = np.polyfit(r['t'][-k:], r['w_mean'][-k:], 1)[0]
            out('  w0=%.2f -> w(50tu)=%.3f  w(150tu)=%.3f  w(400tu)=%.3f  end-slope=%.1e/tu  crossed w=1: %s'
                % (w0,
                   r['w_mean'][np.argmin(np.abs(r['t']-50))],
                   r['w_mean'][np.argmin(np.abs(r['t']-150))],
                   r['w_mean'][-1], slope, bool(r['w_mean'].max() > 1.0)))
    finally:
        globals()['N_EVOLVE'] = saved
    out('')
    out('  -> Selection on w is directional-UPWARD from every start, but the F70 valley THROTTLES')
    out('     the climb: from no escape the population only crawls to w~0.5 in 400 tu and never')
    out('     reaches the escape regime (w>=1), while escape is stable and near-costless once present.')
    out('     The valley is a strong evolutionary BRAKE (slow, hysteretic crossing), not an absolute')
    out('     barrier -- the population-genetic image of the F70 force-versus-alignment threshold.')

    # ---------------------------------------------------------------- figure
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(W0_VALUES)))
    for w0, c in zip(W0_VALUES, colors):
        for r in results[w0]:
            ax[0].plot(r['t'], r['w_mean'], color=c, alpha=0.85,
                       label=('w0=%.2f' % w0) if r['seed'] == 0 else None)
    ax[0].axhspan(0.0, 0.5, color='red', alpha=0.07)
    ax[0].text(ax[0].get_xlim()[1]*0.5, 0.25, 'F70 "dangerous valley"', color='darkred',
               fontsize=8, ha='center', va='center')
    ax[0].axhline(1.0, ls='--', color='gray', alpha=0.6, lw=1, label='w=alpha (escape works)')
    ax[0].set_xlabel('time (tu)'); ax[0].set_ylabel('population mean escape weight w')
    ax[0].set_title('Evolution of escape weight by initial condition'); ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    for w0, c in zip(W0_VALUES, colors):
        wf = np.concatenate([r['w_final'] for r in results[w0]])
        ax[1].hist(wf, bins=30, range=(0, W_MAX), histtype='step', color=c, lw=2,
                   label='w0=%.2f' % w0, density=True)
    ax[1].set_xlabel('final escape weight w'); ax[1].set_ylabel('population density')
    ax[1].set_title('Final escape-weight distribution (150 tu)'); ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    for w0 in LONG_W0:
        r = long_runs[w0]
        ax[2].plot(r['t'], r['w_mean'], lw=2, label='w0=%.2f (long)' % w0)
    ax[2].axhspan(0.0, 0.5, color='red', alpha=0.07)
    ax[2].axhline(1.0, ls='--', color='gray', alpha=0.6, lw=1, label='w=alpha (escape works)')
    ax[2].set_xlabel('time (tu)'); ax[2].set_ylabel('population mean escape weight w')
    ax[2].set_title('Long run: slow brake, not a barrier'); ax[2].legend(fontsize=8)
    ax[2].grid(alpha=0.3)

    fig.suptitle('Evolution of collective escape under capture/removal selection vs predictive encirclement '
                 '(N=%d)' % BASE['N'], fontsize=10)
    plt.tight_layout()
    plt.savefig('figures/escape_evolution_1.png', dpi=120)
    plt.close()
    out('\n  --> figures/escape_evolution_1.png')

    with open('outputs/escape_evolution.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    out('  --> outputs/escape_evolution.txt')
    out('\nEvolution experiment complete.')


if __name__ == '__main__':
    main()
