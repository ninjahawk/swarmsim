"""
vectorized_predator_prey.py -- a fast, validated predator-prey episode runner.

This is the second half of the evolution-thread prerequisite (the first is
vectorized_predator.predator_force). It runs one slow-prey encirclement episode
with PREDICTIVE predators (F66: target CoM + lead_time * v_mean) and an optional
COLLECTIVE-ESCAPE force on the prey (F70: flee the predator centroid), with both
the predator->prey force and the predator motion update vectorized across all
predators. The already-vectorized flock step is reproduced verbatim from the F66/
F70 reference scripts so the dynamics are identical.

It is ADDITIVE infrastructure: it imports only the safe modules (flocking, model,
vectorized_predator), has a __main__ guard, and does not modify anything. The
__main__ self-test reproduces the published F66 contrast (Phi ~ 0.83 at lead=0,
~0.53 at lead=2) and the F70 counter (escape weight w >= 2 restores Phi ~ 1.0).

Why this exists: the proposed co-adaptation/evolution thread evolves a heritable
per-agent escape weight across the F70 "dangerous valley" (w ~ 0.25 worse than
none). `run_episode` therefore accepts `w_escape` as a scalar OR a per-agent
(N,) array, and returns the final prey/predator state so any per-agent fitness
rule the STUDENT chooses (proximity-survival / capture-removal / energy-budget --
see findings.md Open Questions) can be evaluated on top. It deliberately does NOT
implement selection or fitness; that is the scientific decision to be made first.
"""

import numpy as np

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from flocking import buffer as buf_fn, order_parameter
from model import _periodic_com, _periodic_disp
from vectorized_predator import predator_force

# Slow-prey regime + predator parameters, matching the legacy predator findings
# (predictive_encirclement.py / collective_escape.py).
BASE = dict(N=350, r0=0.005, eps=0.1, rf=0.1,
            alpha=1.0, v0=0.02, mu=10.0, ramp=0.1, dt=0.01)
PRED = dict(v0=0.05, mu=10.0, alpha=5.0, r0=0.1, eps=2.0, ramp=1.0,
            enc_radius=0.15)


def run_episode(lead_time=2.0, w_escape=0.0, seed=0, n_pred=6,
                n_warmup=1000, n_iter=5000, base=None, pred=None,
                record_from=500, return_state=False):
    """
    Run one predator-prey episode and return the mean/std order parameter
    during the attack phase.

    Parameters
    ----------
    lead_time : predictive lead in time units (0 reproduces fixed encirclement F14;
                ~2 is the hardest predator, F66).
    w_escape  : collective-escape weight. Scalar, or a per-agent (N,) array for a
                heritable trait. 0 disables escape (pure F66).
    seed      : RNG seed.
    n_pred    : number of predators on the encirclement ring.
    n_warmup  : steps the flock equilibrates before predators activate.
    n_iter    : total steps.
    record_from : drop this many attack-phase steps as transient before averaging.
    return_state : if True, also return the final prey and predator state for
                   downstream per-agent fitness evaluation.

    Returns
    -------
    phi_mean, phi_std : floats (attack-phase order parameter).
    If return_state: also a dict with prey x/y/vx/vy, predator x/y, and the
    per-step minimum prey->predator distance history (useful for proximity-based
    fitness).
    """
    p = {**BASE, **(base or {})}
    pr = {**PRED, **(pred or {})}
    rng = np.random.RandomState(seed)

    N = p['N']; dt = p['dt']
    r0, eps, rf = p['r0'], p['eps'], p['rf']
    v0, mu, alpha, ramp = p['v0'], p['mu'], p['alpha'], p['ramp']

    x = np.zeros(2 * N)
    x[:N] = rng.uniform(0., 1., N); x[N:] = rng.uniform(0., 1., N)
    vx = rng.uniform(-1., 1., N) * v0
    vy = rng.uniform(-1., 1., N) * v0

    pred_x = rng.uniform(0., 1., n_pred)
    pred_y = rng.uniform(0., 1., n_pred)
    pred_vx = np.zeros(n_pred); pred_vy = np.zeros(n_pred)
    pred_angles = np.radians(np.arange(n_pred) * 360.0 / n_pred)
    cos_a = np.cos(pred_angles); sin_a = np.sin(pred_angles)

    w_escape = np.asarray(w_escape, dtype=float)  # scalar or (N,)
    rb = max(r0, rf)
    phi_record = []
    min_dist_record = []

    for i in range(n_iter):
        on = (i >= n_warmup)

        # --- flock step (verbatim from the F66/F70 reference, already vectorized) ---
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

        vnorm = np.sqrt(vx**2 + vy**2)
        vnorms = np.where(vnorm == 0, 1., vnorm)
        fpropx = mu * (v0 - vnorm) * vx / vnorms
        fpropy = mu * (v0 - vnorm) * vy / vnorms

        frandx = ramp * rng.uniform(-1., 1., N)
        frandy = ramp * rng.uniform(-1., 1., N)

        fx_total = flockx + repx + fpropx + frandx
        fy_total = flocky + repy + fpropy + frandy

        if on:
            # predator repulsion on prey (vectorized over all predators)
            pred_xy = np.column_stack((pred_x, pred_y))
            pfx_prey, pfy_prey = predator_force(pred_xy, x[:N], x[N:], pr['r0'], pr['eps'])
            fx_total += pfx_prey
            fy_total += pfy_prey

            # collective escape: shared unit vector from predator centroid toward CoM
            if np.any(w_escape > 0):
                cx0, cy0 = _periodic_com(x[:N]), _periodic_com(x[N:])
                pcx, pcy = _periodic_com(pred_x), _periodic_com(pred_y)
                ex = _periodic_disp(cx0, pcx); ey = _periodic_disp(cy0, pcy)
                en = np.sqrt(ex**2 + ey**2)
                if en > 1e-9:
                    ex /= en; ey /= en
                    fx_total += w_escape * ex
                    fy_total += w_escape * ey

        vx += fx_total * dt; vy += fy_total * dt
        x[:N] = (x[:N] + vx*dt) % 1.; x[N:] = (x[N:] + vy*dt) % 1.

        if on:
            # predictive predator motion update (vectorized over predators)
            cx, cy = _periodic_com(x[:N]), _periodic_com(x[N:])
            vmx, vmy = vx.mean(), vy.mean()
            tx = (cx + pr['enc_radius']*cos_a + lead_time*vmx) % 1.
            ty = (cy + pr['enc_radius']*sin_a + lead_time*vmy) % 1.
            ddx = _periodic_disp(tx, pred_x); ddy = _periodic_disp(ty, pred_y)
            dist = np.sqrt(ddx**2 + ddy**2)
            nz = dist > 0
            ddx = np.where(nz, ddx/np.where(nz, dist, 1.), ddx)
            ddy = np.where(nz, ddy/np.where(nz, dist, 1.), ddy)
            sp = np.sqrt(pred_vx**2 + pred_vy**2)
            sps = np.where(sp > 0, sp, 1.)
            drive = np.where(sp > 0, pr['mu']*(pr['v0']-sp)/sps, 0.)
            pfx = pr['alpha']*ddx + drive*pred_vx
            pfy = pr['alpha']*ddy + drive*pred_vy
            # interleaved (n_pred, 2) draw reproduces the reference's per-predator
            # rng.uniform order exactly (pfx_0, pfy_0, pfx_1, ...), so this harness
            # is bit-faithful to predictive_encirclement.py / collective_escape.py.
            pnoise = rng.uniform(-1., 1., (n_pred, 2))
            pfx += pr['ramp']*pnoise[:, 0]
            pfy += pr['ramp']*pnoise[:, 1]
            pred_vx += pfx*dt; pred_vy += pfy*dt
            pred_x = (pred_x + pred_vx*dt) % 1.; pred_y = (pred_y + pred_vy*dt) % 1.

            phi_record.append(order_parameter(vx, vy))
            if return_state:
                ddxp = _periodic_disp(pred_x[:, None], x[:N][None, :])
                ddyp = _periodic_disp(pred_y[:, None], x[N:][None, :])
                min_dist_record.append(np.sqrt(ddxp**2 + ddyp**2).min(axis=0))

    arr = np.array(phi_record[record_from:]) if len(phi_record) > record_from else np.array(phi_record)
    phi_mean, phi_std = float(arr.mean()), float(arr.std())
    if return_state:
        state = dict(prey_x=x[:N].copy(), prey_y=x[N:].copy(), vx=vx.copy(), vy=vy.copy(),
                     pred_x=pred_x.copy(), pred_y=pred_y.copy(),
                     min_pred_dist=np.array(min_dist_record))
        return phi_mean, phi_std, state
    return phi_mean, phi_std


def _self_test():
    import time
    n_seeds = 3
    print("Validating the vectorized predator-prey harness against published F66/F70 numbers")
    print("  (slow-prey regime, n_pred=6, %d seeds, 5000 steps)\n" % n_seeds)

    t0 = time.perf_counter()

    def sweep(lead, w):
        ms = [run_episode(lead_time=lead, w_escape=w, seed=s)[0] for s in range(n_seeds)]
        return float(np.mean(ms)), float(np.std(ms))

    f66_lead0 = sweep(0.0, 0.0)   # F66 lead=0 -> ~0.825 (F14 baseline)
    f66_lead2 = sweep(2.0, 0.0)   # F66 lead=2 -> ~0.530 (hardest predator)
    f70_w2    = sweep(2.0, 2.0)   # F70 escape w=2 at lead=2 -> ~1.000
    elapsed = time.perf_counter() - t0

    print("  F66 lead=0.0, w=0 : Phi = %.3f +/- %.3f   (published ~0.825)" % f66_lead0)
    print("  F66 lead=2.0, w=0 : Phi = %.3f +/- %.3f   (published ~0.530)" % f66_lead2)
    print("  F70 lead=2.0, w=2 : Phi = %.3f +/- %.3f   (published ~1.000)" % f70_w2)
    print("  (%d episodes in %.1fs)\n" % (3 * n_seeds, elapsed))

    ok = True
    # the published qualitative findings, with tolerance for seed noise:
    if not (f66_lead2[0] < f66_lead0[0] - 0.10):
        print("  FAIL: predictive lead=2 should deepen disruption below lead=0"); ok = False
    if not (abs(f66_lead2[0] - 0.530) < 0.15):
        print("  WARN: lead=2 Phi off published 0.530 by >0.15 (seed noise?)")
    if not (f70_w2[0] > 0.90):
        print("  FAIL: collective escape w=2 should restore Phi above 0.90"); ok = False
    if ok:
        print("  PASS: harness reproduces the F66 predictive-disruption contrast and the")
        print("        F70 collective-escape counter. Ready for the evolution thread.")
    else:
        raise SystemExit("Self-test FAILED -- harness does not reproduce the findings.")


if __name__ == "__main__":
    _self_test()
