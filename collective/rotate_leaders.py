# rotate_leaders.py -- is leadership a SIGNAL or an IDENTITY?
#
# F72 showed a fixed informed minority steers the flock. F62 (contagion thread) showed that
# slow-recoverer vaccination FAILS once the per-agent gamma label DRIFTS -- because there the
# targetable thing was a durable per-agent identity. Leadership is the opposite kind of signal:
# the flock follows a shared DIRECTION, not particular individuals. Prediction: rotating WHICH
# agents are informed (while keeping the goal direction fixed) should NOT hurt steering, because
# the decision depends only on the total injected directed force (the F74 "pull" = count*strength),
# not on who carries it.
#
# Mechanism: keep a fixed informed FRACTION rho and a fixed goal g_hat=+x, but every tau time units
# re-draw which rho*N agents are the informed ones at random. tau -> inf is the F72 fixed-leader
# case; tau -> 0 smears the bias across all agents (each informed a fraction rho of the time, so the
# time-averaged force on every agent is rho*w*g_hat). Since the TOTAL injected force rho*N*w*g_hat is
# identical in both limits, accuracy should be ~tau-independent if leadership is force-not-identity.
#
# Sweep tau at two informed fractions. Metric: steady-state directional accuracy and Phi.

import os
import numpy as np
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from flocking import buffer as buf_fn, order_parameter

os.makedirs('figures', exist_ok=True)
N_SEEDS = 5

BASE = dict(N=350, r0=0.005, eps=0.1, rf=0.1,
            alpha=1.0, v0=1.0, mu=10.0, ramp=0.5, dt=0.01)
N_WARMUP = 1000
N_ITER   = 4000
W_LEAD   = 1.0
G_HAT    = np.array([1.0, 0.0])


def run(rho, tau_tu, seed):
    """tau_tu = rotation period in tu (np.inf = fixed informed set, F72)."""
    rng = np.random.RandomState(seed)
    p = BASE.copy()
    N = p['N']; dt = p['dt']
    r0, eps, rf = p['r0'], p['eps'], p['rf']
    v0, mu = p['v0'], p['mu']

    n_inf = int(round(rho * N))
    rot_steps = np.inf if not np.isfinite(tau_tu) else max(1, int(round(tau_tu / dt)))

    def draw_informed():
        m = np.zeros(N, dtype=bool)
        if n_inf > 0:
            m[rng.choice(N, size=n_inf, replace=False)] = True
        return m
    informed = draw_informed()

    x  = np.zeros(2 * N)
    x[:N] = rng.uniform(0., 1., N); x[N:] = rng.uniform(0., 1., N)
    vx = rng.uniform(-1., 1., N) * v0
    vy = rng.uniform(-1., 1., N) * v0

    rb = max(r0, rf)
    acc_rec = []; phi_rec = []
    for i in range(N_ITER):
        if np.isfinite(rot_steps) and i > 0 and (i % rot_steps == 0):
            informed = draw_informed()

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

        if n_inf > 0:
            fx[informed] += W_LEAD * G_HAT[0]; fy[informed] += W_LEAD * G_HAT[1]

        vx += fx * dt; vy += fy * dt
        x[:N] = (x[:N] + vx * dt) % 1.; x[N:] = (x[N:] + vy * dt) % 1.

        if i >= N_WARMUP:
            mvx = vx.mean(); mvy = vy.mean()
            mnorm = np.hypot(mvx, mvy)
            acc_rec.append((mvx * G_HAT[0] + mvy * G_HAT[1]) / mnorm if mnorm > 1e-9 else 0.0)
            phi_rec.append(order_parameter(vx, vy))
    return float(np.mean(acc_rec)), float(np.mean(phi_rec))


print('Is leadership a SIGNAL or an IDENTITY? Rotating the informed set (goal fixed at +x)')
print('  N=%d  pure-flock  %d seeds  w_lead=%.1f\n' % (BASE['N'], N_SEEDS, W_LEAD))
print('  Prediction: accuracy ~ independent of rotation period tau (total pull rho*N*w fixed),')
print('  the OPPOSITE of F62 where drifting the per-agent label destroyed the advantage.\n')

TAU_VALS = [0.1, 0.5, 2.0, 10.0, np.inf]   # inf = fixed informed set (F72)
RHO_VALS = [0.05, 0.10]
results = {}
for rho in RHO_VALS:
    print('== rho=%.2f (%d informed at any instant) ==' % (rho, int(round(rho * BASE['N']))))
    for tau in TAU_VALS:
        accs = []; phis = []
        for s in range(N_SEEDS):
            a, ph = run(rho, tau, s)
            accs.append(a); phis.append(ph)
        results[(rho, tau)] = (np.mean(accs), np.std(accs), np.mean(phis))
        label = 'fixed (F72)' if not np.isfinite(tau) else ('tau=%4.1f tu' % tau)
        print('   %-12s  accuracy=%+.3f +/- %.3f  Phi=%.3f'
              % (label, np.mean(accs), np.std(accs), np.mean(phis)))
    print()

# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))
fig.suptitle('Leadership is a signal, not an identity: rotating the informed set (N=%d, %d seeds)'
             % (BASE['N'], N_SEEDS), fontsize=11)
xt = [0.1, 0.5, 2.0, 10.0, 40.0]    # plot inf as 40 (off to the right)
xtl = ['0.1', '0.5', '2', '10', 'fixed']
for rho, col in zip(RHO_VALS, ['steelblue', 'crimson']):
    acc = [results[(rho, t)][0] for t in TAU_VALS]
    err = [results[(rho, t)][1] for t in TAU_VALS]
    ax.errorbar(xt, acc, yerr=err, marker='o', capsize=4, lw=2, color=col,
                label='rho=%.2f' % rho)
ax.set_xscale('log')
ax.set_xticks(xt); ax.set_xticklabels(xtl)
ax.set_xlabel('rotation period tau (tu); "fixed" = never rotate (F72)')
ax.set_ylabel('directional accuracy toward goal')
ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig('figures/rotate_leaders_1.png', dpi=120)
plt.close()
print('  --> figures/rotate_leaders_1.png')
print('\nRotating-leaders analysis complete.')
