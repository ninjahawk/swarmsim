# flocking3d_transect.py -- is the 3D encirclement failure (F43-F49) specific to
# PERIMETER-SEALING strategies, or are 3D flocks robust to ANY point predator?
#
# F43-F49: encirclement does not disrupt a 3D flock at all (Phi~1.0) by any
# geometric variant (radius/count/adaptive/sphere-vs-planar). The mechanistic
# claim was that a handful of point predators cannot seal a closed 2D SURFACE
# around a 3D volume the way they can seal a 1D perimeter around a 2D area.
#
# That claim, if correct, only rules out SURROUNDING strategies. A strategy that
# does not rely on sealing -- a predator that TRANSECTS the flock, darting through
# the dense core at high speed and shearing alignment in its wake -- is the
# natural test. A transect predator uses the same CoM target as a naive predator
# but moves fast enough to overshoot and punch through, then is pulled back,
# oscillating through the core repeatedly. With several transect predators along
# different lines the flock is sheared from many directions without being sealed.
#
# Prediction: if F43's "cannot seal a surface" mechanism is the whole story, then
# a non-sealing strategy might disrupt the 3D flock where encirclement cannot.
# If even transecting fails, 3D flocks are robust to point predators generally
# (a stronger statement than F43), and the alignment force heals the wake as fast
# as it is cut (the 3D analog of F22 reversibility).
#
# Experiments:
#   A. Strategy comparison {naive, encircle, transect} at n_pred = 3, 6, 10
#   B. Transect predator-speed sweep at n_pred = 10 (is faster more disruptive?)

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('figures', exist_ok=True)

N          = 350
N_SEEDS    = 3
N_ITER     = 4000
N_WARMUP   = 2500

# 3D prey physics (from F41/F43)
R0_3D  = 0.02
RF_3D  = 0.20
ALPHA  = 1.0
V0_PRY = 0.02
MU     = 10.0
RAMP   = 0.1
EPS    = 0.1
EXP_N  = 1.5
RB_3D  = 2.0 * R0_3D

# Predator parameters (F43 PRED_DEFAULT)
V0_SLOW = 0.05      # naive / encircle speed
V0_FAST = 0.30      # transect speed (punches through)
ALPHA_P = 5.0
MU_P    = 10.0
R0_P    = 0.10
EPS_P   = 2.0
RAMP_P  = 1.0
RENC_DEFAULT = 0.15

NPRED_VALS = [3, 6, 10]
SPEED_SWEEP = [0.05, 0.10, 0.20, 0.40, 0.80]


def com3d(pos):
    cx = np.arctan2(np.sin(2*np.pi*pos[0]).mean(), np.cos(2*np.pi*pos[0]).mean()) / (2*np.pi) % 1.0
    cy = np.arctan2(np.sin(2*np.pi*pos[1]).mean(), np.cos(2*np.pi*pos[1]).mean()) / (2*np.pi) % 1.0
    cz = np.arctan2(np.sin(2*np.pi*pos[2]).mean(), np.cos(2*np.pi*pos[2]).mean()) / (2*np.pi) % 1.0
    return np.array([cx, cy, cz])


def rg3d(pos, c):
    d = pos - c[:, np.newaxis]; d -= np.round(d)
    return float(np.sqrt((d**2).sum(axis=0).mean()))


def order_param3d(vel):
    spd = np.maximum(np.sqrt((vel**2).sum(axis=0)), 1e-10)
    vhat = vel / spd[np.newaxis, :]
    return float(np.sqrt((vhat.mean(axis=1)**2).sum()))


def fibonacci_sphere(n):
    if n == 1:
        return np.array([[0.0, 0.0, 1.0]])
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    idx = np.arange(n)
    theta = np.arccos(1.0 - 2.0*(idx + 0.5)/n)
    phi   = 2.0 * np.pi * idx / golden
    return np.column_stack([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])


class Pred3D:
    """3D predator. strategy in {naive, encircle, transect}.
       naive/transect aim at CoM (transect just moves faster); encircle offsets."""
    def __init__(self, direction, strategy, v0_pred, enc_radius=RENC_DEFAULT, seed=None):
        rng = np.random.default_rng(seed)
        self.pos = rng.uniform(0.0, 1.0, 3)
        raw = rng.standard_normal(3)
        self.vel = v0_pred * raw / np.linalg.norm(raw)
        self.direction = np.asarray(direction, dtype=float)
        self.strategy = strategy
        self.v0 = v0_pred
        self.enc_radius = enc_radius

    def target(self, c):
        if self.strategy == 'encircle':
            return (c + self.enc_radius * self.direction) % 1.0
        return c.copy()   # naive and transect both chase the CoM

    def step(self, c, dt):
        t = self.target(c)
        disp = t - self.pos; disp -= np.round(disp)
        dist = np.linalg.norm(disp)
        speed = np.linalg.norm(self.vel)
        drive = (ALPHA_P * disp / (dist + 1e-12)
                 + MU_P * (self.v0 - speed) * self.vel / (speed + 1e-12))
        drive += RAMP_P * np.random.uniform(-1., 1., 3)
        self.vel += drive * dt
        self.pos = (self.pos + self.vel * dt) % 1.0

    def force_on_prey(self, pos):
        d = pos - self.pos[:, np.newaxis]; d -= np.round(d)
        dist = np.sqrt((d**2).sum(axis=0))
        in_range = (dist > 0) & (dist <= R0_P)
        base = np.maximum(1.0 - dist / R0_P, 0.0)
        strength = np.where(in_range, EPS_P * base**1.5 / (dist + 1e-12), 0.0)
        return strength[np.newaxis, :] * d   # +d: push prey away (corrected sign)


def run_3d(n_pred, strategy, v0_pred, seed):
    np.random.seed(seed)
    pos = np.random.uniform(0., 1., (3, N))
    raw = np.random.randn(3, N); raw /= np.sqrt((raw**2).sum(axis=0))
    vel = V0_PRY * raw

    directions = fibonacci_sphere(n_pred)
    preds = [Pred3D(directions[k], strategy, v0_pred, seed=seed*100+k) for k in range(n_pred)]

    phi_vals = []; rg_vals = []
    for step in range(N_ITER):
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
        fx = (-strength * dx).sum(axis=1); fy = (-strength * dy).sum(axis=1); fz = (-strength * dz).sum(axis=1)

        flock_mask = (d2 <= RF_3D**2) & not_self
        svx = (vel[0]*flock_mask).sum(axis=1); svy = (vel[1]*flock_mask).sum(axis=1); svz = (vel[2]*flock_mask).sum(axis=1)
        vbar = np.sqrt(svx**2 + svy**2 + svz**2)
        has = (flock_mask.sum(axis=1) > 0); safe = np.where(has, vbar, 1.0)
        fx += np.where(has, ALPHA*svx/safe, 0.0); fy += np.where(has, ALPHA*svy/safe, 0.0); fz += np.where(has, ALPHA*svz/safe, 0.0)

        spd = np.maximum(np.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2), 1e-10)
        prop = MU * (V0_PRY - spd) / spd
        fx += prop*vel[0]; fy += prop*vel[1]; fz += prop*vel[2]

        fx += RAMP*np.random.uniform(-1.,1.,N); fy += RAMP*np.random.uniform(-1.,1.,N); fz += RAMP*np.random.uniform(-1.,1.,N)

        c = com3d(pos)
        for pred in preds:
            fp = pred.force_on_prey(pos)
            fx += fp[0]; fy += fp[1]; fz += fp[2]

        vel[0] += fx*0.01; vel[1] += fy*0.01; vel[2] += fz*0.01
        pos = (pos + vel*0.01) % 1.0
        for pred in preds:
            pred.step(c, 0.01)

        if step >= N_WARMUP:
            phi_vals.append(order_param3d(vel)); rg_vals.append(rg3d(pos, c))
    return float(np.mean(phi_vals)), float(np.mean(rg_vals))


# ---------------------------------------------------------------------------
print('3D transect predator -- is encirclement failure (F43) specific to sealing?')
print('  N=%d  N_SEEDS=%d  N_ITER=%d  v0_prey=%.2f' % (N, N_SEEDS, N_ITER, V0_PRY))

print('\nExp A: strategy comparison {naive, encircle, transect}')
strat_speed = {'naive': V0_SLOW, 'encircle': V0_SLOW, 'transect': V0_FAST}
resA = {}
for strat in ('naive', 'encircle', 'transect'):
    for npd in NPRED_VALS:
        phis = []; rgs = []
        for s in range(N_SEEDS):
            phi, rg = run_3d(npd, strat, strat_speed[strat], s)
            phis.append(phi); rgs.append(rg)
        resA[(strat, npd)] = (np.mean(phis), np.std(phis), np.mean(rgs))
        print('  %-9s n_pred=%2d  Phi=%.3f+/-%.3f  Rg=%.3f' %
              (strat, npd, np.mean(phis), np.std(phis), np.mean(rgs)))

print('\nExp B: transect predator-speed sweep at n_pred=10')
resB = {}
for v0p in SPEED_SWEEP:
    phis = []; rgs = []
    for s in range(N_SEEDS):
        phi, rg = run_3d(10, 'transect', v0p, s)
        phis.append(phi); rgs.append(rg)
    resB[v0p] = (np.mean(phis), np.std(phis), np.mean(rgs))
    print('  v0_pred=%.2f  Phi=%.3f+/-%.3f  Rg=%.3f' %
          (v0p, np.mean(phis), np.std(phis), np.mean(rgs)))

# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('3D transect vs encirclement: is the F43 failure specific to perimeter sealing? '
             '(N=%d, %d seeds)' % (N, N_SEEDS), fontsize=11)

ax = axes[0]
colors = {'naive':'gray', 'encircle':'steelblue', 'transect':'crimson'}
for strat in ('naive', 'encircle', 'transect'):
    phi = [resA[(strat, n)][0] for n in NPRED_VALS]
    err = [resA[(strat, n)][1] for n in NPRED_VALS]
    ax.errorbar(NPRED_VALS, phi, yerr=err, marker='o', capsize=4,
                color=colors[strat], lw=2,
                label='%s%s' % (strat, ' (fast)' if strat == 'transect' else ''))
ax.axhline(1.0, ls='--', color='gray', lw=0.8)
ax.axhline(0.67, ls=':', color='black', lw=1, label='2D encirclement floor ~0.67')
ax.set_xlabel('number of predators'); ax.set_ylabel('order parameter Phi')
ax.set_title('Strategy comparison'); ax.set_ylim(0, 1.05)
ax.grid(alpha=0.3); ax.legend(fontsize=8)

ax = axes[1]
v0arr = SPEED_SWEEP
phi = [resB[v][0] for v in v0arr]; err = [resB[v][1] for v in v0arr]
ax.errorbar(v0arr, phi, yerr=err, marker='s', capsize=4, color='crimson', lw=2)
ax.axhline(1.0, ls='--', color='gray', lw=0.8)
ax.set_xlabel('transect predator speed v0_pred (prey v0=%.2f)' % V0_PRY)
ax.set_ylabel('order parameter Phi')
ax.set_title('Transect speed sweep (n_pred=10)'); ax.set_ylim(0, 1.05)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/flocking3d_transect_1.png', dpi=120)
plt.close()
print('\n  --> figures/flocking3d_transect_1.png')
print('\n3D transect predator analysis complete.')
