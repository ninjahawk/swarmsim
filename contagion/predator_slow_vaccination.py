# predator_slow_vaccination.py -- does slow-recoverer vaccination reverse the
# F26 (Finding 34) damage asymmetry?
#
# F34 found the sharpest asymmetry in this study: after an encirclement-driven
# SIS outbreak, removing the predators lets the KINEMATIC damage reverse fast
# (~10 tu, sub-flocks reunite, F22) but the EPIDEMIC persists 100+ tu.  Contagion
# was therefore "the worst combined stressor" -- its damage outlasts the event.
# That experiment used HOMOGENEOUS gamma.
#
# F54-F63 established that endemic persistence is set by the SLOW recoverers (the
# reservoir), and that vaccinating them (F56) is the one robust targeting policy.
# Pointed question: if the post-removal epidemic persists because slow recoverers
# act as a reservoir, then vaccinating the slow class before the attack should let
# the epidemic DIE after predator removal -- converting irreversible contagion
# damage into reversible damage, and overturning F34's "contagion always wins".
#
# Three-phase protocol (slow-prey predator regime, v0=0.02, as in outbreak_removal):
#   Phase 1 (0..N_WARMUP):           pure flock, no contagion
#   Phase 2 (N_WARMUP..N_REMOVE):    6-predator encirclement + SIS (induce outbreak)
#   Phase 3 (N_REMOVE..N_TOTAL):     predators removed, contagion left alone
#
# Heterogeneous recovery: bimodal gamma {0.5, 3.5}, mean 2.0 (== F34 mean), so the
# slow class is a reservoir (beta/gamma_slow = 3.0) while the mean stays at F34's.
# Vaccination (immune agents never panic) applied at t=0; strategies none/random/slow.

import os
import numpy as np
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from flocking import buffer as buf_fn, order_parameter
from model import _periodic_com, _periodic_disp

os.makedirs('figures', exist_ok=True)
N_SEEDS = 4

BASE = dict(
    N=350, r0=0.005, eps=0.1, rf=0.1,
    alpha=1.0, v0=0.02, mu=10.0, ramp=0.1, dt=0.01,
)
PANIC_ALPHA = 0.1
PANIC_RAMP  = 10.0
R_CONT      = 0.05
BETA        = 1.5
GAMMA_MEAN  = 2.0
SPREAD      = 1.5          # bimodal {0.5, 3.5}; slow class beta/gamma = 3.0

PRED = dict(v0=0.05, mu=10.0, alpha=5.0, r0=0.1, eps=2.0, ramp=1.0,
            enc_radius=0.15)

N_WARMUP = 1000
N_REMOVE = 5000   # 4000 steps of attack
N_TOTAL  = 10000  # 5000 steps of recovery


def make_gamma(N, rng):
    gamma = np.full(N, GAMMA_MEAN, dtype=float)
    is_slow = np.zeros(N, dtype=bool)
    idx = rng.choice(N, size=N // 2, replace=False)
    is_slow[idx] = True
    gamma[is_slow]  = GAMMA_MEAN - SPREAD
    gamma[~is_slow] = GAMMA_MEAN + SPREAD
    return gamma, is_slow


def pick_immune(strategy, gamma_arr, p_immune, rng):
    N = gamma_arr.size
    n_imm = int(round(p_immune * N))
    immune = np.zeros(N, dtype=bool)
    if n_imm == 0 or strategy == 'none':
        return immune
    if strategy == 'random':
        idx = rng.choice(N, size=n_imm, replace=False)
    elif strategy == 'slow':
        jit = rng.uniform(0., 1e-9, N)
        idx = np.argsort(gamma_arr + jit)[:n_imm]
    else:
        raise ValueError(strategy)
    immune[idx] = True
    return immune


def run(strategy, p_immune, seed):
    rng = np.random.RandomState(seed)
    p = BASE.copy()
    N = p['N']; dt = p['dt']
    r0, eps, rf = p['r0'], p['eps'], p['rf']
    v0, mu = p['v0'], p['mu']

    gamma_arr, is_slow = make_gamma(N, rng)
    p_recover_arr = 1. - np.exp(-gamma_arr * dt)
    immune = pick_immune(strategy, gamma_arr, p_immune, rng)

    x  = np.zeros(2*N)
    x[:N] = rng.uniform(0., 1., N)
    x[N:] = rng.uniform(0., 1., N)
    vx = rng.uniform(-1., 1., N) * v0
    vy = rng.uniform(-1., 1., N) * v0

    is_panicked = np.zeros(N, dtype=bool)

    n_pred = 6
    pred_x  = rng.uniform(0., 1., n_pred)
    pred_y  = rng.uniform(0., 1., n_pred)
    pred_vx = np.zeros(n_pred); pred_vy = np.zeros(n_pred)
    pred_angles = np.radians(np.arange(n_pred) * 60.0)

    rb = max(r0, rf, R_CONT)
    f_t=[]; phi_t=[]; t_t=[]; slowf_t=[]
    record_every = 50

    for i in range(N_TOTAL):
        encirclement_on = (i >= N_WARMUP) and (i < N_REMOVE)
        contagion_on    = (i >= N_WARMUP)

        if i == N_WARMUP:
            n0 = max(1, round(0.05 * N))
            nonimm = np.where(~immune)[0]
            idx0 = rng.choice(nonimm, size=min(n0, nonimm.size), replace=False)
            is_panicked[idx0] = True

        nb, xb, yb, vxb, vyb = buf_fn(rb, x, vx, vy, N)
        dx = xb[:nb][np.newaxis, :] - x[:N][:, np.newaxis]
        dy = yb[:nb][np.newaxis, :] - x[N:][:, np.newaxis]
        d2 = dx**2 + dy**2
        not_self = np.ones((N, nb), dtype=bool)
        idx = np.arange(min(N, nb)); not_self[idx, idx] = False

        alpha_arr = np.where(is_panicked, PANIC_ALPHA, p['alpha'])
        ramp_arr  = np.where(is_panicked, PANIC_RAMP,  p['ramp'])

        flock_mask = (d2 <= rf**2) & not_self
        flx = np.where(flock_mask, vxb[:nb], 0.).sum(axis=1)
        fly = np.where(flock_mask, vyb[:nb], 0.).sum(axis=1)
        nfl = flock_mask.sum(axis=1)
        nrm = np.sqrt(flx**2 + fly**2); nrm[nfl == 0] = 1.
        flockx = alpha_arr * flx / nrm; flocky = alpha_arr * fly / nrm

        rep_mask = (d2 <= (2*r0)**2) & not_self & (d2 > 0)
        d_safe   = np.where(rep_mask, np.sqrt(d2), 1.)
        base_r   = np.where(rep_mask, 1. - d_safe/(2.*r0), 0.)
        strength = np.where(rep_mask, eps * base_r**1.5 / d_safe, 0.)
        repx = (-strength * dx).sum(axis=1); repy = (-strength * dy).sum(axis=1)

        vnorm  = np.sqrt(vx**2 + vy**2)
        vnorms = np.where(vnorm == 0, 1., vnorm)
        fpropx = mu * (v0 - vnorm) * vx / vnorms
        fpropy = mu * (v0 - vnorm) * vy / vnorms

        frandx = ramp_arr * rng.uniform(-1., 1., N)
        frandy = ramp_arr * rng.uniform(-1., 1., N)

        fx_total = flockx + repx + fpropx + frandx
        fy_total = flocky + repy + fpropy + frandy

        if encirclement_on:
            for k in range(n_pred):
                ddx = _periodic_disp(pred_x[k], x[:N])
                ddy = _periodic_disp(pred_y[k], x[N:])
                d = np.sqrt(ddx**2 + ddy**2)
                mask_p = (d > 0) & (d <= PRED['r0'])
                if mask_p.any():
                    strength_p = PRED['eps'] * (1. - d[mask_p]/PRED['r0'])**1.5 / d[mask_p]
                    fx_total[mask_p] -= strength_p * ddx[mask_p]
                    fy_total[mask_p] -= strength_p * ddy[mask_p]

        vx += fx_total * dt; vy += fy_total * dt
        x[:N] = (x[:N] + vx*dt) % 1.; x[N:] = (x[N:] + vy*dt) % 1.

        if encirclement_on:
            cx, cy = _periodic_com(x[:N]), _periodic_com(x[N:])
            for k in range(n_pred):
                tx = (cx + PRED['enc_radius'] * np.cos(pred_angles[k])) % 1.
                ty = (cy + PRED['enc_radius'] * np.sin(pred_angles[k])) % 1.
                ddx = _periodic_disp(tx, pred_x[k]); ddy = _periodic_disp(ty, pred_y[k])
                dist = np.sqrt(ddx**2 + ddy**2)
                if dist > 0:
                    ddx /= dist; ddy /= dist
                sp_p = np.sqrt(pred_vx[k]**2 + pred_vy[k]**2)
                if sp_p > 0:
                    pfx = PRED['alpha']*ddx + PRED['mu']*(PRED['v0']-sp_p)*pred_vx[k]/sp_p
                    pfy = PRED['alpha']*ddy + PRED['mu']*(PRED['v0']-sp_p)*pred_vy[k]/sp_p
                else:
                    pfx = PRED['alpha']*ddx; pfy = PRED['alpha']*ddy
                pfx += PRED['ramp']*rng.uniform(-1., 1.); pfy += PRED['ramp']*rng.uniform(-1., 1.)
                pred_vx[k] += pfx*dt; pred_vy[k] += pfy*dt
                pred_x[k] = (pred_x[k] + pred_vx[k]*dt) % 1.
                pred_y[k] = (pred_y[k] + pred_vy[k]*dt) % 1.

        if contagion_on and is_panicked.any() and (~is_panicked & ~immune).any():
            real_dx = x[:N][np.newaxis, :] - x[:N][:, np.newaxis]
            real_dy = x[N:][np.newaxis, :] - x[N:][:, np.newaxis]
            real_dx -= np.round(real_dx); real_dy -= np.round(real_dy)
            rd2 = real_dx**2 + real_dy**2
            within = (rd2 <= R_CONT**2) & (rd2 > 0)
            k_arr = within @ is_panicked.astype(np.int32)
            calm_idx = np.where(~is_panicked & ~immune)[0]
            if calm_idx.size:
                p_trans = 1. - np.exp(-BETA * k_arr[calm_idx] * dt)
                r = rng.uniform(0., 1., calm_idx.size)
                flipped = calm_idx[r < p_trans]
                if flipped.size:
                    is_panicked[flipped] = True
        if contagion_on and is_panicked.any():
            panic_idx = np.where(is_panicked)[0]
            r = rng.uniform(0., 1., panic_idx.size)
            recovered = panic_idx[r < p_recover_arr[panic_idx]]
            if recovered.size:
                is_panicked[recovered] = False

        if i % record_every == 0:
            f_t.append(is_panicked.mean())
            phi_t.append(order_parameter(vx, vy))
            avail_slow = is_slow & ~immune
            slowf_t.append(is_panicked[avail_slow].mean() if avail_slow.any() else 0.0)
            t_t.append(i * dt)

    return np.array(t_t), np.array(phi_t), np.array(f_t), np.array(slowf_t)


CONFIGS = [('none', 0.0), ('random', 0.20), ('slow', 0.20),
           ('random', 0.30), ('slow', 0.30),
           ('random', 0.50), ('slow', 0.50)]

print('Predator + slow-recoverer vaccination -- does it reverse the F34 asymmetry?')
print('beta=%.2f, bimodal gamma {%.1f,%.1f} mean %.1f, %d seeds' %
      (BETA, GAMMA_MEAN - SPREAD, GAMMA_MEAN + SPREAD, GAMMA_MEAN, N_SEEDS))
print('encirclement steps %d..%d, removed %d..%d\n' % (N_WARMUP, N_REMOVE, N_REMOVE, N_TOTAL))

dt_val = BASE['dt']
t_remove = N_REMOVE * dt_val
runs = {}
for strat, p_imm in CONFIGS:
    key = (strat, p_imm)
    seeds_data = []
    for s in range(N_SEEDS):
        seeds_data.append(run(strat, p_imm, s))
    runs[key] = seeds_data
    t = seeds_data[0][0]
    f_arr = np.array([r[2] for r in seeds_data])
    phi_arr = np.array([r[1] for r in seeds_data])

    def window(arr, t0, t1):
        m = (t >= t0) & (t < t1)
        return arr[:, m].mean(axis=1)
    f_during = window(f_arr, t_remove - 5.0, t_remove)
    f_post   = window(f_arr, t[-1] - 10.0, t[-1] + 1)
    phi_post = window(phi_arr, t[-1] - 10.0, t[-1] + 1)
    print('  %-7s p_imm=%.2f :  f_during=%.3f+/-%.3f  f_post=%.3f+/-%.3f  Phi_post=%.3f' %
          (strat, p_imm, f_during.mean(), f_during.std(),
           f_post.mean(), f_post.std(), phi_post.mean()))

# =============================================================================
# FIGURE: mean f(t) for none / random@0.2 / slow@0.2, attack span shaded
# =============================================================================
t = runs[('none', 0.0)][0][0]
t_attack = N_WARMUP * dt_val
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
fig.suptitle('Predator+SIS then removal, heterogeneous gamma: does slow-targeting '
             'kill the post-removal epidemic? (%d seeds)' % N_SEEDS, fontsize=11)

colors = {('none',0.0):'black', ('random',0.20):'gray', ('slow',0.20):'crimson',
          ('random',0.30):'steelblue', ('slow',0.30):'darkred',
          ('random',0.50):'lightseagreen', ('slow',0.50):'purple'}
labels = {('none',0.0):'no vaccination', ('random',0.20):'random p=0.20',
          ('slow',0.20):'slow p=0.20', ('random',0.30):'random p=0.30',
          ('slow',0.30):'slow p=0.30', ('random',0.50):'random p=0.50',
          ('slow',0.50):'slow p=0.50'}

ax = axes[0]
ax.axvspan(t_attack, t_remove, color='red', alpha=0.08, label='encirclement active')
for key in CONFIGS:
    f_arr = np.array([r[2] for r in runs[key]])
    ax.plot(t, f_arr.mean(0), color=colors[key], lw=1.8, label=labels[key])
ax.axvline(t_remove, ls=':', color='red', alpha=0.6)
ax.set_ylabel('panic fraction f'); ax.set_ylim(-0.05, 1.05)
ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)
ax.set_title('Outbreak (shaded) then predator removal (dotted)')

ax = axes[1]
ax.axvspan(t_attack, t_remove, color='red', alpha=0.08)
for key in CONFIGS:
    phi_arr = np.array([r[1] for r in runs[key]])
    ax.plot(t, phi_arr.mean(0), color=colors[key], lw=1.8, label=labels[key])
ax.axvline(t_remove, ls=':', color='red', alpha=0.6)
ax.set_ylabel('Phi'); ax.set_xlabel('time'); ax.set_ylim(0, 1.05)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/predator_slow_vaccination_1.png', dpi=120)
plt.close()
print('\n  --> figures/predator_slow_vaccination_1.png')
print('\nPredator + slow-recoverer vaccination analysis complete.')
