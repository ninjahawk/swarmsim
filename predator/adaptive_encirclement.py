# adaptive_encirclement.py -- Adaptive vs fixed R_enc encirclement
#
# Finding 31 showed that optimal disruption is at R_enc/Rg ~ 0.5 for both N=350 and
# N=1000.  Finding 32 showed that long-time encirclement produces a merge/split cycle
# where Rg oscillates as sub-flocks form and re-coalesce.  Under fixed R_enc, the
# predator geometry becomes sub-optimal when the flock contracts (Rg small ->
# R_enc/Rg > 0.5, predators orbit outside) or expands (Rg large -> R_enc/Rg < 0.5,
# predators too central).
#
# Hypothesis: Adaptive predators that set R_enc = 0.5 * Rg (updated live each step)
# maintain optimal geometry throughout the merge/split cycle and should produce:
#   (a) lower mean Phi than fixed R_enc
#   (b) smaller temporal variance in Phi (more sustained disruption)
#   (c) suppression of the merge event (flock cannot re-assemble above Phi ~ 0.7)
#
# Experiment design:
#   Compare fixed vs adaptive at N=350, n_pred=10, 15000 steps (150 time units)
#   Fixed:    R_enc = 0.15 (optimal for N=350 from Finding 31)
#   Adaptive: R_enc = 0.5 * live_Rg (computed from flock positions each step)
#   Metrics per run:  mean Phi, temporal std Phi, mean Rg, fraction of time Phi > 0.85
#                     (proxy for how often the flock successfully re-coalesces)

import os
import numpy as np
import matplotlib.pyplot as plt
from flocking import buffer as buf_fn, order_parameter
from model import _periodic_com, _periodic_disp

os.makedirs('figures', exist_ok=True)

N_SEEDS = 5
N_WARMUP = 1000
N_STEPS  = 15000   # 150 time units
RECORD_EVERY = 50
RG_RATIO = 0.5     # adaptive target: R_enc = RG_RATIO * Rg

BASE = dict(
    N=350, r0=0.005, eps=0.1, rf=0.1,
    alpha=1.0, v0=0.02, mu=10.0, ramp=0.1, dt=0.01,
)
FIXED_RENC = 0.15  # optimal for N=350 per Finding 31

PRED = dict(v0=0.05, mu=10.0, alpha=5.0, r0=0.1, eps=2.0, ramp=1.0)
N_PRED = 10


def compute_rg(x, N):
    cx = _periodic_com(x[:N])
    cy = _periodic_com(x[N:])
    dx = x[:N] - cx; dx -= np.round(dx)
    dy = x[N:] - cy; dy -= np.round(dy)
    return np.sqrt((dx**2 + dy**2).mean())


def run_one(seed, adaptive):
    np.random.seed(seed)
    p = BASE.copy()
    N = p['N']; dt = p['dt']
    r0, eps, rf = p['r0'], p['eps'], p['rf']
    v0, mu = p['v0'], p['mu']
    rb = max(r0, rf)

    x = np.zeros(2*N)
    x[:N] = np.random.uniform(0., 1., N)
    x[N:] = np.random.uniform(0., 1., N)
    vx = np.random.uniform(-1., 1., N) * v0
    vy = np.random.uniform(-1., 1., N) * v0

    pred_x  = np.random.uniform(0., 1., N_PRED)
    pred_y  = np.random.uniform(0., 1., N_PRED)
    pred_vx = np.zeros(N_PRED); pred_vy = np.zeros(N_PRED)
    pred_angles = np.radians(np.arange(N_PRED) * (360.0 / N_PRED))

    def step(x, vx, vy, alpha_arr, ramp_arr, enc_radius=None):
        nb, xb, yb, vxb, vyb = buf_fn(rb, x, vx, vy, N)
        dx = xb[:nb][np.newaxis, :] - x[:N][:, np.newaxis]
        dy = yb[:nb][np.newaxis, :] - x[N:][:, np.newaxis]
        d2 = dx**2 + dy**2

        not_self = np.ones((N, nb), dtype=bool)
        idx = np.arange(min(N, nb))
        not_self[idx, idx] = False

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
        base_r   = np.where(rep_mask, 1. - d_safe/(2.*r0), 0.)
        strength = np.where(rep_mask, eps * base_r**1.5 / d_safe, 0.)
        repx = (-strength * dx).sum(axis=1)
        repy = (-strength * dy).sum(axis=1)

        vnorm  = np.sqrt(vx**2 + vy**2)
        vnorms = np.where(vnorm == 0, 1., vnorm)
        fpropx = mu * (v0 - vnorm) * vx / vnorms
        fpropy = mu * (v0 - vnorm) * vy / vnorms

        frandx = ramp_arr * np.random.uniform(-1., 1., N)
        frandy = ramp_arr * np.random.uniform(-1., 1., N)

        fx_total = flockx + repx + fpropx + frandx
        fy_total = flocky + repy + fpropy + frandy

        if enc_radius is not None:
            for k in range(N_PRED):
                ddx = _periodic_disp(pred_x[k], x[:N])
                ddy = _periodic_disp(pred_y[k], x[N:])
                d = np.sqrt(ddx**2 + ddy**2)
                mask_p = (d > 0) & (d <= PRED['r0'])
                if mask_p.any():
                    sp = PRED['eps'] * (1. - d[mask_p]/PRED['r0'])**1.5 / d[mask_p]
                    fx_total[mask_p] -= sp * ddx[mask_p]
                    fy_total[mask_p] -= sp * ddy[mask_p]

        return fx_total, fy_total

    alpha_arr = np.full(N, p['alpha'])
    ramp_arr  = np.full(N, p['ramp'])

    # warmup
    for _ in range(N_WARMUP):
        fx, fy = step(x, vx, vy, alpha_arr, ramp_arr, enc_radius=None)
        vx += fx * dt; vy += fy * dt
        x[:N] = (x[:N] + vx*dt) % 1.
        x[N:] = (x[N:] + vy*dt) % 1.

    phi_t = []; rg_t = []; renc_t = []

    for i in range(N_STEPS):
        rg = compute_rg(x, N)
        enc_radius = RG_RATIO * rg if adaptive else FIXED_RENC
        enc_radius = max(enc_radius, 0.02)  # safety floor

        fx, fy = step(x, vx, vy, alpha_arr, ramp_arr, enc_radius=enc_radius)
        vx += fx * dt; vy += fy * dt
        x[:N] = (x[:N] + vx*dt) % 1.
        x[N:] = (x[N:] + vy*dt) % 1.

        # move predators
        cx, cy = _periodic_com(x[:N]), _periodic_com(x[N:])
        for k in range(N_PRED):
            tx = (cx + enc_radius * np.cos(pred_angles[k])) % 1.
            ty = (cy + enc_radius * np.sin(pred_angles[k])) % 1.
            ddx = _periodic_disp(tx, pred_x[k])
            ddy = _periodic_disp(ty, pred_y[k])
            dist = np.sqrt(ddx**2 + ddy**2)
            if dist > 0:
                ddx /= dist; ddy /= dist
            sp_p = np.sqrt(pred_vx[k]**2 + pred_vy[k]**2)
            if sp_p > 0:
                pfx = PRED['alpha']*ddx + PRED['mu']*(PRED['v0']-sp_p)*pred_vx[k]/sp_p
                pfy = PRED['alpha']*ddy + PRED['mu']*(PRED['v0']-sp_p)*pred_vy[k]/sp_p
            else:
                pfx = PRED['alpha']*ddx; pfy = PRED['alpha']*ddy
            pfx += PRED['ramp']*np.random.uniform(-1., 1.)
            pfy += PRED['ramp']*np.random.uniform(-1., 1.)
            pred_vx[k] += pfx*dt; pred_vy[k] += pfy*dt
            pred_x[k] = (pred_x[k] + pred_vx[k]*dt) % 1.
            pred_y[k] = (pred_y[k] + pred_vy[k]*dt) % 1.

        if i % RECORD_EVERY == 0:
            phi_t.append(order_parameter(vx, vy))
            rg_t.append(rg)
            renc_t.append(enc_radius)

    return np.array(phi_t), np.array(rg_t), np.array(renc_t)


print('Adaptive vs fixed R_enc encirclement')
print('N=%d n_pred=%d N_steps=%d' % (BASE['N'], N_PRED, N_STEPS))
print('Fixed R_enc=%.3f  Adaptive: R_enc = %.2f * live_Rg' % (FIXED_RENC, RG_RATIO))

fixed_runs = []; adap_runs = []
for s in range(N_SEEDS):
    print('  seed %d fixed...' % s, flush=True)
    fixed_runs.append(run_one(s, adaptive=False))
    print('  seed %d adaptive...' % s, flush=True)
    adap_runs.append(run_one(s, adaptive=True))

t_arr = np.arange(len(fixed_runs[0][0])) * RECORD_EVERY * BASE['dt']

def summarize(runs, name):
    phi_arr = np.array([r[0] for r in runs])
    rg_arr  = np.array([r[1] for r in runs])
    renc_arr= np.array([r[2] for r in runs])
    last = phi_arr[:, -int(50.0/(RECORD_EVERY*BASE['dt'])):]
    mean_phi = last.mean()
    std_phi  = last.std()
    # temporal std per seed (mean of per-seed temporal std)
    t_std = np.mean([phi_arr[s].std() for s in range(len(runs))])
    frac_high = (last > 0.85).mean()   # fraction of time Phi > 0.85
    print('%s:  mean_Phi=%.3f  seed_std=%.3f  temporal_std=%.3f  frac_above_0.85=%.2f' %
          (name, mean_phi, std_phi, t_std, frac_high))
    print('      mean_Rg=%.3f  mean_Renc=%.3f  mean_Renc/Rg=%.3f' % (
        rg_arr.mean(), renc_arr.mean(), (renc_arr/rg_arr).mean()))
    return phi_arr, rg_arr, renc_arr

print('\n=== Results (last 50 time units) ===')
f_phi, f_rg, f_renc = summarize(fixed_runs, 'fixed  ')
a_phi, a_rg, a_renc = summarize(adap_runs,  'adaptive')

# Plot
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

ax = axes[0]
for s in range(N_SEEDS):
    ax.plot(t_arr, f_phi[s], alpha=0.3, lw=0.7, color='steelblue')
    ax.plot(t_arr, a_phi[s], alpha=0.3, lw=0.7, color='crimson')
ax.plot(t_arr, f_phi.mean(0), color='steelblue', lw=2, label='fixed R_enc=0.15')
ax.plot(t_arr, a_phi.mean(0), color='crimson',   lw=2, label='adaptive R_enc=0.5*Rg')
ax.axhline(0.85, ls='--', color='gray', lw=1)
ax.set_ylabel('Phi'); ax.set_ylim(0, 1.05)
ax.set_title('Adaptive vs fixed encirclement (N=%d, n_pred=%d)' % (BASE['N'], N_PRED))
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(t_arr, f_rg.mean(0), color='steelblue', lw=2, label='Rg (fixed)')
ax.plot(t_arr, a_rg.mean(0), color='crimson',   lw=2, label='Rg (adaptive)')
ax.set_ylabel('Radius of gyration Rg'); ax.grid(alpha=0.3); ax.legend(fontsize=9)

ax = axes[2]
ax.plot(t_arr, f_renc.mean(0), color='steelblue', lw=2, label='R_enc used (fixed)')
ax.plot(t_arr, a_renc.mean(0), color='crimson',   lw=2, label='R_enc used (adaptive)')
ax.set_ylabel('Encirclement radius R_enc')
ax.set_xlabel('Time')
ax.grid(alpha=0.3); ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('figures/adaptive_encirclement_1.png', dpi=120)
plt.close()
print('  --> figures/adaptive_encirclement_1.png')
