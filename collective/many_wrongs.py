# many_wrongs.py -- does a NOISY-private-estimate follower rule recover the
# "many wrongs make a right" (wisdom-of-crowds) scaling that F81 found ABSENT?
#
# F81 showed that with leaders carrying an EXACT SHARED goal vector, linear
# velocity alignment delivers exactly the per-capita pull (total leader force / N)
# with NO group-size amplification: a fixed NUMBER of leaders does not suffice as
# N grows, only the FRACTION matters. F81 predicted that recovering the animal-
# navigation literature's "a fixed number suffices in large groups" scaling would
# require a MANY-WRONGS follower rule -- each agent holding an independent NOISY
# estimate of the goal, so that averaging over more agents cancels the noise.
#
# This implements exactly that. EVERY agent i carries a private preferred
# direction g_hat_i = (cos phi_i, sin phi_i) with phi_i = N(0, sigma_pref) about
# the true goal (+x), fixed for the run. Each agent feels a bias w * g_hat_i
# toward its OWN noisy estimate. No exact-vector leaders; everyone is a noisy one.
#
# Many-wrongs prediction: the flock's STEADY heading is the alignment-averaged
# resultant of N independent noisy biases, whose angular error scales as
# sigma_pref / sqrt(N). So directional accuracy toward the TRUE goal should
# IMPROVE with N -- the opposite of F81's exact-vector per-capita result -- and
# the cross-seed RMS heading error should fall like 1/sqrt(N). The catch: global
# averaging requires the flock to stay COHERENT (one cluster); if the
# heterogeneous biases fragment it, averaging is only local and the law breaks.
#
# Exp1: sweep N at fixed sigma_pref -- test the 1/sqrt(N) law and the role of Phi.
# Exp2: sweep sigma_pref at fixed N -- test that error scales with sigma_pref.
# Metric: signed steady heading error (rad) per seed -> RMS over seeds; accuracy
# = cos(error); Phi = order parameter.

import os
import numpy as np
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from flocking import buffer as buf_fn, order_parameter

os.makedirs('figures', exist_ok=True)
N_SEEDS = 8

BASE = dict(r0=0.005, eps=0.1, rf=0.1, alpha=1.0, v0=1.0, mu=10.0, ramp=0.5, dt=0.01)
N_WARMUP = 1000
N_ITER   = 3000
W_BIAS   = 0.5            # per-agent bias strength toward its own estimate (alpha=1.0 ref)


def run(N, sigma_pref, seed):
    """Every agent biased toward its own noisy goal estimate. Returns
    (signed steady heading error [rad], mean per-step accuracy, mean Phi)."""
    rng = np.random.RandomState(seed)
    p = BASE; dt = p['dt']
    r0, eps, rf = p['r0'], p['eps'], p['rf']
    v0, mu, alpha = p['v0'], p['mu'], p['alpha']

    # per-agent fixed preferred direction: angle ~ N(0, sigma_pref) about +x
    phi = rng.normal(0.0, sigma_pref, N)
    gx = np.cos(phi); gy = np.sin(phi)

    x = np.zeros(2*N)
    x[:N] = rng.uniform(0., 1., N); x[N:] = rng.uniform(0., 1., N)
    vx = rng.uniform(-1., 1., N) * v0
    vy = rng.uniform(-1., 1., N) * v0

    rb = max(r0, rf)
    mvx_rec = []; mvy_rec = []; acc_rec = []; phi_rec = []
    for i in range(N_ITER):
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
        d_safe   = np.where(rep_mask, np.sqrt(d2), 1.)
        base_r   = np.where(rep_mask, 1. - d_safe/(2.*r0), 0.)
        strength = np.where(rep_mask, eps * base_r**1.5 / d_safe, 0.)
        repx = (-strength * dx).sum(axis=1); repy = (-strength * dy).sum(axis=1)

        vnorm = np.sqrt(vx**2 + vy**2)
        vnorms = np.where(vnorm == 0, 1., vnorm)
        fpropx = mu * (v0 - vnorm) * vx / vnorms
        fpropy = mu * (v0 - vnorm) * vy / vnorms

        frandx = p['ramp'] * rng.uniform(-1., 1., N)
        frandy = p['ramp'] * rng.uniform(-1., 1., N)

        fx = flockx + repx + fpropx + frandx + W_BIAS * gx
        fy = flocky + repy + fpropy + frandy + W_BIAS * gy

        vx += fx * dt; vy += fy * dt
        x[:N] = (x[:N] + vx*dt) % 1.; x[N:] = (x[N:] + vy*dt) % 1.

        if i >= N_WARMUP:
            mvx = vx.mean(); mvy = vy.mean(); mn = np.hypot(mvx, mvy)
            mvx_rec.append(mvx); mvy_rec.append(mvy)
            acc_rec.append(mvx/mn if mn > 1e-9 else 0.0)
            phi_rec.append(order_parameter(vx, vy))
    # steady heading = angle of the time-averaged mean-velocity vector
    err = np.arctan2(np.mean(mvy_rec), np.mean(mvx_rec))   # signed, rad; true goal at 0
    return float(err), float(np.mean(acc_rec)), float(np.mean(phi_rec))


def sweep_over_seeds(N, sigma_pref):
    errs = []; accs = []; phis = []
    for s in range(N_SEEDS):
        e, a, ph = run(N, sigma_pref, s)
        errs.append(e); accs.append(a); phis.append(ph)
    errs = np.array(errs)
    rms_deg = np.degrees(np.sqrt(np.mean(errs**2)))
    return rms_deg, float(np.mean(accs)), float(np.std(accs)), float(np.mean(phis))


# ===========================================================================
print('Many-wrongs navigation: do NOISY private estimates average to recover')
print('the 1/sqrt(N) wisdom-of-crowds scaling F81 found absent for exact vectors?')
print('  w_bias=%.1f  %d seeds  pure-flock  alpha=1.0\n' % (W_BIAS, N_SEEDS))

# --- Exp1: sweep N at fixed sigma_pref ---
SIGMA1 = 1.0   # rad (~57 deg) per-agent preferred-direction error
N_VALS = [30, 60, 125, 250, 500, 1000]
print('== Exp1: sweep N at sigma_pref=%.2f rad (%.0f deg) ==' % (SIGMA1, np.degrees(SIGMA1)))
print('   (single-agent baseline RMS heading error would be ~%.0f deg)' % np.degrees(SIGMA1))
res1 = {}
for N in N_VALS:
    rms, acc, accsd, phi = sweep_over_seeds(N, SIGMA1)
    res1[N] = (rms, acc, accsd, phi)
    print('   N=%4d  RMS heading err=%5.1f deg  accuracy=%+.3f +/- %.3f  Phi=%.3f'
          % (N, rms, acc, accsd, phi))

# 1/sqrt(N) reference normalised to the smallest-N point
ref_N0 = N_VALS[0]; ref_rms0 = res1[ref_N0][0]
print('   --- 1/sqrt(N) prediction (normalised at N=%d): ---' % ref_N0)
for N in N_VALS:
    pred = ref_rms0 * np.sqrt(ref_N0 / N)
    print('   N=%4d  predicted %5.1f deg   measured %5.1f deg' % (N, pred, res1[N][0]))

# log-log slope of RMS error vs N (expect ~ -0.5)
logN = np.log(np.array(N_VALS, float)); logE = np.log([res1[N][0] for N in N_VALS])
slope = np.polyfit(logN, logE, 1)[0]
print('   --> log-log slope d(log RMSerr)/d(log N) = %.3f  (many-wrongs predicts -0.5)\n' % slope)

# --- Exp2: sweep sigma_pref at fixed N ---
N_FIX = 250
SIGMA_VALS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
print('== Exp2: sweep sigma_pref at N=%d ==' % N_FIX)
res2 = {}
for sg in SIGMA_VALS:
    rms, acc, accsd, phi = sweep_over_seeds(N_FIX, sg)
    res2[sg] = (rms, acc, accsd, phi)
    print('   sigma_pref=%.2f rad (%3.0f deg)  RMS err=%5.1f deg  accuracy=%+.3f +/- %.3f  Phi=%.3f'
          % (sg, np.degrees(sg), rms, acc, accsd, phi))
print('   (prediction: RMS err ~ sigma_pref/sqrt(N), i.e. linear in sigma_pref)\n')

# ===========================================================================
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle('Many-wrongs navigation: noisy private goal estimates averaged by alignment '
             '(w_bias=%.1f, %d seeds)' % (W_BIAS, N_SEEDS), fontsize=12)

ax = axes[0]
rms_vals = [res1[N][0] for N in N_VALS]
ax.loglog(N_VALS, rms_vals, 'o-', lw=2, color='crimson', label='measured RMS heading error')
ref = [ref_rms0 * np.sqrt(ref_N0 / N) for N in N_VALS]
ax.loglog(N_VALS, ref, '--', lw=1.5, color='gray', label='1/sqrt(N) reference')
ax.set_xlabel('flock size N'); ax.set_ylabel('cross-seed RMS heading error (deg)')
ax.set_title('Exp1: error vs N (slope=%.2f, predict -0.5)' % slope)
ax.grid(alpha=0.3, which='both'); ax.legend(fontsize=9)

ax = axes[1]
acc1 = [res1[N][1] for N in N_VALS]; err1 = [res1[N][2] for N in N_VALS]
phi1 = [res1[N][3] for N in N_VALS]
ax.errorbar(N_VALS, acc1, yerr=err1, marker='o', capsize=4, lw=2, color='steelblue',
            label='accuracy toward true goal')
ax.plot(N_VALS, phi1, 's--', lw=1.5, color='seagreen', label='Phi (coherence)')
ax.set_xscale('log'); ax.set_xlabel('flock size N'); ax.set_ylabel('accuracy / Phi')
ax.set_title('Exp1: accuracy IMPROVES with N (many-wrongs)')
ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=9)

ax = axes[2]
rms2 = [res2[sg][0] for sg in SIGMA_VALS]
ax.plot([np.degrees(s) for s in SIGMA_VALS], rms2, 'o-', lw=2, color='darkorange')
ax.set_xlabel('per-agent preferred-direction error sigma_pref (deg)')
ax.set_ylabel('cross-seed RMS heading error (deg)')
ax.set_title('Exp2: error vs sigma_pref at N=%d' % N_FIX)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('figures/many_wrongs_1.png', dpi=120)
plt.close()
print('  --> figures/many_wrongs_1.png')
print('\nMany-wrongs navigation analysis complete.')
