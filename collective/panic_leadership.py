# panic_leadership.py -- can leaders steer a flock while PANIC spreads through it?
# (the contagion thread meets the leadership thread)
#
# F78 showed leadership counters encirclement: a shared heading restores coherence the predator
# removed. Encirclement attacks ALIGNMENT (pushes sub-groups apart). Contagion attacks differently:
# panic (SIS, book Sec 10.5) makes agents ERRATIC, and a panicked leader cannot lead. So panic
# severs the shared signal at its SOURCE (disabled leaders) and adds noise that fights alignment.
# Question: is there a contagion threshold for LOSS OF STEERABILITY -- a beta above which spreading
# panic collapses the flock's ability to follow its leaders -- and how does it relate to the SIS
# epidemic threshold (F25)?
#
# SIS panic among prey: calm<->panic, transmission rate beta per panicked flock-neighbor, recovery
# gamma. Panicked agents get high noise (erratic) and, if informed, suspend their goal bias while
# panicked. Sweep beta at fixed gamma; rho=0.10 informed toward +x.
# Metrics: steady panic fraction f_ss, steering accuracy toward goal, order parameter Phi.

import os
import numpy as np
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from flocking import buffer as buf_fn, order_parameter

os.makedirs('figures', exist_ok=True)
N_SEEDS = 5

BASE = dict(N=350, r0=0.005, eps=0.1, rf=0.1, alpha=1.0, v0=1.0, mu=10.0, ramp=0.5, dt=0.01)
N_WARMUP = 1500
N_ITER   = 5000
W_LEAD   = 1.0
RHO      = 0.10
G_HAT    = np.array([1.0, 0.0])
GAMMA      = 1.0      # panic recovery rate
RAMP_PANIC = 5.0      # erratic noise amplitude when panicked (book Sec 10.5 panic)
F_SEED     = 0.05     # initial panicked fraction


def run(beta, seed):
    rng = np.random.RandomState(seed)
    p = BASE; N = p['N']; dt = p['dt']
    r0, eps, rf = p['r0'], p['eps'], p['rf']
    v0, mu, alpha = p['v0'], p['mu'], p['alpha']

    n_inf = int(round(RHO * N))
    informed = np.zeros(N, dtype=bool)
    if n_inf > 0:
        informed[rng.choice(N, size=n_inf, replace=False)] = True

    panic = np.zeros(N, dtype=bool)
    panic[rng.choice(N, size=int(round(F_SEED * N)), replace=False)] = True

    x = np.zeros(2*N)
    x[:N] = rng.uniform(0., 1., N); x[N:] = rng.uniform(0., 1., N)
    vx = rng.uniform(-1., 1., N) * v0
    vy = rng.uniform(-1., 1., N) * v0

    rb = max(r0, rf)
    acc_rec = []; phi_rec = []; f_rec = []
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

        # noise: erratic (high ramp) for panicked agents, normal otherwise
        ramp_vec = np.where(panic, RAMP_PANIC, p['ramp'])
        frandx = ramp_vec * rng.uniform(-1., 1., N)
        frandy = ramp_vec * rng.uniform(-1., 1., N)

        fx = flockx + repx + fpropx + frandx
        fy = flocky + repy + fpropy + frandy

        # leader bias only on informed agents that are NOT currently panicked
        lead_mask = informed & (~panic)
        if lead_mask.any():
            fx[lead_mask] += W_LEAD * G_HAT[0]; fy[lead_mask] += W_LEAD * G_HAT[1]

        vx += fx * dt; vy += fy * dt
        x[:N] = (x[:N] + vx*dt) % 1.; x[N:] = (x[N:] + vy*dt) % 1.

        # SIS panic update. Count panicked flock-neighbors via periodic min-image distances
        # among REAL agents (the buffer's ghost ordering can't be inverted by index).
        dxr = x[:N][np.newaxis, :] - x[:N][:, np.newaxis]; dxr -= np.round(dxr)
        dyr = x[N:][np.newaxis, :] - x[N:][:, np.newaxis]; dyr -= np.round(dyr)
        d2r = dxr**2 + dyr**2
        neigh = (d2r <= rf**2) & (d2r > 0.)
        k_pan = (neigh & panic[np.newaxis, :]).sum(axis=1)
        p_inf = 1.0 - np.exp(-beta * k_pan * dt)
        new_inf = (~panic) & (rng.random(N) < p_inf)
        rec = panic & (rng.random(N) < (1.0 - np.exp(-GAMMA * dt)))
        panic = (panic | new_inf) & (~rec)

        if i >= N_WARMUP:
            mvx = vx.mean(); mvy = vy.mean(); mn = np.hypot(mvx, mvy)
            acc_rec.append((mvx*G_HAT[0] + mvy*G_HAT[1])/mn if mn > 1e-9 else 0.0)
            phi_rec.append(order_parameter(vx, vy))
            f_rec.append(panic.mean())
    return float(np.mean(acc_rec)), float(np.mean(phi_rec)), float(np.mean(f_rec))


def main():
  print('Leadership under spreading PANIC (SIS): does contagion collapse steerability?')
  print('  N=%d  pure-flock  rho=%.2f  w_lead=%.1f  gamma=%.1f  ramp_panic=%.1f  %d seeds\n'
      % (BASE['N'], RHO, W_LEAD, GAMMA, RAMP_PANIC, N_SEEDS))

  BETA_VALS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
  results = {}
  for beta in BETA_VALS:
    accs = []; phis = []; fs = []
    for s in range(N_SEEDS):
        a, ph, f = run(beta, s)
        accs.append(a); phis.append(ph); fs.append(f)
    results[beta] = (np.mean(accs), np.std(accs), np.mean(phis), np.mean(fs))
    print('   beta=%4.1f (beta/gamma=%4.1f)  panic_frac=%.3f  accuracy=%+.3f +/- %.3f  Phi=%.3f'
          % (beta, beta/GAMMA, np.mean(fs), np.mean(accs), np.std(accs), np.mean(phis)))

  fig, ax = plt.subplots(figsize=(8.5, 5.5))
  fig.suptitle('Leadership under spreading panic (N=%d, rho=%.2f, %d seeds)'
               % (BASE['N'], RHO, N_SEEDS), fontsize=11)
  acc = [results[b][0] for b in BETA_VALS]; err = [results[b][1] for b in BETA_VALS]
  phi = [results[b][2] for b in BETA_VALS]; fr = [results[b][3] for b in BETA_VALS]
  ax.errorbar(BETA_VALS, acc, yerr=err, marker='o', capsize=4, lw=2, color='steelblue',
              label='steering accuracy')
  ax.plot(BETA_VALS, phi, 's--', color='seagreen', lw=2, label='order parameter Phi')
  ax.plot(BETA_VALS, fr, '^:', color='crimson', lw=2, label='panic fraction f_ss')
  ax.set_xlabel('panic transmission rate beta (gamma=%.1f)' % GAMMA)
  ax.set_ylabel('value'); ax.set_ylim(-0.1, 1.05)
  ax.grid(alpha=0.3); ax.legend(fontsize=10)
  plt.tight_layout()
  plt.savefig('figures/panic_leadership_1.png', dpi=120)
  plt.close()
  print('\n  --> figures/panic_leadership_1.png')
  print('\nLeadership-under-panic analysis complete.')


if __name__ == '__main__':
    main()
