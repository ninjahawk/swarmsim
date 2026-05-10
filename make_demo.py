# make_demo.py -- Generate animated GIF for README
# Shows: agents colored by heading angle (rainbow), red predator star, dark background.
# Phase 1 (t=0..3): flock forms from random initial conditions
# Phase 2 (t=3..6): predator enters and chases

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from flocking import params, buffer, force, order_parameter
import os

os.makedirs('figures', exist_ok=True)
np.random.seed(42)

p = params(dict(
    N=120, n_iter=700, dt=0.01,
    r0=0.005, eps=0.1, rf=0.1,
    alpha=1.5, v0=0.02, mu=10.0, ramp=0.08,
))
PRED = dict(v0_pred=0.05, mu_pred=10.0, alpha_pred=6.0,
            r0_pred=0.1, eps_pred=2.5, ramp_pred=0.3)

N = p['N']; dt = p['dt']
x = np.zeros(2*N)
x[:N] = np.random.uniform(0, 1, N)
x[N:] = np.random.uniform(0, 1, N)
vx = np.random.uniform(-1, 1, N) * p['v0']
vy = np.random.uniform(-1, 1, N) * p['v0']

pred_x = 0.05; pred_y = 0.05
pred_vx = 0.01; pred_vy = 0.01
PRED_START = 300   # step at which predator begins chasing

RECORD_EVERY = 3
frames = []

for i in range(p['n_iter']):
    rb = max(p['r0'], p['rf'])
    nb, xb, yb, vxb, vyb = buffer(rb, x, vx, vy, N)
    fx, fy = force(nb, xb, yb, vxb, vyb, x, vx, vy, p)

    if i >= PRED_START:
        for j in range(N):
            ddx = pred_x - x[j];   ddx -= round(ddx)
            ddy = pred_y - x[N+j]; ddy -= round(ddy)
            d = np.sqrt(ddx**2 + ddy**2)
            if 0 < d <= PRED['r0_pred']:
                s = PRED['eps_pred'] * (1 - d/PRED['r0_pred'])**1.5 / d
                fx[j] -= s * ddx
                fy[j] -= s * ddy

        cx = np.arctan2(np.sin(2*np.pi*x[:N]).mean(),
                        np.cos(2*np.pi*x[:N]).mean()) / (2*np.pi) % 1.
        cy = np.arctan2(np.sin(2*np.pi*x[N:]).mean(),
                        np.cos(2*np.pi*x[N:]).mean()) / (2*np.pi) % 1.
        tx = cx - pred_x; tx -= round(tx)
        ty = cy - pred_y; ty -= round(ty)
        dist = np.sqrt(tx**2 + ty**2)
        if dist > 0:
            tx /= dist; ty /= dist
        sp = np.sqrt(pred_vx**2 + pred_vy**2)
        pfx = PRED['alpha_pred'] * tx + PRED['ramp_pred'] * np.random.uniform(-1, 1)
        pfy = PRED['alpha_pred'] * ty + PRED['ramp_pred'] * np.random.uniform(-1, 1)
        if sp > 0:
            pfx += PRED['mu_pred'] * (PRED['v0_pred'] - sp) * pred_vx / sp
            pfy += PRED['mu_pred'] * (PRED['v0_pred'] - sp) * pred_vy / sp
        pred_vx += pfx * dt; pred_vy += pfy * dt
        pred_x = (pred_x + pred_vx * dt) % 1.
        pred_y = (pred_y + pred_vy * dt) % 1.

    vx += fx * dt; vy += fy * dt
    x[:N] = (x[:N] + vx * dt) % 1.
    x[N:] = (x[N:] + vy * dt) % 1.

    if i % RECORD_EVERY == 0:
        angles = np.arctan2(vy, vx)
        speeds = np.sqrt(vx**2 + vy**2)
        show_pred = i >= PRED_START
        frames.append((
            x[:N].copy(), x[N:].copy(),
            vx.copy(), vy.copy(),
            angles.copy(), speeds.copy(),
            pred_x, pred_y, show_pred,
            order_parameter(vx, vy),
        ))

# ── Figure setup ──────────────────────────────────────────────────────────────
BG = '#0d1117'
fig, ax = plt.subplots(figsize=(5, 5), facecolor=BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xticks([]); ax.set_yticks([])
for sp in ax.spines.values():
    sp.set_edgecolor('#21262d')

scat = ax.scatter([], [], s=9, cmap='hsv', vmin=-np.pi, vmax=np.pi,
                  alpha=0.9, linewidths=0)
pred_dot = ax.scatter([], [], s=220, color='#ff3333', marker='*',
                      zorder=6, linewidths=0)

phi_text = ax.text(0.04, 0.96, '', transform=ax.transAxes,
                   ha='left', va='top', color='#58a6ff',
                   fontsize=9, fontfamily='monospace')
phase_text = ax.text(0.96, 0.96, '', transform=ax.transAxes,
                     ha='right', va='top', color='#8b949e',
                     fontsize=8, fontfamily='monospace')

def update(fi):
    px, py, pvx, pvy, angles, speeds, prx, pry, show_pred, phi = frames[fi]
    scat.set_offsets(np.c_[px, py])
    scat.set_array(angles)
    if show_pred:
        pred_dot.set_offsets([[prx, pry]])
        pred_dot.set_sizes([220])
        phase_text.set_text('predator active')
    else:
        pred_dot.set_offsets(np.empty((0, 2)))
        phase_text.set_text('flock forming')
    phi_text.set_text(f'Phi = {phi:.3f}')
    return scat, pred_dot, phi_text, phase_text

ani = animation.FuncAnimation(fig, update, frames=len(frames),
                               interval=50, blit=True)

print(f'Rendering {len(frames)} frames...')
ani.save('figures/demo.gif', writer='pillow', fps=20, dpi=100)
plt.close()
print('Saved: figures/demo.gif')
