import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# =============================================================================
# PARAMETERS  (Table 10.1 defaults — change here to explore)
# =============================================================================
N       = 350       # number of agents
n_iter  = 2000      # temporal iterations
dt      = 0.01      # time step
r0      = 0.005     # repulsion radius
eps     = 0.1       # repulsion amplitude
rf      = 0.1       # flocking radius  (>= 4*r0)
alpha   = 1.0       # flocking amplitude
v0      = 1.0       # target speed
mu      = 10.0      # self-propulsion amplitude
ramp    = 0.5       # random force amplitude

# =============================================================================
# BUFFER FUNCTION  (Figure 10.2)
# Creates ghost copies of agents near boundaries so that force calculations
# wrap correctly on the periodic unit-square domain.
# =============================================================================
def buffer(rb, x, vx, vy):
    """Return expanded arrays xb, yb, vxb, vyb including ghost agents."""
    xb  = np.zeros(4 * N);  yb  = np.zeros(4 * N)
    vxb = np.zeros(4 * N);  vyb = np.zeros(4 * N)
    xb[0:N] = x[0:N];       yb[0:N] = x[N:2*N]
    vxb[0:N] = vx[0:N];     vyb[0:N] = vy[0:N]
    nb = N - 1

    for k in range(N):
        # --- edge ghosts ---
        if x[k] <= rb:                          # close to left
            nb += 1
            xb[nb] = x[k] + 1.
            yb[nb] = x[N+k];  vxb[nb] = vx[k];  vyb[nb] = vy[k]
        if x[k] >= 1. - rb:                     # close to right
            nb += 1
            xb[nb] = x[k] - 1.
            yb[nb] = x[N+k];  vxb[nb] = vx[k];  vyb[nb] = vy[k]
        if x[N+k] <= rb:                        # close to bottom
            nb += 1
            yb[nb] = x[N+k] + 1.
            xb[nb] = x[k];   vxb[nb] = vx[k];  vyb[nb] = vy[k]
        if x[N+k] >= 1. - rb:                   # close to top
            nb += 1
            yb[nb] = x[N+k] - 1.
            xb[nb] = x[k];   vxb[nb] = vx[k];  vyb[nb] = vy[k]
        # --- corner ghosts ---
        if x[k] <= rb and x[N+k] <= rb:         # bottom-left
            nb += 1
            xb[nb] = x[k] + 1.;  yb[nb] = x[N+k] + 1.
            vxb[nb] = vx[k];      vyb[nb] = vy[k]
        if x[k] >= 1.-rb and x[N+k] <= rb:      # bottom-right
            nb += 1
            xb[nb] = x[k] - 1.;  yb[nb] = x[N+k] + 1.
            vxb[nb] = vx[k];      vyb[nb] = vy[k]
        if x[k] <= rb and x[N+k] >= 1.-rb:      # top-left
            nb += 1
            xb[nb] = x[k] + 1.;  yb[nb] = x[N+k] - 1.
            vxb[nb] = vx[k];      vyb[nb] = vy[k]
        if x[k] >= 1.-rb and x[N+k] >= 1.-rb:   # top-right
            nb += 1
            xb[nb] = x[k] - 1.;  yb[nb] = x[N+k] - 1.
            vxb[nb] = vx[k];      vyb[nb] = vy[k]

    return nb, xb, yb, vxb, vyb


# =============================================================================
# FORCE FUNCTION  (Figure 10.3)
# Computes total force on every real agent j from all agents + ghosts.
# =============================================================================
def force(nb, xb, yb, vxb, vyb, x, vx, vy):
    """Return fx, fy arrays of total force on each of the N real agents."""
    fx = np.zeros(N);  fy = np.zeros(N)

    for j in range(N):
        repx = 0.;  repy = 0.
        flockx = 0.;  flocky = 0.;  nflock = 0

        for k in range(nb):
            d2 = (xb[k] - x[j])**2 + (yb[k] - x[N+j])**2
            if d2 <= rf**2 and j != k:          # within flocking radius
                flockx += vxb[k]
                flocky += vyb[k]
                nflock += 1
                if d2 <= (2.*r0)**2:            # within repulsion radius
                    d = np.sqrt(d2)
                    repx += eps * (1. - d/(2.*r0))**1.5 * (x[j]   - xb[k]) / d
                    repy += eps * (1. - d/(2.*r0))**1.5 * (x[N+j] - yb[k]) / d

        normflock = np.sqrt(flockx**2 + flocky**2)
        if nflock == 0:
            normflock = 1.                      # avoid 0/0
        flockx = alpha * flockx / normflock
        flocky = alpha * flocky / normflock

        vnorm  = np.sqrt(vx[j]**2 + vy[j]**2)
        fpropx = mu * (v0 - vnorm) * (vx[j] / vnorm)
        fpropy = mu * (v0 - vnorm) * (vy[j] / vnorm)

        frandx = ramp * np.random.uniform(-1., 1.)
        frandy = ramp * np.random.uniform(-1., 1.)

        fx[j] = flockx + frandx + fpropx + repx
        fy[j] = flocky + frandy + fpropy + repy

    return fx, fy


# =============================================================================
# INITIALISATION
# =============================================================================
def initialise():
    """Random positions and velocities in the unit square."""
    x  = np.zeros(2 * N)    # x[0:N] = x-coords, x[N:2N] = y-coords
    vx = np.zeros(N)
    vy = np.zeros(N)
    for j in range(N):
        x[j]   = np.random.uniform()
        x[N+j] = np.random.uniform()
        vx[j]  = np.random.uniform(-1., 1.)
        vy[j]  = np.random.uniform(-1., 1.)
    return x, vx, vy


# =============================================================================
# MAIN SIMULATION LOOP  (returns snapshots for animation)
# =============================================================================
def run(n_frames=200, frame_every=None):
    """
    Run the simulation and return a list of (pos_x, pos_y, vel_x, vel_y)
    snapshots for animation.  frame_every defaults to n_iter // n_frames.
    """
    if frame_every is None:
        frame_every = max(1, n_iter // n_frames)

    x, vx, vy = initialise()
    frames = []

    for i in range(n_iter):
        nb, xb, yb, vxb, vyb = buffer(max(r0, rf), x, vx, vy)
        fx, fy = force(nb, xb, yb, vxb, vyb, x, vx, vy)

        vx += fx * dt
        vy += fy * dt
        x[0:N]   = (x[0:N]   + vx * dt) % 1.
        x[N:2*N] = (x[N:2*N] + vy * dt) % 1.

        if i % frame_every == 0:
            frames.append((x[0:N].copy(), x[N:2*N].copy(),
                           vx.copy(),     vy.copy()))

    return frames


# =============================================================================
# ANIMATION
# =============================================================================
def animate(frames):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 1);  ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title('Flocking simulation')
    time_text = ax.text(0.02, 0.97, '', transform=ax.transAxes,
                        fontsize=9, va='top')
    frame_step = max(1, n_iter // len(frames))

    # initialise with first frame so quiver knows the array size
    px0, py0, vx0, vy0 = frames[0]
    speeds0 = np.sqrt(vx0**2 + vy0**2);  speeds0[speeds0 == 0] = 1.
    scat = ax.scatter(px0, py0, s=4, color='steelblue', zorder=3)
    quiv = ax.quiver(px0, py0, vx0/speeds0, vy0/speeds0,
                     scale=40, width=0.002, color='firebrick', zorder=4)

    def update(frame_idx):
        px, py, vx, vy = frames[frame_idx]
        scat.set_offsets(np.column_stack([px, py]))
        speeds = np.sqrt(vx**2 + vy**2);  speeds[speeds == 0] = 1.
        quiv.set_offsets(np.column_stack([px, py]))
        quiv.set_UVC(vx / speeds, vy / speeds)
        time_text.set_text(f't = {frame_idx * frame_step * dt:.2f}')
        return scat, quiv, time_text

    ani = animation.FuncAnimation(fig, update, frames=len(frames),
                                  interval=50, blit=True)
    plt.tight_layout()
    plt.show()
    return ani


# =============================================================================
# STATIC SNAPSHOT (quick check without animation)
# =============================================================================
def plot_snapshot(frames, idx=-1):
    px, py, vx, vy = frames[idx]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(px, py, s=4, color='steelblue')
    speeds = np.sqrt(vx**2 + vy**2);  speeds[speeds == 0] = 1.
    ax.quiver(px, py, vx/speeds, vy/speeds,
              scale=30, width=0.003, color='firebrick')
    ax.set_xlim(0, 1);  ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title(f'Snapshot at t ≈ {(len(frames)-1 if idx==-1 else idx) * dt * (n_iter // len(frames)):.1f}')
    plt.tight_layout()
    plt.show()


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    print(f'Running flocking simulation: N={N}, n_iter={n_iter}, dt={dt}')
    print(f'Parameters: r0={r0}, eps={eps}, rf={rf}, alpha={alpha}, '
          f'v0={v0}, mu={mu}, ramp={ramp}')
    print('Simulating...')
    frames = run(n_frames=200)
    print(f'Done. Captured {len(frames)} frames.')
    animate(frames)
