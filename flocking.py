import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# =============================================================================
# DEFAULT PARAMETERS  (Table 10.1)
# =============================================================================
DEFAULT = dict(
    N      = 350,    # number of agents
    n_iter = 2000,   # time steps
    dt     = 0.01,   # time step size
    r0     = 0.005,  # repulsion radius
    eps    = 0.1,    # repulsion amplitude
    rf     = 0.1,    # flocking radius  (>= 4*r0)
    alpha  = 1.0,    # flocking amplitude
    v0     = 1.0,    # target speed
    mu     = 10.0,   # self-propulsion amplitude
    ramp   = 0.5,    # random force amplitude
)

def params(overrides=None):
    """Return a full parameter dict, merging DEFAULT with any overrides."""
    p = DEFAULT.copy()
    if overrides:
        p.update(overrides)
    return p


# =============================================================================
# BUFFER FUNCTION  (Figure 10.2)
# Replicates agents near the boundary outward by one unit so that force
# calculations wrap correctly on the periodic unit-square domain.
# =============================================================================
def buffer(rb, x, vx, vy, N):
    """
    x[0:N]  = x-positions, x[N:2N] = y-positions.
    Returns nb (count of real+ghost agents), xb, yb, vxb, vyb.
    """
    xb  = np.zeros(4 * N);  yb  = np.zeros(4 * N)
    vxb = np.zeros(4 * N);  vyb = np.zeros(4 * N)
    xb[:N] = x[:N];   yb[:N] = x[N:2*N]
    vxb[:N] = vx;     vyb[:N] = vy
    nb = N  # next free index

    for k in range(N):
        xk, yk = x[k], x[N+k]
        # edges
        if xk <= rb:
            xb[nb]=xk+1.; yb[nb]=yk; vxb[nb]=vx[k]; vyb[nb]=vy[k]; nb+=1
        if xk >= 1.-rb:
            xb[nb]=xk-1.; yb[nb]=yk; vxb[nb]=vx[k]; vyb[nb]=vy[k]; nb+=1
        if yk <= rb:
            xb[nb]=xk; yb[nb]=yk+1.; vxb[nb]=vx[k]; vyb[nb]=vy[k]; nb+=1
        if yk >= 1.-rb:
            xb[nb]=xk; yb[nb]=yk-1.; vxb[nb]=vx[k]; vyb[nb]=vy[k]; nb+=1
        # corners
        if xk <= rb and yk <= rb:
            xb[nb]=xk+1.; yb[nb]=yk+1.; vxb[nb]=vx[k]; vyb[nb]=vy[k]; nb+=1
        if xk >= 1.-rb and yk <= rb:
            xb[nb]=xk-1.; yb[nb]=yk+1.; vxb[nb]=vx[k]; vyb[nb]=vy[k]; nb+=1
        if xk <= rb and yk >= 1.-rb:
            xb[nb]=xk+1.; yb[nb]=yk-1.; vxb[nb]=vx[k]; vyb[nb]=vy[k]; nb+=1
        if xk >= 1.-rb and yk >= 1.-rb:
            xb[nb]=xk-1.; yb[nb]=yk-1.; vxb[nb]=vx[k]; vyb[nb]=vy[k]; nb+=1

    return nb, xb, yb, vxb, vyb


# =============================================================================
# FORCE FUNCTION  (Figure 10.3, vectorized)
# Computes all four forces on every real agent simultaneously using NumPy
# broadcasting instead of nested Python loops.
# =============================================================================
def force(nb, xb, yb, vxb, vyb, x, vx, vy, p):
    """Return fx, fy: force arrays for the N real agents."""
    N = p['N']
    r0, eps, rf, alpha, v0, mu, ramp = (
        p['r0'], p['eps'], p['rf'], p['alpha'],
        p['v0'], p['mu'],  p['ramp']
    )

    # distance matrix between real agents (rows) and all buffer agents (cols)
    # shapes: (N, nb)
    dx = xb[:nb][np.newaxis, :] - x[:N][:, np.newaxis]
    dy = yb[:nb][np.newaxis, :] - x[N:2*N][:, np.newaxis]
    d2 = dx**2 + dy**2

    # exclude self-interaction (only relevant for k < N)
    not_self = np.ones((N, nb), dtype=bool)
    idx = np.arange(min(N, nb))
    not_self[idx, idx] = False

    # --- flocking force (Eq. 10.2) ---
    flock_mask = (d2 <= rf**2) & not_self
    flockx = np.where(flock_mask, vxb[:nb], 0.).sum(axis=1)
    flocky = np.where(flock_mask, vyb[:nb], 0.).sum(axis=1)
    nflock = flock_mask.sum(axis=1)
    norm_f = np.sqrt(flockx**2 + flocky**2)
    norm_f[nflock == 0] = 1.   # avoid 0/0 when no neighbours
    flockx = alpha * flockx / norm_f
    flocky = alpha * flocky / norm_f

    # --- repulsion force (Eq. 10.1) ---
    rep_mask = (d2 <= (2*r0)**2) & not_self & (d2 > 0)
    d_safe = np.where(rep_mask, np.sqrt(d2), 1.)
    base   = np.where(rep_mask, 1. - d_safe/(2.*r0), 0.)   # 0 outside mask avoids negative^1.5
    strength = np.where(rep_mask, eps * base**1.5 / d_safe, 0.)
    # force on j is in direction (x_j - x_k), i.e. -dx and -dy
    repx = (-strength * dx).sum(axis=1)
    repy = (-strength * dy).sum(axis=1)

    # --- self-propulsion (Eq. 10.4) ---
    vnorm = np.sqrt(vx**2 + vy**2)
    vnorm_s = np.where(vnorm == 0, 1., vnorm)
    fpropx = mu * (v0 - vnorm) * vx / vnorm_s
    fpropy = mu * (v0 - vnorm) * vy / vnorm_s

    # --- random force (Eq. 10.7) ---
    frandx = ramp * np.random.uniform(-1., 1., N)
    frandy = ramp * np.random.uniform(-1., 1., N)

    return (flockx + frandx + fpropx + repx,
            flocky + frandy + fpropy + repy)


# =============================================================================
# INITIALISATION
# =============================================================================
def initialise(p):
    N = p['N']
    x  = np.zeros(2*N)
    x[:N]  = np.random.uniform(0., 1., N)   # x-positions
    x[N:]  = np.random.uniform(0., 1., N)   # y-positions
    vx = np.random.uniform(-1., 1., N)
    vy = np.random.uniform(-1., 1., N)
    return x, vx, vy


# =============================================================================
# MAIN SIMULATION LOOP
# =============================================================================
def run(overrides=None, n_frames=200, seed=None):
    """
    Run the flocking simulation.
    overrides: dict of parameter values to change from DEFAULT.
    Returns list of (px, py, vx, vy) snapshot tuples.
    """
    if seed is not None:
        np.random.seed(seed)
    p = params(overrides)
    N, n_iter, dt = p['N'], p['n_iter'], p['dt']
    frame_every = max(1, n_iter // n_frames)

    x, vx, vy = initialise(p)
    frames = []

    for i in range(n_iter):
        rb = max(p['r0'], p['rf'])
        nb, xb, yb, vxb, vyb = buffer(rb, x, vx, vy, N)
        fx, fy = force(nb, xb, yb, vxb, vyb, x, vx, vy, p)

        vx += fx * dt
        vy += fy * dt
        x[:N] = (x[:N] + vx*dt) % 1.
        x[N:] = (x[N:] + vy*dt) % 1.

        if i % frame_every == 0:
            frames.append((x[:N].copy(), x[N:].copy(), vx.copy(), vy.copy()))

    return frames


# =============================================================================
# MEASUREMENT UTILITIES
# =============================================================================
def order_parameter(vx, vy):
    """Φ = |mean unit velocity|. 1 = perfect alignment, 0 = random."""
    speeds = np.sqrt(vx**2 + vy**2)
    speeds[speeds == 0] = 1.
    return np.sqrt((vx/speeds).mean()**2 + (vy/speeds).mean()**2)

def kinetic_energy(vx, vy):
    """Total KE assuming unit mass."""
    return 0.5 * (vx**2 + vy**2).sum()

def compactness(p):
    """C = π N r₀² — ratio of agent area to domain area."""
    return np.pi * p['N'] * p['r0']**2


# =============================================================================
# ANIMATION
# =============================================================================
def animate(frames, p=None, title='Flocking simulation'):
    _p = params(p)
    n_iter = _p['n_iter']
    dt     = _p['dt']
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 1);  ax.set_ylim(0, 1);  ax.set_aspect('equal')
    ax.set_title(title)
    time_text = ax.text(0.02, 0.97, '', transform=ax.transAxes,
                        fontsize=9, va='top')
    frame_step = max(1, n_iter // len(frames))

    px0, py0, vx0, vy0 = frames[0]
    sp0 = np.sqrt(vx0**2 + vy0**2);  sp0[sp0==0] = 1.
    scat = ax.scatter(px0, py0, s=4, color='steelblue', zorder=3)
    quiv = ax.quiver(px0, py0, vx0/sp0, vy0/sp0,
                     scale=40, width=0.002, color='firebrick', zorder=4)

    def update(fi):
        px, py, vx, vy = frames[fi]
        scat.set_offsets(np.column_stack([px, py]))
        sp = np.sqrt(vx**2 + vy**2);  sp[sp==0] = 1.
        quiv.set_offsets(np.column_stack([px, py]))
        quiv.set_UVC(vx/sp, vy/sp)
        time_text.set_text(f't = {fi * frame_step * dt:.2f}')
        return scat, quiv, time_text

    ani = animation.FuncAnimation(fig, update, frames=len(frames),
                                  interval=50, blit=True)
    plt.tight_layout()
    plt.show()
    return ani


# =============================================================================
# STATIC SNAPSHOT
# =============================================================================
def plot_snapshot(frames, p=None, idx=-1, ax=None, title=None, save=None):
    _p = params(p)
    dt, n_iter = _p['dt'], _p['n_iter']
    frame_step = max(1, n_iter // len(frames))
    fi = len(frames)-1 if idx == -1 else idx
    px, py, vx, vy = frames[fi]
    sp = np.sqrt(vx**2 + vy**2);  sp[sp==0] = 1.

    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(px, py, s=4, color='steelblue')
    ax.quiver(px, py, vx/sp, vy/sp, scale=40, width=0.003, color='firebrick')
    ax.set_xlim(0, 1);  ax.set_ylim(0, 1);  ax.set_aspect('equal')
    t = fi * frame_step * dt
    ax.set_title(title or f't = {t:.1f}')
    if own:
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=120)
        plt.show()


# =============================================================================
# ENTRY POINT  (run with default parameters and animate)
# =============================================================================
if __name__ == '__main__':
    p = params()
    print('Parameters:', {k: p[k] for k in ['N','n_iter','r0','rf','alpha','v0','mu','ramp']})
    print('Simulating...')
    frames = run()
    px, py, vx, vy = frames[-1]
    print(f'Final Φ = {order_parameter(vx, vy):.3f}  '
          f'KE = {kinetic_energy(vx, vy):.1f}  '
          f'C = {compactness(p):.3f}')
    animate(frames)
