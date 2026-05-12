"""
model.py -- Object-oriented flocking model foundation.

Classes
-------
Flock    : N prey agents on a periodic 2D unit square.
Predator : A single predator with configurable strategy.

Typical usage
-------------
    flock = Flock(N=350, seed=42)
    preds = [Predator(strategy='encircle', angle=k*360/6)
             for k in range(6)]
    for _ in range(2000):
        flock.evolve(predators=preds)
    print(flock.phi)

All existing scripts (flocking.py, predator.py, ...) remain intact and
importable.  New experiments should import from this module instead.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Module-level defaults (match book Table 10.1 and prior experiments)
# ---------------------------------------------------------------------------
PREY_DEFAULTS = dict(
    N      = 350,
    dt     = 0.01,
    r0     = 0.005,
    eps    = 0.1,
    rf     = 0.1,
    alpha  = 1.0,
    v0     = 1.0,
    mu     = 10.0,
    ramp   = 0.5,
)

PRED_DEFAULTS = dict(
    v0     = 0.05,
    mu     = 10.0,
    alpha  = 5.0,
    rf     = 0.15,
    r0     = 0.1,
    eps    = 2.0,
    ramp   = 1.0,
)


# ---------------------------------------------------------------------------
# Helpers (mirror flocking.py internals; kept private here)
# ---------------------------------------------------------------------------

def _periodic_com(pos):
    """Center of mass on a periodic [0,1] domain via the angle-mean trick."""
    angles = 2.0 * np.pi * pos
    cx = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean()) / (2 * np.pi)
    return cx % 1.0


def _periodic_disp(target, source):
    """Shortest signed displacement target - source on [0,1] torus."""
    d = target - source
    d -= np.round(d)
    return d


def _buffer(rb, px, py, vx, vy, N):
    """Ghost-agent buffer for periodic force calculation."""
    xb  = np.zeros(4 * N);  yb  = np.zeros(4 * N)
    vxb = np.zeros(4 * N);  vyb = np.zeros(4 * N)
    xb[:N] = px;   yb[:N] = py
    vxb[:N] = vx;  vyb[:N] = vy
    nb = N

    for k in range(N):
        xk, yk = px[k], py[k]
        if xk <= rb:
            xb[nb]=xk+1.; yb[nb]=yk; vxb[nb]=vx[k]; vyb[nb]=vy[k]; nb+=1
        if xk >= 1.-rb:
            xb[nb]=xk-1.; yb[nb]=yk; vxb[nb]=vx[k]; vyb[nb]=vy[k]; nb+=1
        if yk <= rb:
            xb[nb]=xk; yb[nb]=yk+1.; vxb[nb]=vx[k]; vyb[nb]=vy[k]; nb+=1
        if yk >= 1.-rb:
            xb[nb]=xk; yb[nb]=yk-1.; vxb[nb]=vx[k]; vyb[nb]=vy[k]; nb+=1
        if xk <= rb and yk <= rb:
            xb[nb]=xk+1.; yb[nb]=yk+1.; vxb[nb]=vx[k]; vyb[nb]=vy[k]; nb+=1
        if xk >= 1.-rb and yk <= rb:
            xb[nb]=xk-1.; yb[nb]=yk+1.; vxb[nb]=vx[k]; vyb[nb]=vy[k]; nb+=1
        if xk <= rb and yk >= 1.-rb:
            xb[nb]=xk+1.; yb[nb]=yk-1.; vxb[nb]=vx[k]; vyb[nb]=vy[k]; nb+=1
        if xk >= 1.-rb and yk >= 1.-rb:
            xb[nb]=xk-1.; yb[nb]=yk-1.; vxb[nb]=vx[k]; vyb[nb]=vy[k]; nb+=1

    return nb, xb[:nb], yb[:nb], vxb[:nb], vyb[:nb]


def _prey_forces(px, py, vx, vy, N, r0, eps, rf, alpha, v0, mu, ramp,
                 nb, xb, yb, vxb, vyb):
    """Vectorised force calculation for prey agents (identical to flocking.py)."""
    dx = xb[np.newaxis, :] - px[:, np.newaxis]
    dy = yb[np.newaxis, :] - py[:, np.newaxis]
    d2 = dx**2 + dy**2

    not_self = np.ones((N, nb), dtype=bool)
    idx = np.arange(min(N, nb))
    not_self[idx, idx] = False

    flock_mask = (d2 <= rf**2) & not_self
    flockx = np.where(flock_mask, vxb, 0.).sum(axis=1)
    flocky = np.where(flock_mask, vyb, 0.).sum(axis=1)
    nflock = flock_mask.sum(axis=1)
    norm_f = np.sqrt(flockx**2 + flocky**2)
    norm_f[nflock == 0] = 1.
    flockx = alpha * flockx / norm_f
    flocky = alpha * flocky / norm_f

    rep_mask = (d2 <= (2*r0)**2) & not_self & (d2 > 0)
    d_safe   = np.where(rep_mask, np.sqrt(d2), 1.)
    base     = np.where(rep_mask, 1. - d_safe/(2.*r0), 0.)
    strength = np.where(rep_mask, eps * base**1.5 / d_safe, 0.)
    repx = (-strength * dx).sum(axis=1)
    repy = (-strength * dy).sum(axis=1)

    vnorm   = np.sqrt(vx**2 + vy**2)
    vnorm_s = np.where(vnorm == 0, 1., vnorm)
    propx = mu * (v0 - vnorm) * vx / vnorm_s
    propy = mu * (v0 - vnorm) * vy / vnorm_s

    randx = ramp * np.random.uniform(-1., 1., N)
    randy = ramp * np.random.uniform(-1., 1., N)

    return flockx + repx + propx + randx, flocky + repy + propy + randy


# ===========================================================================
# Flock class
# ===========================================================================

class Flock:
    """
    N prey agents on a periodic 2D unit square [0,1]^2.

    Parameters (all keyword; defaults match PREY_DEFAULTS)
    ----------
    N, dt, r0, eps, rf, alpha, v0, mu, ramp : floats/ints
    seed : optional int -- RNG seed for reproducibility
    """

    def __init__(self, seed=None, **kwargs):
        p = {**PREY_DEFAULTS, **kwargs}
        self.N     = int(p['N'])
        self.dt    = p['dt']
        self.r0    = p['r0']
        self.eps   = p['eps']
        self.rf    = p['rf']
        self.alpha = p['alpha']
        self.v0    = p['v0']
        self.mu    = p['mu']
        self.ramp  = p['ramp']
        self.t     = 0

        if seed is not None:
            np.random.seed(seed)

        N = self.N
        self.px = np.random.uniform(0., 1., N)
        self.py = np.random.uniform(0., 1., N)
        self.vx = np.random.uniform(-1., 1., N)
        self.vy = np.random.uniform(-1., 1., N)

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def evolve(self, predators=None, n_steps=1):
        """
        Advance the flock (and all predators) by n_steps timesteps.

        predators : list of Predator objects, or None
        """
        for _ in range(n_steps):
            self._step(predators or [])
            self.t += self.dt

    def _step(self, predators):
        rb = max(self.r0, self.rf)
        nb, xb, yb, vxb, vyb = _buffer(rb, self.px, self.py,
                                         self.vx, self.vy, self.N)

        fx, fy = _prey_forces(
            self.px, self.py, self.vx, self.vy, self.N,
            self.r0, self.eps, self.rf, self.alpha, self.v0, self.mu, self.ramp,
            nb, xb, yb, vxb, vyb,
        )

        for pred in predators:
            pfx, pfy = pred.force_on_prey(self.px, self.py)
            fx += pfx
            fy += pfy

        self.vx += fx * self.dt
        self.vy += fy * self.dt
        self.px = (self.px + self.vx * self.dt) % 1.
        self.py = (self.py + self.vy * self.dt) % 1.

        for pred in predators:
            pred.update(self, predators, self.dt)

    # ------------------------------------------------------------------
    # Measurements
    # ------------------------------------------------------------------

    @property
    def phi(self):
        """Order parameter Phi = |mean unit velocity|. 1 = aligned, 0 = random."""
        speeds = np.sqrt(self.vx**2 + self.vy**2)
        speeds[speeds == 0] = 1.
        return float(np.sqrt((self.vx/speeds).mean()**2 + (self.vy/speeds).mean()**2))

    @property
    def com(self):
        """Periodic center of mass (cx, cy)."""
        return _periodic_com(self.px), _periodic_com(self.py)

    @property
    def mean_speed(self):
        return float(np.sqrt(self.vx**2 + self.vy**2).mean())

    @property
    def kinetic_energy(self):
        return float(0.5 * (self.vx**2 + self.vy**2).sum())

    def snapshot(self):
        """Return (px, py, vx, vy) as copies of current state."""
        return (self.px.copy(), self.py.copy(),
                self.vx.copy(), self.vy.copy())


# ===========================================================================
# Predator class
# ===========================================================================

class Predator:
    """
    A single predator with configurable targeting strategy.

    Strategy options
    ----------------
    'naive'     : chase flock center of mass directly
    'encircle'  : target a point offset from CoM by enc_radius at fixed angle

    Coordination
    ------------
    coord_alpha > 0 adds predator-predator repulsion during update().
    Encirclement predators already spread by design; coord_alpha is most
    useful with strategy='naive'.

    Parameters
    ----------
    x, y       : initial position (random if None)
    v0         : target speed
    mu         : self-propulsion amplitude
    alpha      : drive force toward target
    rf         : perception radius (unused for now; reserved for sensing)
    r0         : repulsion radius on prey
    eps        : repulsion amplitude on prey
    ramp       : random noise on predator motion
    strategy   : 'naive' or 'encircle'
    angle      : fixed bearing (degrees) for encirclement target; ignored for naive
    enc_radius : offset distance from CoM for encirclement target
    coord_alpha: predator-predator repulsion strength (0 = none)
    """

    def __init__(self, x=None, y=None, **kwargs):
        p = {**PRED_DEFAULTS, **kwargs}
        self.v0          = p['v0']
        self.mu          = p['mu']
        self.alpha       = p['alpha']
        self.rf          = p['rf']
        self.r0          = p['r0']
        self.eps         = p['eps']
        self.ramp        = p['ramp']
        self.strategy    = kwargs.get('strategy', 'naive')
        self.angle       = np.radians(kwargs.get('angle', 0.0))
        self.enc_radius  = kwargs.get('enc_radius', 0.15)
        self.coord_alpha = kwargs.get('coord_alpha', 0.0)

        self.x  = float(x) if x is not None else np.random.uniform(0., 1.)
        self.y  = float(y) if y is not None else np.random.uniform(0., 1.)
        self.vx = np.random.uniform(-1., 1.) * self.v0
        self.vy = np.random.uniform(-1., 1.) * self.v0

    # ------------------------------------------------------------------
    # Force on prey
    # ------------------------------------------------------------------

    def force_on_prey(self, px, py):
        """Return (fx, fy) arrays: repulsive force this predator exerts on each prey."""
        N = len(px)
        fx = np.zeros(N)
        fy = np.zeros(N)
        for j in range(N):
            ddx = _periodic_disp(self.x, px[j])
            ddy = _periodic_disp(self.y, py[j])
            d = np.sqrt(ddx**2 + ddy**2)
            if 0 < d <= self.r0:
                strength = self.eps * (1. - d/self.r0)**1.5 / d
                fx[j] -= strength * ddx
                fy[j] -= strength * ddy
        return fx, fy

    # ------------------------------------------------------------------
    # Predator motion update
    # ------------------------------------------------------------------

    def update(self, flock, all_predators, dt):
        """Advance predator velocity and position by one timestep."""
        cx, cy = flock.com

        if self.strategy == 'encircle':
            tx = (cx + self.enc_radius * np.cos(self.angle)) % 1.
            ty = (cy + self.enc_radius * np.sin(self.angle)) % 1.
        else:
            tx, ty = cx, cy

        # drive toward target
        dx = _periodic_disp(tx, self.x)
        dy = _periodic_disp(ty, self.y)
        dist = np.sqrt(dx**2 + dy**2)
        if dist > 0:
            dx /= dist;  dy /= dist

        speed = np.sqrt(self.vx**2 + self.vy**2)
        if speed > 0:
            pfx = self.alpha * dx + self.mu * (self.v0 - speed) * self.vx / speed
            pfy = self.alpha * dy + self.mu * (self.v0 - speed) * self.vy / speed
        else:
            pfx = self.alpha * dx
            pfy = self.alpha * dy

        # predator-predator repulsion
        if self.coord_alpha > 0:
            for other in all_predators:
                if other is self:
                    continue
                ox = _periodic_disp(self.x, other.x)
                oy = _periodic_disp(self.y, other.y)
                od = np.sqrt(ox**2 + oy**2)
                if 0 < od <= 0.5:
                    strength = self.coord_alpha / (od + 1e-6)
                    pfx += strength * ox / od
                    pfy += strength * oy / od

        pfx += self.ramp * np.random.uniform(-1., 1.)
        pfy += self.ramp * np.random.uniform(-1., 1.)

        self.vx += pfx * dt
        self.vy += pfy * dt
        self.x = (self.x + self.vx * dt) % 1.
        self.y = (self.y + self.vy * dt) % 1.

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def pos(self):
        return self.x, self.y

    def dist_to_flock(self, flock):
        """Mean distance from this predator to all prey agents."""
        ddx = _periodic_disp(self.x, flock.px)
        ddy = _periodic_disp(self.y, flock.py)
        return float(np.sqrt(ddx**2 + ddy**2).min())


# ===========================================================================
# Convenience: run a fixed simulation and return metrics
# ===========================================================================

def simulate(n_iter=2000, n_warmup=500, predators=None, record_every=10,
             seed=None, **flock_kwargs):
    """
    Run one simulation and return a dict of recorded timeseries.

    Parameters
    ----------
    n_iter      : total timesteps
    n_warmup    : steps before recording begins (let flock settle)
    predators   : list of Predator objects (or None)
    record_every: record metrics every this many steps
    seed        : RNG seed
    flock_kwargs: keyword args forwarded to Flock()

    Returns dict with keys: 'phi', 't', and optionally 'pred_dist'.
    """
    flock = Flock(seed=seed, **flock_kwargs)
    preds = predators or []

    phi_ts   = []
    dist_ts  = []
    t_ts     = []

    for i in range(n_iter):
        flock.evolve(predators=preds)
        if i >= n_warmup and i % record_every == 0:
            phi_ts.append(flock.phi)
            t_ts.append(flock.t)
            if preds:
                dists = [p.dist_to_flock(flock) for p in preds]
                dist_ts.append(min(dists))

    result = {'phi': np.array(phi_ts), 't': np.array(t_ts)}
    if dist_ts:
        result['pred_dist'] = np.array(dist_ts)
    return result


if __name__ == '__main__':
    print('Smoke test: plain flock, 500 steps...')
    flock = Flock(N=100, seed=0)
    for _ in range(500):
        flock.evolve()
    print(f'  phi = {flock.phi:.3f}  (expect > 0.9 for default params)')

    print('Smoke test: 3 encircling predators, 500 steps...')
    flock2 = Flock(N=100, seed=1)
    preds  = [Predator(strategy='encircle', angle=k*120, enc_radius=0.15)
              for k in range(3)]
    for _ in range(500):
        flock2.evolve(predators=preds)
    print(f'  phi = {flock2.phi:.3f}')
    print('Done.')
