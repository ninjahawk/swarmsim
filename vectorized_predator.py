"""
vectorized_predator.py -- vectorized predator->prey repulsion force.

Optional performance helper. The proposed co-adaptation / evolution thread
(heritable prey escape weight across the F70 "dangerous valley") needs many
generations x many steps, and model.Predator.force_on_prey uses a per-prey
Python loop that is the bottleneck noted for that thread. This module evaluates
all prey -- and all predators at once -- in vectorized NumPy.

It is ADDITIVE: it does not modify or monkey-patch model.py, has no import-time
side effects, and is __main__-guarded. The __main__ self-test asserts that it
reproduces model.Predator.force_on_prey to within floating-point round-off, so
it is a drop-in replacement only where the caller chooses to use it.

Force convention (matches model.py and the PREDATOR FORCE SIGN rule in CLAUDE.md):
    f_on_prey = +strength * (prey - pred)   (repulsion: pushes prey away)
    strength  = eps * (1 - d/r0)**1.5 / d   for 0 < d <= r0, else 0
where d is the shortest-image distance on the [0,1]^2 torus. The masked
computation avoids the negative**1.5 RuntimeWarning, matching model.py's idiom.
"""

import numpy as np


def predator_force(pred_xy, prey_x, prey_y, r0, eps):
    """
    Repulsive force on each prey from one or more predators, fully vectorized.

    Parameters
    ----------
    pred_xy : array of predator positions. Shape (P, 2) for P predators, or
              (2,) for a single predator.
    prey_x, prey_y : (N,) arrays of prey positions on the unit torus.
    r0, eps : repulsion radius and amplitude (as in model.Predator).

    Returns
    -------
    fx, fy : (N,) arrays -- the force from all predators summed on each prey.
    """
    pred_xy = np.atleast_2d(np.asarray(pred_xy, dtype=float))   # (P, 2)
    prey_x = np.asarray(prey_x, dtype=float)
    prey_y = np.asarray(prey_y, dtype=float)

    px = pred_xy[:, 0][:, None]                  # (P, 1)
    py = pred_xy[:, 1][:, None]

    # shortest signed displacement pred - prey on the torus, shape (P, N)
    ddx = px - prey_x[None, :]
    ddx -= np.round(ddx)
    ddy = py - prey_y[None, :]
    ddy -= np.round(ddy)

    d = np.sqrt(ddx * ddx + ddy * ddy)           # (P, N)
    mask = (d > 0.0) & (d <= r0)

    strength = np.zeros_like(d)
    dm = d[mask]
    strength[mask] = eps * (1.0 - dm / r0) ** 1.5 / dm

    # -strength*(pred - prey) == +strength*(prey - pred): repulsion, sum over predators
    fx = -(strength * ddx).sum(axis=0)           # (N,)
    fy = -(strength * ddy).sum(axis=0)
    return fx, fy


def _self_test():
    """Verify equivalence to model.Predator.force_on_prey and report speedup."""
    import time
    import model

    rng = np.random.default_rng(0)
    N, P = 350, 6
    R0, EPS = 0.05, 0.1

    max_err = 0.0
    for _ in range(20):
        prey_x = rng.uniform(0., 1., N)
        prey_y = rng.uniform(0., 1., N)
        preds = [model.Predator(x=rng.uniform(0., 1.), y=rng.uniform(0., 1.),
                                r0=R0, eps=EPS) for _ in range(P)]

        ref_fx = np.zeros(N)
        ref_fy = np.zeros(N)
        for pr in preds:
            a, b = pr.force_on_prey(prey_x, prey_y)
            ref_fx += a
            ref_fy += b

        pred_xy = np.array([[pr.x, pr.y] for pr in preds])
        vx, vy = predator_force(pred_xy, prey_x, prey_y, R0, EPS)
        err = max(np.abs(vx - ref_fx).max(), np.abs(vy - ref_fy).max())
        max_err = max(max_err, err)

    print("max abs error vs model.Predator.force_on_prey (20 trials): %.2e" % max_err)
    assert max_err < 1e-10, "vectorized force does not match the reference!"
    print("PASS: vectorized predator force matches model.py to round-off.")

    # speed comparison on a representative configuration
    prey_x = rng.uniform(0., 1., N)
    prey_y = rng.uniform(0., 1., N)
    preds = [model.Predator(x=rng.uniform(0., 1.), y=rng.uniform(0., 1.),
                            r0=R0, eps=EPS) for _ in range(P)]
    pred_xy = np.array([[pr.x, pr.y] for pr in preds])

    steps = 2000
    t0 = time.perf_counter()
    for _ in range(steps):
        fx = np.zeros(N)
        fy = np.zeros(N)
        for pr in preds:
            a, b = pr.force_on_prey(prey_x, prey_y)
            fx += a
            fy += b
    t_loop = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(steps):
        predator_force(pred_xy, prey_x, prey_y, R0, EPS)
    t_vec = time.perf_counter() - t0

    print("%d steps, P=%d N=%d: loop %.3fs, vectorized %.3fs (%.1fx faster)"
          % (steps, P, N, t_loop, t_vec, t_loop / max(t_vec, 1e-9)))


if __name__ == "__main__":
    _self_test()
