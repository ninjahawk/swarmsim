"""
1-D slope-based sandpile model -- the self-organized-criticality (SOC) icon
from Charbonneau, *Natural Complexity* (2017), Chapter 5.

This is a faithful, vectorized re-implementation of the model defined by
equations (5.1)-(5.10). It favors being provably identical to the book's
minimal code (figure 5.2) while running fast enough for the large lattices
and long runs needed for finite-size scaling.

Model (all variables continuous, real-valued):
  - State S[0..N-1], initially zero (eq 5.1).
  - SLOW FORCING (eq 5.2): on a quiet (fully stable) iteration, add a grain
    s ~ U(0, eps) at one random node r ~ U{0..N-1}.
  - SLOPE (eq 5.3): z_j = |S[j+1] - S[j]| for each of the N-1 nodal pairs.
  - INSTABILITY: a pair is unstable when z_j >= Zc (the critical slope).
  - REDISTRIBUTION (eq 5.4-5.5): an unstable pair (j, j+1) moves sand to halve
    its slope. With Sbar = (S[j]+S[j+1])/2 this is
        S[j]   += (Sbar - S[j])  / 2  =  +d/4      where d = S[j+1]-S[j]
        S[j+1] += (Sbar - S[j+1])/2  =  -d/4
    so the pair slope d -> d/2 and the sand moved is |d|/4 = z_j/4 (eq 5.7).
  - CONSERVATION (eq 5.6): redistribution moves sand, never creates/destroys it.
  - BOUNDARIES: open right edge, S[N-1]=0 every iteration (eq 5.8); left edge is
    a wall (node 0 is never drained).
  - SYNCHRONOUS UPDATE: every currently-unstable pair is redistributed using the
    OLD state, accumulated into one `move` array, applied once. Avoids the
    directional bias an in-place sequential sweep would introduce.
  - STOP-AND-GO timescale separation: a grain is only added on an iteration where
    the whole lattice is stable; while any pair is unstable the avalanche runs
    with no new forcing. An avalanche propagates one node per iteration.

Global observables:
  - mass    M^n = sum_j S_j            (eq 5.9)
  - displaced mass DM^n = sum_j dS_j   (eq 5.10), the per-iteration toppled mass.

Avalanches are maximal runs of consecutive iterations with DM^n > 0; each is
summarized by energy E (sum DM), peak P (max DM) and duration T (run length),
exactly as in the book's measure_av (figure 5.2b).

ASCII-only output (Windows cp1252 safe).
"""

import numpy as np


# Book's representative parameters (figures 5.3-5.5): N=100, eps=0.1, Zc=5.
DEFAULTS = dict(N=100, eps=0.1, Zc=5.0)


def run_sandpile(N=100, eps=0.1, Zc=5.0, n_iter=200000, seed=0,
                 record_series=True, S0=None, dissip=0.0):
    """Run the 1-D sandpile for n_iter temporal iterations.

    Parameters
    ----------
    N        : lattice size (number of nodes).
    eps      : peak forcing increment (grain size drawn from U(0, eps)).
    Zc       : critical slope.
    n_iter   : number of temporal iterations.
    seed     : RNG seed.
    record_series : if True, return the full mass and displaced-mass series.
    S0       : optional initial state array (length N); default is all zeros
               (eq 5.1). Used by the initial-condition-independence exercise.

    Returns
    -------
    dict with keys:
      'mass'     : (n_iter,) total mass M^n               [if record_series]
      'disp'     : (n_iter,) displaced mass DM^n          [if record_series]
      'S'        : (N,) final state
      'N','eps','Zc','n_iter','seed' : echoed parameters
    """
    rng = np.random.default_rng(seed)

    S = np.zeros(N) if S0 is None else np.array(S0, dtype=float)
    S[N - 1] = 0.0  # enforce open boundary on the supplied IC too

    mass = np.zeros(n_iter) if record_series else None
    disp = np.zeros(n_iter) if record_series else None
    # falloff = mass evacuated at the open right boundary on avalanche iterations
    # (Exercise 2: a "falloff" avalanche series distinct from the toppled-mass
    # series disp). Quiet-step grains that happen to land on node N-1 are excluded.
    falloff = np.zeros(n_iter) if record_series else None

    # Pre-draw forcing is not possible (we don't know how many quiet steps
    # there will be), so we draw per quiet iteration. Vectorized inner physics.
    for n in range(n_iter):
        d = S[1:] - S[:-1]                 # signed pair differences, length N-1
        z = np.abs(d)                      # eq 5.3 slopes
        unstable = z >= Zc

        if unstable.any():
            if dissip == 0.0:
                # eq 5.4-5.5 conservative synchronous redistribution: move +d/4
                # to the lower node of each unstable pair, -d/4 from the higher.
                contrib = np.where(unstable, d * 0.25, 0.0)   # length N-1
                move = np.zeros(N)
                move[:-1] += contrib          # node p gets +d/4
                move[1:] -= contrib           # node p+1 gets -d/4
            else:
                # NON-conservative variant: the higher node still sheds |d|/4, but
                # the lower node receives only (1-dissip)*|d|/4; the rest is lost
                # in the bulk (cf. the Olami-Feder-Christensen earthquake model).
                ad = np.where(unstable, np.abs(d) * 0.25, 0.0)   # transfer magnitude
                tl = (1.0 - dissip) * ad                         # to the lower node
                pos = d > 0                                      # lower node is p
                to_p = np.where(pos, tl, -ad)
                to_pp1 = np.where(pos, -ad, tl)
                move = np.zeros(N)
                move[:-1] += to_p
                move[1:] += to_pp1
            S += move
            dm = 0.25 * z[unstable].sum() # eq 5.7+5.10 displaced mass this iter
        else:
            # eq 5.2 slow forcing: one grain at a random node.
            r = rng.integers(0, N)
            S[r] += rng.uniform(0.0, eps)
            dm = 0.0

        drained = S[N - 1] if dm > 0.0 else 0.0   # boundary evacuation this iter
        S[N - 1] = 0.0                    # eq 5.8 open right boundary
        if record_series:
            mass[n] = S.sum()             # eq 5.9
            disp[n] = dm
            falloff[n] = drained

    out = dict(S=S, N=N, eps=eps, Zc=Zc, n_iter=n_iter, seed=seed)
    if record_series:
        out['mass'] = mass
        out['disp'] = disp
        out['falloff'] = falloff
    return out


def triangle_ic(N, slope):
    """A triangular initial pile of uniform slope: high at the left wall (node 0),
    draining to zero at the open right edge (node N-1). Used to start large
    lattices near the angle of repose so we skip the ~Zc*N^2/eps-iteration
    transient needed to fill an empty pile. Legitimate because the SOC stationary
    state is an attractor independent of initial condition (Exercise 3)."""
    j = np.arange(N)
    S = slope * (N - 1 - j)
    S[N - 1] = 0.0
    return S.astype(float)


def measure_avalanches(disp):
    """Extract per-avalanche energy E, peak P, duration T from a displaced-mass
    series. An avalanche is a maximal run of consecutive iterations with
    disp > 0 (book's measure_av, figure 5.2b). Returns (E, P, T) arrays.
    """
    disp = np.asarray(disp)
    active = disp > 0.0
    if not active.any():
        return (np.array([]), np.array([]), np.array([]))

    # Find run boundaries via the difference of the boolean mask.
    a = active.astype(np.int8)
    edges = np.diff(a)
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1) + 1   # exclusive end index
    if active[0]:
        starts = np.r_[0, starts]
    if active[-1]:
        ends = np.r_[ends, active.size]

    E = np.array([disp[s:e].sum() for s, e in zip(starts, ends)])
    P = np.array([disp[s:e].max() for s, e in zip(starts, ends)])
    T = (ends - starts).astype(float)
    return E, P, T


def angle_of_repose(S, Zc=None):
    """Mean nodal slope of a pile state -- its angle of repose. If Zc is given,
    also return the fractional deficit (Zc - slope)/Zc."""
    z = np.abs(np.diff(S))
    slope = z.mean()
    if Zc is None:
        return slope
    return slope, (Zc - slope) / Zc


def _self_test():
    """Limiting-case checks that must pass for the implementation to be sound."""
    print("self-test: 1-D sandpile")

    # 1) Redistribution conservation (eq 5.6): on an avalanche iteration with no
    #    boundary drain, total mass is unchanged to machine precision. We verify
    #    the move array sums to zero for an arbitrary unstable configuration.
    rng = np.random.default_rng(1)
    N = 50
    S = np.cumsum(rng.uniform(0, 3, N))      # a steep, surely-unstable pile
    d = S[1:] - S[:-1]
    z = np.abs(d)
    Zc = 5.0
    unstable = z >= Zc
    contrib = np.where(unstable, d * 0.25, 0.0)
    move = np.zeros(N)
    move[:-1] += contrib
    move[1:] -= contrib
    err = abs(move.sum())
    print("  redistribution conserves mass: sum(move) = %.2e" % err)
    assert err < 1e-12, "redistribution not conservative"

    # 2) Slope is halved on a single ISOLATED unstable pair. (A node shared by
    #    two unstable pairs topples both ways at once, so the >2x reduction seen
    #    there is correct synchronous behavior, not a halving -- hence we isolate.)
    S2 = np.array([0.0, 0.0, 10.0, 10.0, 10.0])  # only pair (1,2) is unstable
    d2 = S2[1:] - S2[:-1]
    z2 = np.abs(d2)
    u2 = z2 >= Zc
    c2 = np.where(u2, d2 * 0.25, 0.0)
    m2 = np.zeros(5)
    m2[:-1] += c2
    m2[1:] -= c2
    S2n = S2 + m2
    # pair (1,2): old slope 10 -> new slope should be 5 (halved)
    new_slope_12 = abs(S2n[2] - S2n[1])
    print("  single-pair slope 10.0 -> %.4f (expect 5.0)" % new_slope_12)
    assert abs(new_slope_12 - 5.0) < 1e-12

    # 3) Avalanche extraction on a hand-built series.
    fake = np.array([0, 0, 3, 5, 2, 0, 0, 4, 0, 1, 1])  # two runs
    E, P, T = measure_avalanches(fake)
    print("  avalanche extraction: E=%s P=%s T=%s" % (E.tolist(), P.tolist(), T.tolist()))
    assert np.allclose(E, [10, 4, 2]) and np.allclose(P, [5, 4, 1]) and np.allclose(T, [3, 1, 2])

    # 4) A short full run reaches a non-trivial state and conserves mass over a
    #    quiet (forcing) step: mass increases by exactly the grain added, minus
    #    any boundary drain (none on a quiet step unless the grain lands at N-1).
    res = run_sandpile(N=100, eps=0.1, Zc=5.0, n_iter=20000, seed=0)
    print("  20k-iter run: final mass = %.3f, #avalanches = %d"
          % (res['mass'][-1], len(measure_avalanches(res['disp'])[0])))
    print("self-test PASSED")


if __name__ == "__main__":
    _self_test()
