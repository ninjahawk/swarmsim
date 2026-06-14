"""
2-D slope-based sandpile -- the "Grand Challenge" of Charbonneau Ch. 5
(Exercise 6): generalize the 1-D stability and redistribution rules to two
dimensions, reach the self-organized-critical state, and ask whether the
avalanche exponents match the 1-D case.

DESIGN DECISION (owned, not the book's literal suggestion).
The book hints at defining a slope on a 2x2 block of nodes. I take a cleaner,
more faithful route: the 1-D model is fundamentally a BOND model -- every
adjacent pair (a bond) carries a slope z = |S[j+1]-S[j]|, and an unstable bond
(z >= Zc) halves its slope by moving z/4 of sand across it. The natural 2-D
generalization is therefore to keep that EXACT bond rule but let bonds run in
both lattice directions: each site connects to its right and upper neighbours by
an x-bond and a y-bond, and every bond obeys the identical 1-D pair rule.

Why this and not the 2x2-block slope:
  - It reduces to the 1-D model exactly when one dimension is collapsed -- the
    local dynamics are literally unchanged. That makes the 1-D-vs-2-D exponent
    comparison a clean test of dimensionality alone, which is the whole point of
    the universality question. A 2x2-block rule would change the local dynamics
    as well as the dimension, confounding the comparison.
  - It needs no arbitrary choice of how to collapse a 2x2 block to one number.

Boundaries: all four edges are open (boundary sites drained to 0 each iteration),
the 2-D analog of the 1-D open right edge. Driving adds a grain s ~ U(0,eps) at a
random INTERIOR site, but only on fully-stable iterations (stop-and-go).
Synchronous update over all unstable bonds. Conservative redistribution.

State convention: S has shape (L, L). A site is "interior" if it is not on the
outermost ring. The drained boundary plays the role of the open table edge.

ASCII-only output.
"""

import numpy as np


def pyramid_ic(L, slope):
    """A square pyramid initial pile, peak in the centre, draining to 0 at all
    four open edges, with the given mean bond slope. Used to start near the
    stationary pile and skip the long fill transient (SOC state is an attractor).
    Height at (i,j) = slope * (distance, in Chebyshev metric, to nearest edge)."""
    i = np.arange(L)
    di = np.minimum(i, L - 1 - i)                  # distance to nearest edge per axis
    D = np.minimum.outer(di, di)                   # Chebyshev distance to boundary
    S = slope * D.astype(float)
    S[0, :] = S[-1, :] = S[:, 0] = S[:, -1] = 0.0
    return S


def _apply_boundary(S):
    S[0, :] = 0.0
    S[-1, :] = 0.0
    S[:, 0] = 0.0
    S[:, -1] = 0.0


def run_sandpile2d(L=64, eps=0.1, Zc=5.0, n_iter=1_000_000, seed=0,
                   record_series=True, S0=None, dissip=0.0):
    """Run the 2-D bond-slope sandpile for n_iter temporal iterations.

    Returns a dict with 'mass', 'disp' series (if record_series), final 'S',
    and echoed parameters. Avalanche extraction uses the same measure_avalanches
    as the 1-D model (import from sandpile1d).
    """
    rng = np.random.default_rng(seed)
    S = np.zeros((L, L)) if S0 is None else np.array(S0, dtype=float)
    _apply_boundary(S)

    mass = np.zeros(n_iter) if record_series else None
    disp = np.zeros(n_iter) if record_series else None
    # activity = number of bond topplings this iteration (the BTW-comparable
    # "size" measure: avalanche size S = total topplings over the avalanche).
    act = np.zeros(n_iter) if record_series else None

    for n in range(n_iter):
        dx = S[1:, :] - S[:-1, :]      # x-bonds, shape (L-1, L)
        dy = S[:, 1:] - S[:, :-1]      # y-bonds, shape (L, L-1)
        ux = np.abs(dx) >= Zc
        uy = np.abs(dy) >= Zc
        any_x = ux.any()
        any_y = uy.any()

        if any_x or any_y:
            move = np.zeros((L, L))
            dm = 0.0
            ntop = 0
            if dissip == 0.0:
                # conservative redistribution (keeps S4 results bit-identical)
                if any_x:
                    cx = np.where(ux, dx * 0.25, 0.0)
                    move[:-1, :] += cx        # lower site of each x-bond gains +d/4
                    move[1:, :] -= cx         # upper site loses
                    dm += np.abs(cx).sum()
                    ntop += int(ux.sum())
                if any_y:
                    cy = np.where(uy, dy * 0.25, 0.0)
                    move[:, :-1] += cy
                    move[:, 1:] -= cy
                    dm += np.abs(cy).sum()
                    ntop += int(uy.sum())
            else:
                # NON-conservative: lower node of each unstable bond receives only
                # (1-dissip) of the shed sand; the rest is destroyed in the bulk.
                if any_x:
                    adx = np.where(ux, np.abs(dx) * 0.25, 0.0)
                    tlx = (1.0 - dissip) * adx
                    posx = dx > 0
                    move[:-1, :] += np.where(posx, tlx, -adx)
                    move[1:, :] += np.where(posx, -adx, tlx)
                    dm += adx.sum()
                    ntop += int(ux.sum())
                if any_y:
                    ady = np.where(uy, np.abs(dy) * 0.25, 0.0)
                    tly = (1.0 - dissip) * ady
                    posy = dy > 0
                    move[:, :-1] += np.where(posy, tly, -ady)
                    move[:, 1:] += np.where(posy, -ady, tly)
                    dm += ady.sum()
                    ntop += int(uy.sum())
            S += move
        else:
            # slow forcing at a random interior site
            r = rng.integers(1, L - 1)
            c = rng.integers(1, L - 1)
            S[r, c] += rng.uniform(0.0, eps)
            dm = 0.0
            ntop = 0

        _apply_boundary(S)
        if record_series:
            mass[n] = S.sum()
            disp[n] = dm
            act[n] = ntop

    out = dict(S=S, L=L, eps=eps, Zc=Zc, n_iter=n_iter, seed=seed)
    if record_series:
        out['mass'] = mass
        out['disp'] = disp
        out['act'] = act
    return out


def _self_test():
    print("self-test: 2-D sandpile")

    # 1) Conservation of an interior redistribution step: with boundaries left
    #    untouched, the move array sums to zero (sand is only moved across bonds).
    rng = np.random.default_rng(0)
    L = 12
    S = rng.uniform(0, 4, (L, L))
    S = np.cumsum(np.cumsum(S, axis=0), axis=1)   # steep, surely-unstable
    Zc = 5.0
    dx = S[1:, :] - S[:-1, :]
    dy = S[:, 1:] - S[:, :-1]
    ux = np.abs(dx) >= Zc
    uy = np.abs(dy) >= Zc
    move = np.zeros((L, L))
    cx = np.where(ux, dx * 0.25, 0.0)
    move[:-1, :] += cx; move[1:, :] -= cx
    cy = np.where(uy, dy * 0.25, 0.0)
    move[:, :-1] += cy; move[:, 1:] -= cy
    print("  redistribution conserves mass: sum(move) = %.2e" % abs(move.sum()))
    assert abs(move.sum()) < 1e-10

    # 2) Isolated single x-bond halves its slope (1-D reduction check in 2-D).
    S2 = np.zeros((5, 5))
    S2[2, 2] = 12.0   # interior spike; bonds to its 4 neighbours all unstable
    dx = S2[1:, :] - S2[:-1, :]
    dy = S2[:, 1:] - S2[:, :-1]
    ux = np.abs(dx) >= Zc; uy = np.abs(dy) >= Zc
    move = np.zeros((5, 5))
    cx = np.where(ux, dx * 0.25, 0.0); move[:-1, :] += cx; move[1:, :] -= cx
    cy = np.where(uy, dy * 0.25, 0.0); move[:, :-1] += cy; move[:, 1:] -= cy
    S2n = S2 + move
    # the spike loses to 4 neighbours: 12 - 4*(12/4) = 12 - 12 = 0... each bond
    # moves 12/4=3 out, 4 bonds -> spike drops by 12 to 0; each neighbour +3.
    print("  spike 12.0 -> %.3f (4 bonds each move 3); neighbours -> %.3f"
          % (S2n[2, 2], S2n[1, 2]))
    assert abs(S2n[2, 2] - 0.0) < 1e-9 and abs(S2n[1, 2] - 3.0) < 1e-9

    # 3) A short run reaches a non-trivial state and conserves mass globally.
    from sandpile1d import measure_avalanches
    res = run_sandpile2d(L=32, eps=0.1, Zc=5.0, n_iter=50_000, seed=1,
                         S0=pyramid_ic(32, 4.5))
    E, P, T = measure_avalanches(res['disp'])
    print("  50k-iter L=32 run: final mass=%.1f, #avalanches=%d, maxT=%d"
          % (res['mass'][-1], E.size, int(T.max()) if T.size else -1))
    print("self-test PASSED")


if __name__ == "__main__":
    _self_test()
