"""
Fast active-list 1-D slope sandpile -- Charbonneau Ch. 5, Exercise 5.

The reference engine in sandpile1d.py rescans all N-1 nodal pairs on every
temporal iteration, so a run costs O(N) per iteration regardless of how little of
the lattice is actually moving. That is fine for the N <= 1024 used in S1-S8 but
it caps how far the finite-size scaling can reach, which is what leaves the S6
duration self-test inconclusive and the S7 2-D saturation incomplete.

Exercise 5 asks for the standard cure: keep an explicit list of the sites that are
currently active (part of an unstable pair) and touch only those, plus their
immediate neighbours, each step. An avalanche then costs O(its own activity) and a
quiet loading step costs O(1), so the whole run scales with the total number of
topplings rather than N x iterations. The inner loop is JIT-compiled with numba.

This file does NOT redefine the model -- it reproduces sandpile1d.run_sandpile
exactly. The synchronous update rule, the open right / walled left boundaries, the
stop-and-go forcing, and the optional bulk dissipation are all identical. The only
thing that changes is the bookkeeping that decides which pairs to examine. The
self-test drives this engine and the reference engine with one shared forcing
stream and checks they produce the same avalanche series to machine precision.

ASCII-only output (Windows cp1252 safe).
"""

import os
import sys
import numpy as np
from numba import njit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sandpile1d import run_sandpile, measure_avalanches, triangle_ic, DEFAULTS  # noqa: E402
from sandpile2d import run_sandpile2d, pyramid_ic  # noqa: E402


@njit(cache=True)
def _run_core(S, eps, Zc, n_iter, dissip, seed,
              use_forcing, f_nodes, f_sizes,
              record, mass_out, disp_out, fall_out):
    """JIT inner loop. See run_sandpile_fast for the public wrapper.

    Active-list bookkeeping (pairs are indexed 0..N-2; pair j joins nodes j, j+1):
      - `cur` holds the pair indices that are unstable at the start of the current
        iteration. If it is empty the step is a quiet loading step, else it is one
        synchronous avalanche step in which every pair in `cur` topples at once.
      - after toppling, only the pairs adjacent to a node that just moved can have
        changed stability, so just those are re-examined to build the next list.
    This makes each step cost O(active pairs) instead of O(N).
    """
    N = S.shape[0]
    np.random.seed(seed)

    # Persistent scratch (allocated once, kept zero/false outside their active use
    # so they never need a full O(N) clear inside the loop).
    move = np.zeros(N)               # node deltas for the current synchronous step
    touched = np.empty(N, np.int64)  # nodes written into `move` this step
    is_touched = np.zeros(N, np.uint8)
    cur = np.empty(N, np.int64)      # currently-unstable pair indices
    nxt = np.empty(N, np.int64)      # candidate pairs for the next step
    in_nxt = np.zeros(N, np.uint8)   # dedup flag while building `nxt`
    ncur = 0

    mass = S.sum()                   # maintained incrementally from the IC mass
    fi = 0                           # forcing-stream cursor

    for n in range(n_iter):
        if ncur == 0:
            # ---- quiet loading step (eq 5.2): one grain, then check locally ----
            if use_forcing:
                r = f_nodes[fi]
                g = f_sizes[fi]
                fi += 1
            else:
                r = np.random.randint(0, N)
                g = np.random.random() * eps
            S[r] += g
            mass += g
            dm = 0.0
            drained = 0.0
            # open right boundary every iteration (eq 5.8)
            mass -= S[N - 1]
            S[N - 1] = 0.0
            # a grain (or the boundary reset) only changes pairs around r and N-2
            ncnt = 0
            for k in (r - 1, r, N - 2):
                if 0 <= k <= N - 2 and in_nxt[k] == 0:
                    d = S[k + 1] - S[k]
                    z = d if d >= 0.0 else -d
                    if z >= Zc:
                        nxt[ncnt] = k
                        in_nxt[k] = 1
                        ncnt += 1
            # move nxt -> cur, clear flags
            for i in range(ncnt):
                cur[i] = nxt[i]
                in_nxt[nxt[i]] = 0
            ncur = ncnt
        else:
            # ---- avalanche step: topple every currently-unstable pair at once ----
            nt = 0
            dm = 0.0
            for i in range(ncur):
                j = cur[i]
                d = S[j + 1] - S[j]
                z = d if d >= 0.0 else -d
                # cur is built to hold only unstable pairs, but guard anyway.
                if z < Zc:
                    continue
                dm += 0.25 * z
                if dissip == 0.0:
                    c = d * 0.25
                    if is_touched[j] == 0:
                        is_touched[j] = 1
                        touched[nt] = j
                        nt += 1
                    move[j] += c
                    if is_touched[j + 1] == 0:
                        is_touched[j + 1] = 1
                        touched[nt] = j + 1
                        nt += 1
                    move[j + 1] -= c
                else:
                    ad = z * 0.25                     # |transfer| shed by high node
                    tl = (1.0 - dissip) * ad          # received by the low node
                    if d > 0.0:                        # low node is j
                        to_j, to_j1 = tl, -ad
                    else:
                        to_j, to_j1 = -ad, tl
                    if is_touched[j] == 0:
                        is_touched[j] = 1
                        touched[nt] = j
                        nt += 1
                    move[j] += to_j
                    if is_touched[j + 1] == 0:
                        is_touched[j + 1] = 1
                        touched[nt] = j + 1
                        nt += 1
                    move[j + 1] += to_j1

            # apply the accumulated synchronous move, build next candidate list
            ncnt = 0
            for i in range(nt):
                node = touched[i]
                S[node] += move[node]
                mass += move[node]
                move[node] = 0.0
                is_touched[node] = 0
                # pairs adjacent to a changed node may have flipped stability
                for k in (node - 1, node):
                    if 0 <= k <= N - 2 and in_nxt[k] == 0:
                        d = S[k + 1] - S[k]
                        z = d if d >= 0.0 else -d
                        if z >= Zc:
                            nxt[ncnt] = k
                            in_nxt[k] = 1
                            ncnt += 1

            # open right boundary (eq 5.8); node N-1 changed -> recheck pair N-2
            drained = S[N - 1]
            mass -= drained
            S[N - 1] = 0.0
            k = N - 2
            if k >= 0 and in_nxt[k] == 0:
                d = S[k + 1] - S[k]
                z = d if d >= 0.0 else -d
                if z >= Zc:
                    nxt[ncnt] = k
                    in_nxt[k] = 1
                    ncnt += 1

            for i in range(ncnt):
                cur[i] = nxt[i]
                in_nxt[nxt[i]] = 0
            ncur = ncnt

        if record:
            mass_out[n] = mass
            disp_out[n] = dm
            fall_out[n] = drained if dm > 0.0 else 0.0

    return mass


def run_sandpile_fast(N=100, eps=0.1, Zc=5.0, n_iter=200000, seed=0,
                      record_series=True, S0=None, dissip=0.0, forcing=None):
    """Active-list re-implementation of sandpile1d.run_sandpile (same signature,
    same model, same return dict). Faster for large N because each step costs
    O(active pairs) rather than O(N). See module docstring.

    Note: 'mass' is maintained incrementally here (the reference recomputes
    S.sum() each iteration). The two agree to floating-point accumulation error
    (~1e-9 over a long run); the avalanche series 'disp'/'falloff' -- the
    quantities used for the science -- are identical to machine precision when the
    two engines are driven by the same forcing stream (see _test_equivalence).
    """
    S = np.zeros(N) if S0 is None else np.array(S0, dtype=float)
    S[N - 1] = 0.0

    if forcing is None:
        use_forcing = False
        f_nodes = np.empty(0, np.int64)
        f_sizes = np.empty(0, np.float64)
    else:
        use_forcing = True
        f_nodes = np.asarray(forcing[0], np.int64)
        f_sizes = np.asarray(forcing[1], np.float64)

    if record_series:
        mass_out = np.zeros(n_iter)
        disp_out = np.zeros(n_iter)
        fall_out = np.zeros(n_iter)
    else:
        mass_out = np.zeros(1)
        disp_out = np.zeros(1)
        fall_out = np.zeros(1)

    _run_core(S, eps, Zc, n_iter, dissip, seed,
              use_forcing, f_nodes, f_sizes,
              record_series, mass_out, disp_out, fall_out)

    out = dict(S=S, N=N, eps=eps, Zc=Zc, n_iter=n_iter, seed=seed)
    if record_series:
        out['mass'] = mass_out
        out['disp'] = disp_out
        out['falloff'] = fall_out
    return out


@njit(cache=True)
def _run_core2d(S, eps, Zc, n_iter, dissip, seed,
                use_forcing, f_rows, f_cols, f_sizes,
                record, mass_out, disp_out, act_out,
                track_area, area_out):
    """JIT inner loop for the 2-D bond-slope sandpile (active-list).

    Bonds are encoded as a single integer id over [0, 2*L*L):
      id <  L*L : x-bond at (i, j) joining sites (i, j) and (i+1, j),  i=id//L, j=id%L
      id >= L*L : y-bond at (i, j) joining sites (i, j) and (i, j+1),  i,j from id-L*L
    A node (i, j) (flat index i*L+j) touches up to four bonds; only those are
    re-examined after the node moves, so each step costs O(active bonds). The model
    (synchronous bond toppling, four open edges, interior forcing, optional bulk
    dissipation) is identical to sandpile2d.run_sandpile2d.

    track_area (additive, default off via the wrapper): also record per iteration
    the number of bonds toppling for the FIRST time in the current avalanche, so
    that summing area_out over an avalanche's iterations gives its AREA = number
    of DISTINCT toppled bonds. Size (sum of act) counts every toppling; a bond can
    topple many times, so size >= area and their ratio is the mean topplings-per-
    bond. Area is bounded by the lattice and, unlike size/energy here, is not
    cutoff-dominated, so its moment scaling is the clean observable (cf. S11/S12).
    The conservative and dissipative dynamics are untouched -- this only counts.
    """
    L = S.shape[0]
    LL = L * L
    np.random.seed(seed)

    move = np.zeros((L, L))
    touched = np.empty(LL, np.int64)
    is_touched = np.zeros(LL, np.uint8)
    cur = np.empty(2 * LL, np.int64)
    nxt = np.empty(2 * LL, np.int64)
    in_nxt = np.zeros(2 * LL, np.uint8)
    ncur = 0

    # area bookkeeping: which bonds have toppled in the CURRENT avalanche, and a
    # list of them so the flags can be cleared in O(area) when the avalanche ends.
    na = 2 * LL if track_area else 1
    seen = np.zeros(na, np.uint8)
    seen_list = np.empty(na, np.int64)
    n_seen = 0

    mass = S.sum()
    fi = 0

    for n in range(n_iter):
        if ncur == 0:
            # ---- quiet loading step: one grain at a random interior site ----
            if use_forcing:
                r = f_rows[fi]
                c = f_cols[fi]
                g = f_sizes[fi]
                fi += 1
            else:
                r = np.random.randint(1, L - 1)
                c = np.random.randint(1, L - 1)
                g = np.random.random() * eps
            S[r, c] += g
            mass += g
            dm = 0.0
            ntop = 0
            new_area = 0
            # a quiet step ends any avalanche: clear the per-avalanche seen flags
            if track_area and n_seen > 0:
                for ii in range(n_seen):
                    seen[seen_list[ii]] = 0
                n_seen = 0
            # only the four bonds around (r, c) can have become unstable
            ncnt = 0
            for b in range(4):
                if b == 0:      # x-bond above: (r-1,c)-(r,c)
                    if r - 1 < 0:
                        continue
                    bid = (r - 1) * L + c
                    d = S[r, c] - S[r - 1, c]
                elif b == 1:    # x-bond below: (r,c)-(r+1,c)
                    if r > L - 2:
                        continue
                    bid = r * L + c
                    d = S[r + 1, c] - S[r, c]
                elif b == 2:    # y-bond left: (r,c-1)-(r,c)
                    if c - 1 < 0:
                        continue
                    bid = LL + r * L + (c - 1)
                    d = S[r, c] - S[r, c - 1]
                else:           # y-bond right: (r,c)-(r,c+1)
                    if c > L - 2:
                        continue
                    bid = LL + r * L + c
                    d = S[r, c + 1] - S[r, c]
                z = d if d >= 0.0 else -d
                if z >= Zc and in_nxt[bid] == 0:
                    nxt[ncnt] = bid
                    in_nxt[bid] = 1
                    ncnt += 1
            for i in range(ncnt):
                cur[i] = nxt[i]
                in_nxt[nxt[i]] = 0
            ncur = ncnt
        else:
            # ---- avalanche step: topple every currently-unstable bond at once ----
            nt = 0
            dm = 0.0
            ntop = 0
            new_area = 0
            for ii in range(ncur):
                bid = cur[ii]
                if bid < LL:
                    i = bid // L
                    j = bid - i * L
                    d = S[i + 1, j] - S[i, j]
                    a_i, a_j, b_i, b_j = i, j, i + 1, j   # low site, high site
                else:
                    m = bid - LL
                    i = m // L
                    j = m - i * L
                    d = S[i, j + 1] - S[i, j]
                    a_i, a_j, b_i, b_j = i, j, i, j + 1
                z = d if d >= 0.0 else -d
                if z < Zc:
                    continue
                dm += 0.25 * z
                ntop += 1
                if track_area and seen[bid] == 0:
                    seen[bid] = 1
                    seen_list[n_seen] = bid
                    n_seen += 1
                    new_area += 1
                if dissip == 0.0:
                    cc = d * 0.25
                    to_a, to_b = cc, -cc
                else:
                    ad = z * 0.25
                    tl = (1.0 - dissip) * ad
                    if d > 0.0:        # low site a is the lower one, receives tl
                        to_a, to_b = tl, -ad
                    else:
                        to_a, to_b = -ad, tl
                ka = a_i * L + a_j
                kb = b_i * L + b_j
                if is_touched[ka] == 0:
                    is_touched[ka] = 1
                    touched[nt] = ka
                    nt += 1
                move[a_i, a_j] += to_a
                if is_touched[kb] == 0:
                    is_touched[kb] = 1
                    touched[nt] = kb
                    nt += 1
                move[b_i, b_j] += to_b

            # pass 1: apply moves, drain touched boundary sites
            drained = 0.0
            for t in range(nt):
                node = touched[t]
                i = node // L
                j = node - i * L
                S[i, j] += move[i, j]
                mass += move[i, j]
                move[i, j] = 0.0
                if i == 0 or i == L - 1 or j == 0 or j == L - 1:
                    drained += S[i, j]
                    mass -= S[i, j]
                    S[i, j] = 0.0

            # pass 2: re-examine the bonds around every touched (now-current) site
            ncnt = 0
            for t in range(nt):
                node = touched[t]
                is_touched[node] = 0
                i = node // L
                j = node - i * L
                for b in range(4):
                    if b == 0:
                        if i - 1 < 0:
                            continue
                        bid = (i - 1) * L + j
                        d = S[i, j] - S[i - 1, j]
                    elif b == 1:
                        if i > L - 2:
                            continue
                        bid = i * L + j
                        d = S[i + 1, j] - S[i, j]
                    elif b == 2:
                        if j - 1 < 0:
                            continue
                        bid = LL + i * L + (j - 1)
                        d = S[i, j] - S[i, j - 1]
                    else:
                        if j > L - 2:
                            continue
                        bid = LL + i * L + j
                        d = S[i, j + 1] - S[i, j]
                    z = d if d >= 0.0 else -d
                    if z >= Zc and in_nxt[bid] == 0:
                        nxt[ncnt] = bid
                        in_nxt[bid] = 1
                        ncnt += 1
            for i2 in range(ncnt):
                cur[i2] = nxt[i2]
                in_nxt[nxt[i2]] = 0
            ncur = ncnt

        if record:
            mass_out[n] = mass
            disp_out[n] = dm
            act_out[n] = ntop
            if track_area:
                area_out[n] = new_area

    return mass


def run_sandpile2d_fast(L=64, eps=0.1, Zc=5.0, n_iter=1_000_000, seed=0,
                        record_series=True, S0=None, dissip=0.0, forcing=None,
                        track_area=False):
    """Active-list re-implementation of sandpile2d.run_sandpile2d (same signature,
    model, and return dict, including the 'act' bond-toppling-count series).
    Faster for large L because each step costs O(active bonds) not O(L^2).

    track_area (additive, default off so the result dict is unchanged for existing
    callers): also return an 'area' per-iteration series of first-time bond
    topplings; summed over an avalanche it gives the number of DISTINCT toppled
    bonds (the avalanche footprint). Used by the area moment analysis (S13)."""
    S = np.zeros((L, L)) if S0 is None else np.array(S0, dtype=float)
    S[0, :] = S[-1, :] = S[:, 0] = S[:, -1] = 0.0

    if forcing is None:
        use_forcing = False
        f_rows = np.empty(0, np.int64)
        f_cols = np.empty(0, np.int64)
        f_sizes = np.empty(0, np.float64)
    else:
        use_forcing = True
        f_rows = np.asarray(forcing[0], np.int64)
        f_cols = np.asarray(forcing[1], np.int64)
        f_sizes = np.asarray(forcing[2], np.float64)

    if record_series:
        mass_out = np.zeros(n_iter)
        disp_out = np.zeros(n_iter)
        act_out = np.zeros(n_iter)
        area_out = np.zeros(n_iter) if track_area else np.zeros(1)
    else:
        mass_out = np.zeros(1)
        disp_out = np.zeros(1)
        act_out = np.zeros(1)
        area_out = np.zeros(1)

    _run_core2d(S, eps, Zc, n_iter, dissip, seed,
                use_forcing, f_rows, f_cols, f_sizes,
                record_series, mass_out, disp_out, act_out,
                track_area, area_out)

    out = dict(S=S, L=L, eps=eps, Zc=Zc, n_iter=n_iter, seed=seed)
    if record_series:
        out['mass'] = mass_out
        out['disp'] = disp_out
        out['act'] = act_out
        if track_area:
            out['area'] = area_out
    return out


def _make_forcing(N, n_iter, eps, seed):
    """A fixed forcing stream long enough for any run: one (node, size) per
    iteration is a safe upper bound on the number of quiet steps."""
    rng = np.random.default_rng(seed)
    nodes = rng.integers(0, N, size=n_iter)
    sizes = rng.uniform(0.0, eps, size=n_iter)
    return nodes, sizes


def _test_equivalence():
    """Drive the fast and reference engines with ONE shared forcing stream and
    confirm they produce the same avalanche series to machine precision."""
    print("self-test: fast vs reference engine (shared forcing)")
    Zc, eps = 5.0, 0.1
    for N, n_iter in ((60, 150_000), (200, 400_000)):
        forcing = _make_forcing(N, n_iter, eps, seed=11)
        S0 = triangle_ic(N, 0.9 * Zc)
        ref = run_sandpile(N=N, eps=eps, Zc=Zc, n_iter=n_iter, S0=S0.copy(),
                           forcing=forcing)
        fast = run_sandpile_fast(N=N, eps=eps, Zc=Zc, n_iter=n_iter, S0=S0.copy(),
                                 forcing=forcing)
        ddisp = np.abs(ref['disp'] - fast['disp']).max()
        dfall = np.abs(ref['falloff'] - fast['falloff']).max()
        dS = np.abs(ref['S'] - fast['S']).max()
        dmass = np.abs(ref['mass'] - fast['mass']).max()
        # avalanche structure must be identical, not merely close
        Er, _, Tr = measure_avalanches(ref['disp'])
        Ef, _, Tf = measure_avalanches(fast['disp'])
        print("  N=%4d: max|d disp|=%.2e  max|d falloff|=%.2e  max|d S|=%.2e  "
              "max|d mass|=%.2e" % (N, ddisp, dfall, dS, dmass))
        print("          #avalanches ref=%d fast=%d   T_max ref=%d fast=%d"
              % (Er.size, Ef.size, int(Tr.max()) if Tr.size else -1,
                 int(Tf.max()) if Tf.size else -1))
        assert ddisp < 1e-9, "displaced-mass series diverged"
        assert dfall < 1e-9, "falloff series diverged"
        assert dS < 1e-9, "final state diverged"
        assert Er.size == Ef.size, "avalanche count differs"
        assert np.array_equal(Tr, Tf), "avalanche durations differ"
    print("  equivalence OK")


def _make_forcing2d(L, n_iter, eps, seed):
    rng = np.random.default_rng(seed)
    rows = rng.integers(1, L - 1, size=n_iter)
    cols = rng.integers(1, L - 1, size=n_iter)
    sizes = rng.uniform(0.0, eps, size=n_iter)
    return rows, cols, sizes


def _test_equivalence2d():
    """2-D: drive fast and reference engines with one shared forcing stream and
    confirm identical avalanche series (disp, act) to machine precision."""
    print("self-test: 2-D fast vs reference engine (shared forcing)")
    Zc, eps = 5.0, 0.1
    for L, n_iter in ((16, 120_000), (40, 400_000)):
        forcing = _make_forcing2d(L, n_iter, eps, seed=9)
        S0 = pyramid_ic(L, 0.9 * Zc)
        ref = run_sandpile2d(L=L, eps=eps, Zc=Zc, n_iter=n_iter, S0=S0.copy(),
                             forcing=forcing)
        fast = run_sandpile2d_fast(L=L, eps=eps, Zc=Zc, n_iter=n_iter, S0=S0.copy(),
                                   forcing=forcing)
        ddisp = np.abs(ref['disp'] - fast['disp']).max()
        dact = np.abs(ref['act'] - fast['act']).max()
        dS = np.abs(ref['S'] - fast['S']).max()
        Er, _, Tr = measure_avalanches(ref['disp'])
        Ef, _, Tf = measure_avalanches(fast['disp'])
        print("  L=%3d: max|d disp|=%.2e  max|d act|=%.0f  max|d S|=%.2e   "
              "#av ref=%d fast=%d  T_max ref=%d fast=%d"
              % (L, ddisp, dact, dS, Er.size, Ef.size,
                 int(Tr.max()) if Tr.size else -1, int(Tf.max()) if Tf.size else -1))
        assert ddisp < 1e-9, "2-D displaced-mass series diverged"
        assert dact == 0, "2-D toppling-count series differs"
        assert dS < 1e-9, "2-D final state diverged"
        assert Er.size == Ef.size and np.array_equal(Tr, Tf), "2-D avalanches differ"
    print("  2-D equivalence OK")


def _brute_area2d(L, eps, Zc, n_iter, forcing):
    """Independent full-scan reference (mirrors sandpile2d.run_sandpile2d) that
    additionally records, per avalanche, the set of DISTINCT bonds that toppled,
    using the SAME bond encoding as the fast engine. Returns (disp, act, area)
    per-iteration arrays, with area[n] = number of bonds toppling for the first
    time this avalanche on iteration n (so summing over an avalanche = its area)."""
    f_rows, f_cols, f_sizes = forcing
    LL = L * L
    S = pyramid_ic(L, 0.9 * Zc)
    disp = np.zeros(n_iter); act = np.zeros(n_iter); area = np.zeros(n_iter)
    seen = set()
    fi = 0
    for n in range(n_iter):
        dx = S[1:, :] - S[:-1, :]
        dy = S[:, 1:] - S[:, :-1]
        ux = np.abs(dx) >= Zc
        uy = np.abs(dy) >= Zc
        if ux.any() or uy.any():
            move = np.zeros((L, L))
            cx = np.where(ux, dx * 0.25, 0.0)
            move[:-1, :] += cx; move[1:, :] -= cx
            cy = np.where(uy, dy * 0.25, 0.0)
            move[:, :-1] += cy; move[:, 1:] -= cy
            S += move
            disp[n] = np.abs(cx).sum() + np.abs(cy).sum()
            act[n] = int(ux.sum()) + int(uy.sum())
            # distinct-bond bookkeeping with the fast engine's id convention
            new = 0
            xi, xj = np.nonzero(ux)              # x-bond row i (0..L-2), col j
            for i, j in zip(xi, xj):
                bid = i * L + j
                if bid not in seen:
                    seen.add(bid); new += 1
            yi, yj = np.nonzero(uy)              # y-bond row i, col j (0..L-2)
            for i, j in zip(yi, yj):
                bid = LL + i * L + j
                if bid not in seen:
                    seen.add(bid); new += 1
            area[n] = new
        else:
            seen.clear()                          # avalanche ended -> reset
            S[f_rows[fi], f_cols[fi]] += f_sizes[fi]
            fi += 1
        S[0, :] = S[-1, :] = S[:, 0] = S[:, -1] = 0.0
    return disp, act, area


def _group_sum(disp, x):
    """Sum the parallel series x over each avalanche (maximal run of disp>0)."""
    active = disp > 0.0
    if not active.any():
        return np.array([])
    a = active.astype(np.int8)
    edges = np.diff(a)
    starts = np.flatnonzero(edges == 1) + 1
    ends = np.flatnonzero(edges == -1) + 1
    if active[0]:
        starts = np.r_[0, starts]
    if active[-1]:
        ends = np.r_[ends, active.size]
    return np.array([x[s:e].sum() for s, e in zip(starts, ends)])


def _test_area2d():
    """Validate the 2-D area counter: (1) tracking area must not perturb the
    dynamics (disp/act identical with track_area on vs off), and (2) the
    per-avalanche area must match an independent full-scan distinct-bond recount."""
    print("self-test: 2-D avalanche AREA (distinct toppled bonds)")
    Zc, eps = 5.0, 0.1
    for L, n_iter in ((16, 150_000), (32, 250_000)):
        forcing = _make_forcing2d(L, n_iter, eps, seed=9)
        S0 = pyramid_ic(L, 0.9 * Zc)
        off = run_sandpile2d_fast(L=L, eps=eps, Zc=Zc, n_iter=n_iter, S0=S0.copy(),
                                  forcing=forcing, track_area=False)
        on = run_sandpile2d_fast(L=L, eps=eps, Zc=Zc, n_iter=n_iter, S0=S0.copy(),
                                 forcing=forcing, track_area=True)
        d_disp = np.abs(off['disp'] - on['disp']).max()
        d_act = np.abs(off['act'] - on['act']).max()
        bdisp, bact, barea = _brute_area2d(L, eps, Zc, n_iter, forcing)
        # per-avalanche area: fast (sum of first-topple flags) vs brute (union size)
        area_fast = _group_sum(on['disp'], on['area'])
        area_brute = _group_sum(bdisp, barea)
        size_av = _group_sum(on['disp'], on['act'])
        d_area = (np.abs(area_fast - area_brute).max()
                  if area_fast.size == area_brute.size else 9e9)
        print("  L=%3d: track on/off  max|d disp|=%.2e max|d act|=%.0f | "
              "#av=%d  max|area_fast-area_brute|=%.0f  area<=size:%s"
              % (L, d_disp, d_act, area_fast.size, d_area,
                 bool((area_fast <= size_av + 1e-9).all())))
        assert d_disp < 1e-12 and d_act == 0, "area tracking perturbed the dynamics"
        assert area_fast.size == area_brute.size, "avalanche count mismatch"
        assert d_area == 0, "area series disagrees with brute-force recount"
        assert (area_fast <= size_av + 1e-9).all(), "area exceeds size (impossible)"
        assert (area_fast >= 1).all(), "an avalanche with zero distinct bonds"
    print("  2-D area OK")


def _test_dissipation_equivalence():
    """The non-conservative path (dissip>0) must also match the reference."""
    print("self-test: fast vs reference engine (dissipative, shared forcing)")
    Zc, eps, N, n_iter = 5.0, 0.1, 120, 300_000
    forcing = _make_forcing(N, n_iter, eps, seed=5)
    S0 = triangle_ic(N, 0.9 * Zc)
    for d in (0.05, 0.20):
        ref = run_sandpile(N=N, eps=eps, Zc=Zc, n_iter=n_iter, S0=S0.copy(),
                           dissip=d, forcing=forcing)
        fast = run_sandpile_fast(N=N, eps=eps, Zc=Zc, n_iter=n_iter, S0=S0.copy(),
                                 dissip=d, forcing=forcing)
        ddisp = np.abs(ref['disp'] - fast['disp']).max()
        print("  dissip=%.2f: max|d disp|=%.2e" % (d, ddisp))
        assert ddisp < 1e-9, "dissipative path diverged"
    print("  dissipative equivalence OK")


def _test_dissipation_equivalence2d():
    """2-D non-conservative path must also match the reference."""
    print("self-test: 2-D fast vs reference engine (dissipative, shared forcing)")
    Zc, eps, L, n_iter = 5.0, 0.1, 32, 300_000
    forcing = _make_forcing2d(L, n_iter, eps, seed=4)
    S0 = pyramid_ic(L, 0.9 * Zc)
    for d in (0.05, 0.20):
        ref = run_sandpile2d(L=L, eps=eps, Zc=Zc, n_iter=n_iter, S0=S0.copy(),
                             dissip=d, forcing=forcing)
        fast = run_sandpile2d_fast(L=L, eps=eps, Zc=Zc, n_iter=n_iter, S0=S0.copy(),
                                   dissip=d, forcing=forcing)
        ddisp = np.abs(ref['disp'] - fast['disp']).max()
        print("  dissip=%.2f: max|d disp|=%.2e" % (d, ddisp))
        assert ddisp < 1e-9, "2-D dissipative path diverged"
    print("  2-D dissipative equivalence OK")


def _benchmark():
    """Wall-clock comparison at a few sizes (steady state via triangle IC)."""
    import time
    print("benchmark 1-D: reference vs fast (triangle IC, steady state)")
    Zc, eps = 5.0, 0.1
    # warm the JIT once so the timing excludes compilation
    run_sandpile_fast(N=50, n_iter=2000, S0=triangle_ic(50, 4.5))
    for N, n_iter in ((256, 2_000_000), (1024, 4_000_000), (4096, 4_000_000)):
        S0 = triangle_ic(N, 0.9 * Zc)
        t = time.time()
        rf = run_sandpile_fast(N=N, eps=eps, Zc=Zc, n_iter=n_iter, S0=S0.copy())
        tf = time.time() - t
        nav = measure_avalanches(rf['disp'])[0].size
        msg = "  N=%5d  %9d iter   fast %6.2fs  (#av=%d)" % (N, n_iter, tf, nav)
        if N <= 1024:
            t = time.time()
            run_sandpile(N=N, eps=eps, Zc=Zc, n_iter=n_iter, S0=S0.copy())
            tr = time.time() - t
            msg += "   ref %7.2fs   speedup %5.1fx" % (tr, tr / tf)
        print(msg)

    print("benchmark 2-D: reference vs fast (pyramid IC, steady state)")
    run_sandpile2d_fast(L=16, n_iter=2000, S0=pyramid_ic(16, 4.5))
    for L, n_iter in ((64, 1_000_000), (128, 1_000_000), (256, 1_000_000)):
        S0 = pyramid_ic(L, 0.9 * Zc)
        t = time.time()
        rf = run_sandpile2d_fast(L=L, eps=eps, Zc=Zc, n_iter=n_iter, S0=S0.copy())
        tf = time.time() - t
        nav = measure_avalanches(rf['disp'])[0].size
        msg = "  L=%4d  %9d iter   fast %6.2fs  (#av=%d)" % (L, n_iter, tf, nav)
        if L <= 128:
            t = time.time()
            run_sandpile2d(L=L, eps=eps, Zc=Zc, n_iter=n_iter, S0=S0.copy())
            tr = time.time() - t
            msg += "   ref %7.2fs   speedup %5.1fx" % (tr, tr / tf)
        print(msg)


if __name__ == "__main__":
    _test_equivalence()
    _test_equivalence2d()
    _test_area2d()
    _test_dissipation_equivalence()
    _test_dissipation_equivalence2d()
    _benchmark()
