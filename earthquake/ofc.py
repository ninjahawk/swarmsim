"""
ofc.py -- Olami-Feder-Christensen earthquake model (Charbonneau Ch. 8).

A 2-D lattice of "force" values F[i,j] driven uniformly until a node reaches a
threshold Fc and topples: F -> 0 and each of the 4 nearest neighbours gains
alpha*F, with 0 <= alpha <= 0.25.  At alpha = 0.25 redistribution is conservative;
for alpha < 0.25 a fraction (1 - 4*alpha) of the toppling node's force is lost to
the "upper plate" (bulk dissipation).  Open boundaries (the ghost border, held at
0) also leak force off the lattice.  Stop-and-go driving: forcing pauses during an
avalanche (timescale separation), exactly as in the Ch. 5 sandpile.

This is the earthquake analogue of the chapter-5 slope sandpile, but with the
stability criterion on the nodal VALUE rather than the slope, and -- crucially --
with a tunable bulk conservation parameter alpha.  That single change is what makes
this the natural continuation of the sandpile conservation thread (findings S5/S7):
S5 found bulk conservation NECESSARY for true SOC in the slope sandpile; OFC is the
canonical model in which that question is "perennially debated", and alpha is the
exact knob.

Reference: Charbonneau, Natural Complexity, Ch. 8, eqs. 8.10-8.14, fig. 8.3.
Olami, Feder & Christensen, PRL 68, 1244 (1992).

Conventions match the rest of the repo: vectorized, ASCII-only prints,
__main__-guarded self-tests.  The size measure E is the total number of topplings
in an avalanche (counting repeats), per the book.  Avalanche duration T is the
number of synchronous relaxation sweeps.

The engine is a synchronous (parallel) relaxation identical in update order to the
book's listing (fig. 8.3): all currently-unstable nodes topple together using the
pre-sweep force values, then the lattice is re-scanned.  Forcing between avalanches
uses the exact skip-ahead of Exercise 3 (jump (Fc - max F)/delta_f forcing steps at
once); this is bit-identical to one-step-at-a-time forcing for the avalanche
statistics and reproduces the iteration timeline exactly, so the recurrence-period
and prediction analyses (which count iterations) are unaffected.
"""

import numpy as np

FC_DEFAULT = 1.0
DELTA_F_DEFAULT = 1.0e-4


def _relax(F, alpha, bc, rng=None, alpha_lo=None, alpha_hi=None):
    """Run one avalanche to completion on lattice F (modified in place).

    Returns (size, duration): total topplings and number of synchronous sweeps.
    If alpha_lo/alpha_hi are given, a fresh uniform alpha in [alpha_lo, alpha_hi]
    is drawn for every toppling node each sweep (Exercise 4, stochastic alpha).
    """
    size = 0
    dur = 0
    while True:
        U = F >= FC_DEFAULT
        n = int(U.sum())
        if n == 0:
            break
        dur += 1
        size += n
        if alpha_lo is not None:
            a = np.where(U, rng.uniform(alpha_lo, alpha_hi, F.shape), 0.0)
            send = a * F            # per-node alpha (zero where stable)
        else:
            send = alpha * (F * U)
        # zero the toppling nodes, then add neighbour contributions
        newF = F - F * U
        if bc == 'open':
            newF[1:, :] += send[:-1, :]     # contribution downward (from row above)
            newF[:-1, :] += send[1:, :]     # upward
            newF[:, 1:] += send[:, :-1]     # rightward
            newF[:, :-1] += send[:, 1:]     # leftward
        elif bc == 'periodic':
            newF += np.roll(send, 1, axis=0)
            newF += np.roll(send, -1, axis=0)
            newF += np.roll(send, 1, axis=1)
            newF += np.roll(send, -1, axis=1)
        else:
            raise ValueError("bc must be 'open' or 'periodic'")
        F[:, :] = newF
    return size, dur


def run_ofc(N=128, alpha=0.15, n_events=20000, bc='open', seed=0,
            warmup_events=2000, delta_f=DELTA_F_DEFAULT, Fc=FC_DEFAULT,
            record_iter=False, record_lattice_every=0, alpha_noise=None,
            naive_forcing=False):
    """Drive an OFC lattice and collect avalanche statistics.

    Parameters
    ----------
    N, alpha, bc, delta_f, Fc : model parameters (Fc fixed at 1 by convention).
    n_events     : number of avalanches to record (after warmup).
    warmup_events: avalanches discarded to reach the statistically stationary state.
    record_iter  : also return the global iteration index at each recorded avalanche
                   (forcing steps + relaxation sweeps), needed for recurrence-period
                   and prediction analyses.
    record_lattice_every : if >0, store a copy of F every this-many recorded events.
    alpha_noise  : (lo, hi) to draw alpha per toppling node each sweep (Exercise 4).
    naive_forcing: if True, force one delta_f step at a time (slow; for validating
                   the skip-ahead). Default False uses the exact Exercise-3 skip.

    Returns a dict: sizes, durations, (iters), (lattices), n_iter_total, params.
    """
    if Fc != FC_DEFAULT:
        # the engine hard-codes Fc=1 in _relax for speed; rescale is trivial but
        # the book uses Fc=1 throughout, so we keep it fixed.
        raise ValueError("this implementation fixes Fc=1 (book convention)")
    rng = np.random.RandomState(seed)
    F = rng.uniform(0.0, Fc, size=(N, N))
    alo, ahi = (alpha_noise if alpha_noise is not None else (None, None))

    total = warmup_events + n_events
    sizes = np.empty(n_events, dtype=np.int64)
    durs = np.empty(n_events, dtype=np.int64)
    iters = np.empty(n_events, dtype=np.int64) if record_iter else None
    lattices = []

    it = 0          # global iteration counter (forcing steps + sweeps)
    rec = 0
    for e in range(total):
        # ---- forcing phase (stop-and-go): advance until the max node hits Fc ----
        fmax = F.max()
        if fmax < Fc:
            if naive_forcing:
                steps = 0
                while F.max() < Fc:
                    F += delta_f
                    steps += 1
            else:
                steps = int(np.ceil((Fc - fmax) / delta_f))
                F += steps * delta_f
            it += steps
        # ---- avalanche ----
        size, dur = _relax(F, alpha, bc, rng=rng, alpha_lo=alo, alpha_hi=ahi)
        it += dur
        if e >= warmup_events:
            sizes[rec] = size
            durs[rec] = dur
            if record_iter:
                iters[rec] = it
            if record_lattice_every and (rec % record_lattice_every == 0):
                lattices.append(F.copy())
            rec += 1

    out = dict(sizes=sizes, durations=durs, n_iter_total=it,
               N=N, alpha=alpha, bc=bc, delta_f=delta_f)
    if record_iter:
        out['iters'] = iters
    if record_lattice_every:
        out['lattices'] = lattices
    return out


# ----------------------------------------------------------------- analysis utils
def logbin_pdf(sizes, nbins=24, smin=1):
    """Log-binned PDF of a set of (integer) avalanche sizes.

    Returns (centers, pdf) for bins with at least one count, normalized so the
    histogram integrates to 1 over size.
    """
    sizes = np.asarray(sizes)
    sizes = sizes[sizes >= smin]
    if sizes.size == 0:
        return np.array([]), np.array([])
    smax = sizes.max()
    edges = np.unique(np.floor(np.logspace(np.log10(smin),
                                           np.log10(smax + 1), nbins + 1)).astype(np.int64))
    edges = edges.astype(float)
    counts, _ = np.histogram(sizes, bins=edges)
    widths = np.diff(edges)
    centers = np.sqrt(edges[:-1] * edges[1:])
    pdf = counts / widths / sizes.size
    good = counts > 0
    return centers[good], pdf[good]


def powerlaw_slope(centers, pdf, lo=None, hi=None):
    """Least-squares slope of log10(pdf) vs log10(center) over [lo, hi]."""
    centers = np.asarray(centers)
    pdf = np.asarray(pdf)
    m = (pdf > 0)
    if lo is not None:
        m &= (centers >= lo)
    if hi is not None:
        m &= (centers <= hi)
    if m.sum() < 2:
        return np.nan
    return np.polyfit(np.log10(centers[m]), np.log10(pdf[m]), 1)[0]


# ----------------------------------------------------------------- self-tests
def _selftest():
    print("OFC model self-tests")
    print("-" * 60)

    # 1. single interior topple, conservative alpha=0.25
    F = np.zeros((5, 5))
    F[2, 2] = 1.2
    s, d = _relax(F, 0.25, 'open')
    ok = (s == 1 and d == 1 and abs(F[2, 2]) < 1e-12 and
          abs(F[1, 2] - 0.3) < 1e-12 and abs(F[3, 2] - 0.3) < 1e-12 and
          abs(F[2, 1] - 0.3) < 1e-12 and abs(F[2, 3] - 0.3) < 1e-12)
    print("1. interior topple alpha=0.25 (4 nbrs x 0.30, site->0): %s" % ok)
    assert ok

    # 2. dissipation bookkeeping: alpha=0.15 bulk topple loses (1-4a)F
    F = np.zeros((5, 5))
    F[2, 2] = 1.2
    before = F.sum()
    _relax(F, 0.15, 'open')
    lost = before - F.sum()
    ok = abs(lost - (1 - 4 * 0.15) * 1.2) < 1e-12
    print("2. bulk dissipation alpha=0.15 lost=%.4f == (1-4a)F=%.4f: %s"
          % (lost, (1 - 4 * 0.15) * 1.2, ok))
    assert ok

    # 3. corner topple on open BC: only 2 in-lattice neighbours
    F = np.zeros((5, 5))
    F[0, 0] = 1.2
    before = F.sum()
    _relax(F, 0.25, 'open')
    ok = (abs(F[0, 1] - 0.3) < 1e-12 and abs(F[1, 0] - 0.3) < 1e-12 and
          abs(F[0, 0]) < 1e-12 and abs((before - F.sum()) - 0.6) < 1e-12)
    print("3. corner topple open BC (2 nbrs x 0.30, 0.60 leaks off-edge): %s" % ok)
    assert ok

    # 4. periodic + conservative alpha=0.25 conserves total over a bulk topple
    F = np.zeros((5, 5))
    F[2, 2] = 1.2
    before = F.sum()
    _relax(F, 0.25, 'periodic')
    ok = abs(before - F.sum()) < 1e-12
    print("4. periodic conservative topple conserves total (%.4f): %s"
          % (F.sum(), ok))
    assert ok

    # 5. after every avalanche the lattice is stable; skip-forcing == naive forcing
    r_fast = run_ofc(N=24, alpha=0.2, n_events=500, warmup_events=200, seed=1)
    r_slow = run_ofc(N=24, alpha=0.2, n_events=500, warmup_events=200, seed=1,
                     naive_forcing=True)
    ok = np.array_equal(r_fast['sizes'], r_slow['sizes']) and \
        np.array_equal(r_fast['durations'], r_slow['durations'])
    print("5. skip-ahead forcing bit-identical to naive forcing: %s" % ok)
    assert ok

    # 6. stationary-state sanity: alpha=0.25 produces a broad size distribution
    r = run_ofc(N=32, alpha=0.25, n_events=3000, warmup_events=1000, seed=2)
    print("6. alpha=0.25 N=32: %d events, max size %d, mean %.1f"
          % (r['sizes'].size, r['sizes'].max(), r['sizes'].mean()))
    assert r['sizes'].max() > 50

    print("-" * 60)
    print("all self-tests passed")


if __name__ == '__main__':
    _selftest()
