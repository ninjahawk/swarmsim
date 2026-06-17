"""
Phase 2b -- universality head-to-head: our 2-D slope sandpile vs the canonical
2-D abelian sandpile (Bak-Tang-Wiesenfeld, 1987).

The slope model (sandpile2d.py) tracks continuous heights and topples *bonds*
when a height DIFFERENCE exceeds Zc. The canonical BTW model is different in
character: it tracks integer heights and topples a SITE when its own height
reaches a threshold, shedding one grain to each of its 4 neighbours. Both are
SOC, but are they in the same universality class? The decisive test is whether
the avalanche-size exponent tau_S agrees. We measure BTW here under the SAME
log-binned PDF / finite-size-scaling pipeline used for the slope model, so the
comparison is method-matched rather than against quoted literature numbers
(reported value for 2-D BTW size: tau_S ~ 1.2, with known multifractal
corrections).

BTW rules (open boundaries, grains topple off the lattice edges and are lost):
  - height h[i,j] integer, threshold Zc = 4 (= number of neighbours);
  - drive: add 1 grain at a random site; then relax fully before the next grain
    (separation of timescales);
  - topple: every site with h >= 4 loses 4 and gives 1 to each neighbour,
    synchronously, repeated until all stable;
  - avalanche size S = total number of topplings; duration T = number of
    parallel relaxation sweeps.

Run from repo root:  python sandpile/btw_compare.py
Writes figures/sandpile_btw.png and outputs/sandpile_btw.txt.
ASCII-only.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validate1d import logbin_pdf, powerlaw_slope

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "figures")
OUTDIR = os.path.join(ROOT, "outputs")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

LOG = []
def log(msg):
    print(msg)
    LOG.append(msg)


def btw_run(L, n_events, warm, seed=0, track_area=False):
    """Canonical 2-D abelian BTW. Returns (S, T) arrays of avalanche size
    (total topplings) and duration for events past warmup that produced at least
    one toppling.

    track_area (additive, default off so existing callers are unchanged): also
    return A, the avalanche AREA = number of DISTINCT sites that toppled at least
    once. Size S counts every toppling (a site can topple many times in one
    avalanche), so S >= A; the gap between them is what drives BTW's toppling-
    number multifractality (multiple topplings per site), and the moment thread
    (S11) tests whether area is the simpler, FSS-obeying observable.
    """
    rng = np.random.default_rng(seed)
    h = np.zeros((L, L), dtype=np.int64)
    Sz, Tz, Az = [], [], []
    ever = np.zeros((L, L), dtype=bool) if track_area else None
    for ev in range(n_events):
        h[rng.integers(0, L), rng.integers(0, L)] += 1
        size = 0
        dur = 0
        if track_area:
            ever[:] = False
        while True:
            unstable = h >= 4
            n_un = int(unstable.sum())
            if n_un == 0:
                break
            size += n_un
            dur += 1
            if track_area:
                ever |= unstable
            h -= 4 * unstable
            # ship one grain to each neighbour; off-grid grains are lost (open BC)
            h[1:, :] += unstable[:-1, :]
            h[:-1, :] += unstable[1:, :]
            h[:, 1:] += unstable[:, :-1]
            h[:, :-1] += unstable[:, 1:]
        if ev >= warm and size > 0:
            Sz.append(size)
            Tz.append(dur)
            if track_area:
                Az.append(int(ever.sum()))
    if track_area:
        return (np.array(Sz, dtype=float), np.array(Tz, dtype=float),
                np.array(Az, dtype=float))
    return np.array(Sz, dtype=float), np.array(Tz, dtype=float)


def cutoff_moment(x):
    x = np.asarray(x, dtype=float)
    return (x**2).mean() / x.mean()


def main():
    log("=" * 70)
    log("CANONICAL 2-D ABELIAN BTW SANDPILE -- exponents under our pipeline")
    log("=" * 70)

    configs = [
        (32,  120_000, 20_000),
        (48,  120_000, 30_000),
        (64,  100_000, 30_000),
        (96,   80_000, 30_000),
        (128,  60_000, 25_000),
    ]
    data = {}
    log("\nRunning BTW lattices...")
    for L, n_ev, warm in configs:
        S, T = btw_run(L, n_ev, warm, seed=5)
        data[L] = dict(S=S, T=T)
        log("  L=%4d : %6d avalanches  (S_max=%.3g, T_max=%d, <S>=%.1f)"
            % (L, S.size, S.max() if S.size else -1,
               int(T.max()) if T.size else -1, S.mean() if S.size else -1))

    Ls = np.array([c[0] for c in configs], dtype=float)
    Lbig = Ls[-1]

    cb, db = logbin_pdf(data[Lbig]['S'])
    tauS = -powerlaw_slope(cb, db, lo=cb.min() * 5, hi=cb.max() * 0.1)
    xc = np.array([cutoff_moment(data[L]['S']) for L in Ls])
    DS = np.polyfit(np.log10(Ls), np.log10(xc), 1)[0]
    log("\n[BTW size exponent]")
    log("  tau_S(BTW) = %.3f   D_S(BTW) = %.3f   (literature tau_S ~ 1.2)" % (tauS, DS))
    log("\n  Compare with the 2-D SLOPE model (fss2d.py): tau_S(slope) printed there.")
    log("  If tau_S(slope) != tau_S(BTW), the slope rule and BTW are in DIFFERENT")
    log("  universality classes despite both being 2-D SOC sandpiles.")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    cmap = plt.cm.viridis(np.linspace(0, 0.85, len(Ls)))
    for c, L in zip(cmap, Ls):
        cs, ds = logbin_pdf(data[L]['S'])
        ax[0].loglog(cs, ds, "-", color=c, label="L=%d" % int(L))
        ax[1].loglog(cs / L**DS, ds * cs**tauS, "-", color=c, label="L=%d" % int(L))
    ax[0].set_title("BTW avalanche-size PDFs (raw)")
    ax[0].set_xlabel("size S (topplings)"); ax[0].set_ylabel("PDF(S)"); ax[0].legend(fontsize=8)
    ax[1].set_title("BTW size collapse (tau_S=%.2f, D_S=%.2f)" % (tauS, DS))
    ax[1].set_xlabel("S / L^%.2f" % DS); ax[1].set_ylabel("S^%.2f PDF(S)" % tauS); ax[1].legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "sandpile_btw.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    log("\nsaved %s" % os.path.relpath(p, ROOT))

    log("\nBTW COMPARISON COMPLETE")
    with open(os.path.join(OUTDIR, "sandpile_btw.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
