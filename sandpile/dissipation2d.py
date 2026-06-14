"""
S7 -- does the conservation-necessity result (S5) transfer to two dimensions?

S5 showed, in 1-D, that bulk conservation is necessary for true SOC: with
conservation the avalanche cutoff scales with system size (no characteristic
size), but any bulk dissipation truncates avalanches at a dissipation-set scale
and the cutoff stops scaling. A good result should not depend on dimensionality.
Here we repeat the test for the 2-D bond-slope sandpile (sandpile2d.py, with the
same dissip parameter): break conservation and check whether the cutoff still
scales with L.

Run from repo root:  python sandpile/dissipation2d.py
Writes figures/sandpile_dissipation2d.png and outputs/sandpile_dissipation2d.txt.
ASCII-only.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sandpile2d import run_sandpile2d, pyramid_ic
from sandpile1d import measure_avalanches
from validate1d import logbin_pdf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(ROOT, "figures")
OUTDIR = os.path.join(ROOT, "outputs")
os.makedirs(FIGDIR, exist_ok=True)
os.makedirs(OUTDIR, exist_ok=True)

LOG = []
def log(msg):
    print(msg)
    LOG.append(msg)


def cutoff_moment(x):
    x = np.asarray(x, dtype=float)
    return np.nan if x.size == 0 else (x**2).mean() / x.mean()


def main():
    log("=" * 70)
    log("DOES CONSERVATION-NECESSITY TRANSFER TO 2-D?  (S7)")
    log("=" * 70)

    Ls = [32, 48, 64, 96]
    iters = {32: 600_000, 48: 900_000, 64: 1_200_000, 96: 1_800_000}
    warms = {32: 200_000, 48: 300_000, 64: 400_000, 96: 600_000}
    dissips = [0.0, 0.05, 0.20]
    Zc, eps = 5.0, 0.1

    cutoff = {d: {} for d in dissips}
    pdf_fix = {}
    Lfix = Ls[-1]
    for d in dissips:
        row = []
        for L in Ls:
            res = run_sandpile2d(L=L, eps=eps, Zc=Zc, n_iter=iters[L], seed=2,
                                 S0=pyramid_ic(L, 0.90 * Zc), dissip=d)
            E, P, T = measure_avalanches(res['disp'][warms[L]:])
            cutoff[d][L] = cutoff_moment(E)
            row.append(cutoff[d][L])
            if L == Lfix:
                pdf_fix[d] = logbin_pdf(E)
        log("  dissip=%.2f : cutoff(E) by L %s = %s"
            % (d, Ls, ["%.3g" % v for v in row]))

    log("\n[cutoff scaling exponent  d log(cutoff)/d log(L)]")
    log("  (2-D conservative SOC: energy cutoff ~ L^~2; -> ~0 means saturation)")
    slopes = {}
    for d in dissips:
        cv = np.array([cutoff[d][L] for L in Ls], dtype=float)
        slopes[d] = np.polyfit(np.log10(Ls), np.log10(cv), 1)[0]
        log("  dissip=%.2f : slope %.2f" % (d, slopes[d]))

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    cmap = plt.cm.cool(np.linspace(0, 1, len(dissips)))
    for c, d in zip(cmap, dissips):
        cv = [cutoff[d][L] for L in Ls]
        ax[0].loglog(Ls, cv, "o-", color=c, label="dissip=%.2f (slope %.2f)" % (d, slopes[d]))
    ax[0].set_xlabel("system size L"); ax[0].set_ylabel("avalanche-energy cutoff")
    ax[0].set_title("2-D: conservative cutoff scales with L;\ndissipation flattens it (cf. 1-D S5)")
    ax[0].legend(fontsize=9)
    for c, d in zip(cmap, dissips):
        ce, de = pdf_fix[d]
        if ce.size:
            ax[1].loglog(ce, de, "-", color=c, label="dissip=%.2f" % d)
    ax[1].set_xlabel("avalanche energy E"); ax[1].set_ylabel("PDF(E)")
    ax[1].set_title("2-D energy PDF at L=%d: dissipation truncates\nthe power law" % Lfix)
    ax[1].legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "sandpile_dissipation2d.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    log("\nsaved %s" % os.path.relpath(p, ROOT))

    log("\nVerdict: if d=0 scales with L while d>0 flattens, the 1-D")
    log("  conservation-necessity result (S5) transfers to 2-D.")
    with open(os.path.join(OUTDIR, "sandpile_dissipation2d.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
