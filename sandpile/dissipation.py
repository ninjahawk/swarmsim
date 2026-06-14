"""
Phase 3 -- which SOC ingredient is necessary? Testing conservation.

Charbonneau lists four conditions that appear sufficient (perhaps necessary) for
self-organized criticality: a system that is (1) open and dissipative, (2) loaded
by slow forcing, (3) subject to a local threshold instability, and (4) restores
stability through local relaxation. The 1-D slope sandpile is CONSERVATIVE in its
bulk: the redistribution rule moves sand without destroying it, and sand leaves
ONLY at the open boundary. Is that bulk conservation essential, or incidental?

We break it. The non-conservative variant (sandpile1d.run_sandpile(dissip=d))
keeps the same threshold and relaxation, but each topple now destroys a fraction
d of the sand it moves (the higher node sheds |slope|/4, the lower node receives
only (1-d)*|slope|/4). This is the sandpile analog of the Olami-Feder-Christensen
earthquake model, where non-conservation is the famous controversial ingredient.

The decisive test of criticality is NOT whether avalanches still occur, but
whether their cutoff still SCALES with system size. A truly critical system has
no characteristic avalanche size: the cutoff grows with N without bound (here
~N^2, S3). If bulk dissipation introduces a finite correlation length, the cutoff
will SATURATE to an N-independent value -- avalanches acquire a characteristic
size set by the dissipation, not the system, and the system is only
"sub-critical" / approximately critical, no longer scale-free.

Prediction: conservative (d=0) cutoff ~ N^2 (scales); any d>0 cutoff saturates
for N beyond a dissipation-set length, so conservation IS necessary for true SOC.

Run from repo root:  python sandpile/dissipation.py
Writes figures/sandpile_dissipation.png and outputs/sandpile_dissipation.txt.
ASCII-only.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sandpile1d import run_sandpile, measure_avalanches, triangle_ic
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
    if x.size == 0:
        return np.nan
    return (x**2).mean() / x.mean()


def main():
    log("=" * 70)
    log("IS CONSERVATION NECESSARY FOR SOC?  (1-D sandpile, bulk dissipation)")
    log("=" * 70)

    Ns = [100, 200, 400, 800]
    iters = {100: 1_000_000, 200: 1_500_000, 400: 2_000_000, 800: 3_000_000}
    warms = {100: 300_000, 200: 400_000, 400: 600_000, 800: 800_000}
    dissips = [0.0, 0.02, 0.05, 0.10, 0.20]
    Zc, eps = 5.0, 0.1

    # cutoff[d][N] = moment-ratio cutoff of the energy distribution
    cutoff = {d: {} for d in dissips}
    pdf_at_fixedN = {}   # for the PDF panel, at the largest N
    Nfix = Ns[-1]

    for d in dissips:
        row = []
        for N in Ns:
            S0 = triangle_ic(N, 0.90 * Zc)
            res = run_sandpile(N=N, eps=eps, Zc=Zc, n_iter=iters[N],
                               seed=2, S0=S0, dissip=d)
            E, P, T = measure_avalanches(res['disp'][warms[N]:])
            cutoff[d][N] = cutoff_moment(E)
            row.append(cutoff[d][N])
            if N == Nfix:
                pdf_at_fixedN[d] = logbin_pdf(E)
        log("  dissip=%.2f : cutoff(E) by N %s = %s"
            % (d, Ns, ["%.3g" % v for v in row]))

    # quantify scaling: slope of log cutoff vs log N for each dissipation
    log("\n[cutoff scaling exponent  d log(cutoff) / d log(N)]")
    log("  (conservative SOC ~ 2.0; a value -> 0 means the cutoff saturates,")
    log("   i.e. a characteristic avalanche size appears and criticality is lost)")
    slopes = {}
    for d in dissips:
        cv = np.array([cutoff[d][N] for N in Ns], dtype=float)
        # use the two largest N for the asymptotic slope (where saturation shows)
        sl_all = np.polyfit(np.log10(Ns), np.log10(cv), 1)[0]
        sl_big = (np.log10(cv[-1]) - np.log10(cv[-2])) / (np.log10(Ns[-1]) - np.log10(Ns[-2]))
        slopes[d] = (sl_all, sl_big)
        log("  dissip=%.2f : full-range slope %.2f, large-N slope %.2f"
            % (d, sl_all, sl_big))

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    cmap = plt.cm.cool(np.linspace(0, 1, len(dissips)))
    for c, d in zip(cmap, dissips):
        cv = [cutoff[d][N] for N in Ns]
        ax[0].loglog(Ns, cv, "o-", color=c,
                     label="dissip=%.2f (slope %.2f)" % (d, slopes[d][0]))
    ax[0].set_xlabel("system size N"); ax[0].set_ylabel("avalanche-energy cutoff")
    ax[0].set_title("Cutoff vs size: conservative scales (~N^2),\n"
                    "dissipative saturates -> criticality lost")
    ax[0].legend(fontsize=8)

    for c, d in zip(cmap, dissips):
        ce, de = pdf_at_fixedN[d]
        if ce.size:
            ax[1].loglog(ce, de, "-", color=c, label="dissip=%.2f" % d)
    ax[1].set_xlabel("avalanche energy E"); ax[1].set_ylabel("PDF(E)")
    ax[1].set_title("Energy PDF at N=%d: dissipation cuts the power law\n"
                    "at a characteristic size" % Nfix)
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "sandpile_dissipation.png")
    fig.savefig(p, dpi=130)
    plt.close(fig)
    log("\nsaved %s" % os.path.relpath(p, ROOT))

    log("\nVerdict: conservation IS necessary for true SOC if the d=0 cutoff")
    log("  scales (~N^2) while every d>0 cutoff saturates to N-independence.")
    with open(os.path.join(OUTDIR, "sandpile_dissipation.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
