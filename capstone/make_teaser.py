"""Teaser composite for the capstone site (plan section 2): "the law and its edge".

Three panels, all recomputed honestly from the validated research code (no
hardcoded data; cached npz anchors are the S11/S22 measured spectra):

  (a) One avalanche from each model at the SAME footprint radius (Rg ~ 14):
      the slope model's one-bond filament vs BTW's and Manna's compact blobs.
      Selection is by a neutral criterion (footprint closest to the target Rg
      within an area band), not hand-picked. Slope/BTW footprints come from the
      S14 dumps (geometry2d.slope_footprints / btw_footprints); Manna from the
      S22 engine (mirrored verbatim, minus bookkeeping, from manna.manna_run).

  (b) Avalanche-AREA moment spectrum D(q): the slope model's D(q) RISES with q
      (means lawful, tails multifractal -- S12/S18), while literal Manna and
      BTW area read flat (single FSS line -- S22/S11 cached spectra). The slope
      curve is re-measured here at L = 64-192 (4 seeds) with the S12 machinery.

  (c) The E7 forecaster protocol re-run end to end (half of E7's event count):
      phase-locked prediction catches ordinary recurrent large events and
      misses the largest decile. Recall numbers annotated are THIS run's.

Run from repo root:  python capstone/make_teaser.py
Writes docs/figures/teaser.png. ASCII-only prints.
"""

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
import figstyle  # noqa: E402
figstyle.apply()

import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, ROOT)                              # earthquake package
sys.path.insert(0, os.path.join(ROOT, "sandpile"))    # sandpile modules
from geometry2d import (slope_footprints, btw_footprints, decode_points,
                        gyration, footprint_metrics)  # noqa: E402
from moment_slope import equilibrated_run, jackknife_Dq  # noqa: E402
from earthquake.ofc import run_ofc  # noqa: E402
from earthquake.ofc_predict import detect_large, autocorr_period  # noqa: E402

OUTDIR = os.path.join(ROOT, "outputs")
SITE = os.path.join(ROOT, "docs", "figures")
os.makedirs(SITE, exist_ok=True)

RG_TARGET = 14.0


# ---------------------------------------------------------------- panel (a)
def manna_footprints(L, n_events, warm, seed, area_min=30):
    """S22's canonical 2-D Manna engine (manna.manna_run mirrored verbatim,
    bookkeeping dropped), capturing each avalanche's toppled-site set."""
    rng = np.random.default_rng(seed)
    h = np.zeros((L, L), np.int64)
    ever = np.zeros((L, L), bool)
    fps = []
    for ev in range(n_events):
        r0 = int(rng.integers(0, L)); c0 = int(rng.integers(0, L))
        h[r0, c0] += 1
        ever[:] = False
        ra, rb, ca, cb = r0, r0 + 1, c0, c0 + 1
        while True:
            sub = h[ra:rb, ca:cb]
            un = sub >= 2
            n_un = int(un.sum())
            if n_un == 0:
                break
            ever[ra:rb, ca:cb] |= un
            rows, cols = np.nonzero(un)
            rows = rows + ra
            cols = cols + ca
            h[rows, cols] -= 2
            for _ in range(2):
                d = rng.integers(0, 4, n_un)
                nr = rows + (d == 0) - (d == 1)
                nc = cols + (d == 2) - (d == 3)
                ok = (nr >= 0) & (nr < L) & (nc >= 0) & (nc < L)
                np.add.at(h, (nr[ok], nc[ok]), 1)
            ra = max(ra - 1, 0); rb = min(rb + 1, L)
            ca = max(ca - 1, 0); cb = min(cb + 1, L)
        if ev >= warm and int(ever.sum()) >= area_min:
            fps.append(np.argwhere(ever))
    return fps


def pick_by_rg(pt_sets, target):
    """Neutral selection: the footprint whose Rg is closest to target."""
    best, best_d, best_rg = None, np.inf, np.nan
    for pts in pt_sets:
        lam1, lam2, _, _ = gyration(pts.astype(float))
        rg = np.sqrt(lam1 + lam2)
        if abs(rg - target) < best_d:
            best, best_d, best_rg = pts, abs(rg - target), rg
    return best, best_rg


def panel_a_data():
    print("[a] slope footprints (S14 dump, L=192, 1 seed)...")
    t = time.time()
    foots, ms, _ = slope_footprints(L=192, warm=12_000_000, window=5_000_000,
                                    n_seeds=1, area_cut=8, fp_cap=4_000_000,
                                    max_dump=300_000)
    print("    %d footprints, mean slope %.2f (%.0fs)" % (len(foots), ms, time.time() - t))
    cand = []
    for (_, bonds, _t) in foots:
        if 20 <= bonds.size <= 90:
            cand.append(decode_points(bonds, 192))
    slope_pts, rg_s = pick_by_rg(cand, RG_TARGET)

    print("[a] BTW footprints (S14 baseline, L=96)...")
    t = time.time()
    bfoots = btw_footprints(L=96, n_events=40_000, warm=12_000, area_cut=8, seed=5)
    print("    %d footprints (%.0fs)" % (len(bfoots), time.time() - t))
    cand = [s.astype(float) for (_, s) in bfoots if 250 <= s.shape[0] <= 1100]
    btw_pts, rg_b = pick_by_rg(cand, RG_TARGET)

    print("[a] Manna footprints (S22 engine, L=96)...")
    t = time.time()
    mfoots = manna_footprints(L=96, n_events=45_000, warm=20_000, seed=11)
    print("    %d footprints (%.0fs)" % (len(mfoots), time.time() - t))
    cand = [s.astype(float) for s in mfoots if 250 <= s.shape[0] <= 1100]
    manna_pts, rg_m = pick_by_rg(cand, RG_TARGET)

    for nm, rg in (("slope", rg_s), ("BTW", rg_b), ("Manna", rg_m)):
        print("    picked %s footprint at Rg = %.1f (target %.0f)" % (nm, rg, RG_TARGET))
        assert abs(rg - RG_TARGET) < 4.0, "no footprint near the target radius"
    return [(slope_pts, figstyle.ACCENT, "slope model", "mass-radius D = 1.00 (S14)"),
            (btw_pts, figstyle.MUTED, "BTW", "D = 2.0 (S14)"),
            (manna_pts, figstyle.SAND, "Manna", "D = 2.07 (S22)")]


def draw_panel_a(ax, trio):
    ext = 27.0
    gap = 62.0
    for k, (pts, color, name, dtxt) in enumerate(trio):
        p = pts - pts.mean(axis=0)
        x0 = k * gap
        ax.scatter(p[:, 1] + x0, p[:, 0], s=4.0, color=color, linewidths=0)
        ax.text(x0, ext + 6.0, name, ha="center", va="top", fontsize=10.5,
                fontweight="bold", color=color)
        ax.text(x0, ext + 13.5, "%s\nA = %d sites" % (dtxt, pts.shape[0]),
                ha="center", va="top", fontsize=8.5, color=figstyle.MUTED)
    # shared scale bar
    ax.plot([-ext, -ext + 20], [-ext - 4, -ext - 4], "-",
            color=figstyle.INK, lw=1.8, solid_capstyle="butt")
    ax.text(-ext + 10, -ext - 7.5, "20 lattice units", ha="center", va="bottom",
            fontsize=8, color=figstyle.MUTED)
    ax.set_xlim(-ext - 6, 2 * gap + ext + 6)
    ax.set_ylim(ext + 26.0, -ext - 12.0)
    # datalim: pad the data window instead of shrinking the axes box, so the
    # title and panel label stay aligned with the other panels
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    ax.set_title("Same radius, different physics:\none avalanche from each model (Rg $\\approx$ 14)")


# ---------------------------------------------------------------- panel (b)
def panel_b_data():
    print("[b] slope-model area moments, S12 machinery at L=64-192, 4 seeds...")
    configs = [(64, 5_000_000, 4_000_000), (96, 8_000_000, 5_000_000),
               (128, 12_000_000, 6_000_000), (192, 18_000_000, 8_000_000)]
    Ls = [c[0] for c in configs]
    q_grid = np.arange(0.5, 5.01, 0.25)
    A_seeds = {}
    for L, warm, window in configs:
        t = time.time()
        As = []
        for sd in range(4):
            _, _, A, ms = equilibrated_run(L, warm, window, seed=3 + sd)
            As.append(A)
        A_seeds[L] = As
        print("    L=%3d: %6d avalanches, mean slope %.2f (%.0fs)"
              % (L, sum(a.size for a in As), ms, time.time() - t))
    _, DqA, DqA_sd = jackknife_Dq(A_seeds, Ls, q_grid)
    manna = np.load(os.path.join(OUTDIR, "sandpile_moments_manna.npz"))
    btw = np.load(os.path.join(OUTDIR, "sandpile_moments_btw.npz"))
    sel = (q_grid >= 1.0) & (q_grid <= 4.0)
    print("    slope area D(q): %.2f at q=1 -> %.2f at q=4 (drift %.2f)"
          % (DqA[np.argmin(np.abs(q_grid - 1))], DqA[np.argmin(np.abs(q_grid - 4))],
             DqA[sel].max() - DqA[sel].min()))
    return q_grid, DqA, DqA_sd, manna, btw


def draw_panel_b(ax, q_grid, DqA, DqA_sd, manna, btw):
    m = q_grid >= 0.75
    ax.fill_between(q_grid[m], (DqA - DqA_sd)[m], (DqA + DqA_sd)[m],
                    color=figstyle.ACCENT, alpha=0.18, lw=0)
    ax.plot(q_grid[m], DqA[m], "o-", color=figstyle.ACCENT, ms=4,
            label="slope model (re-run, L$\\leq$192)")
    qm = np.asarray(manna["q_grid"]); mm = qm >= 0.75
    ax.plot(qm[mm], np.asarray(manna["DqA"])[mm], "s-", color=figstyle.SAND,
            ms=3.5, lw=1.3, label="Manna, measured (S22)")
    if "DqA" in getattr(btw, "files", []):
        qb = np.asarray(btw["q_grid"]); mb = qb >= 0.75
        ax.plot(qb[mb], np.asarray(btw["DqA"])[mb], "^--", color=figstyle.MUTED,
                ms=3.5, lw=1.2, label="BTW, measured (S11)")
    ax.annotate("means\n(q $\\approx$ 1)", xy=(0.95, 1.13), ha="center", va="bottom",
                fontsize=8.5, color=figstyle.MUTED)
    ax.annotate("tails\n(large avalanches)", xy=(4.5, 0.02), xycoords=("data", "axes fraction"),
                ha="center", va="bottom", fontsize=8.5, color=figstyle.MUTED)
    ax.set_xlabel("moment order q")
    ax.set_ylabel("area moment dimension D(q)")
    ax.set_ylim(0.85, 2.45)
    ax.legend(loc="center right", fontsize=8.5)
    ax.set_title("Means are lawful, tails escape:\narea D(q) rises for the slope model only")


# ---------------------------------------------------------------- panel (c)
def panel_c_data(n_events=400_000, warm_events=400_000, seed=0):
    """E7's protocol at E7's full scale, fresh seed. The forecaster predicts the
    recurrent largest event once per ~10k-iter cycle, so its headline skill is
    forecast PRECISION vs the chance rate (E7: 0.68 vs 0.44); the tail question
    is recall on the largest decile of events (E7: 0.09)."""
    print("[c] OFC prediction, E7 protocol re-run (N=128, alpha=0.15, %dk events)..."
          % (n_events // 1000))
    t = time.time()
    r = run_ofc(N=128, alpha=0.15, n_events=n_events, warmup_events=warm_events,
                seed=seed, record_iter=True)
    print("    run done (%.0fs)" % (time.time() - t))
    it, sz = r["iters"], r["sizes"]
    it = it - it[0]
    mid = it[-1] // 2
    tr = it < mid
    it_tr, sz_tr = it[tr], sz[tr]
    it_te, sz_te = it[~tr], sz[~tr]

    thr = 0.20 * sz_tr.max()
    big_tr_t, big_tr_e = detect_large(it_tr, sz_tr, thr)
    P = autocorr_period(it_tr, sz_tr, it_tr[0], it_tr[-1])
    mean_big = big_tr_e.mean()
    big_te_t, big_te_e = detect_large(it_te, sz_te, thr)

    # E7's phase-locked forecaster, verbatim logic (ofc_predict.main)
    pred_t = []
    anchor = big_tr_t[-1]
    tt = anchor + P
    obs_t, obs_e = big_te_t.astype(float), big_te_e.astype(float)
    while tt <= it_te[-1] + P:
        pred_t.append(tt)
        wm = np.abs(obs_t - tt) <= 0.5 * P
        anchor = obs_t[wm][np.argmax(obs_e[wm])] if wm.any() else tt
        tt = anchor + P
    pred_t = np.array(pred_t)

    # greedy prediction->observation matching (E7's evaluate), keeping per-event
    # and per-prediction outcomes so the panel can show them
    WIN = 300
    caught = np.zeros(obs_t.size, bool)
    pred_hit = np.zeros(pred_t.size, bool)
    for k, pt in enumerate(pred_t):
        d = np.abs(obs_t - pt)
        d[caught] = np.inf
        j = int(np.argmin(d))
        if d[j] <= WIN:
            caught[j] = True
            pred_hit[k] = True
    precision = pred_hit.mean()
    test_span = it_te[-1] - it_te[0]
    chance = min(1.0, obs_t.size * 2.0 * WIN / test_span)
    top = obs_e >= np.quantile(obs_e, 0.90)
    rec_top = caught[top].mean() if top.any() else np.nan
    print("    period P=%.0f iter; %d predictions over %d test large events (%d in top decile)"
          % (P, pred_t.size, obs_t.size, int(top.sum())))
    print("    forecast precision (+/-%d iter) = %.2f vs chance %.2f (skill %.1fx)  [E7: 0.68 / 0.44 / 1.5x]"
          % (WIN, precision, chance, precision / chance))
    print("    largest decile: %d/%d caught (recall %.2f)  [E7: 4/43, 0.09]"
          % (int(caught[top].sum()), int(top.sum()), rec_top))
    if not (precision > 1.2 * chance and rec_top < 0.5 * precision):
        print("    WARNING: this re-run does NOT reproduce the E7 precision-vs-tail contrast")
    return dict(it_te=it_te, sz_te=sz_te, obs_t=obs_t, obs_e=obs_e,
                caught=caught, top=top, pred_t=pred_t, P=P, thr=thr,
                precision=precision, chance=chance, rec_top=rec_top,
                n_top_caught=int(caught[top].sum()), n_top=int(top.sum()),
                mean_big=mean_big)


def draw_panel_c(ax, d):
    span = 80_000
    t0 = d["it_te"][0]
    m = d["it_te"] - t0 < span
    ax.plot(d["it_te"][m] - t0, d["sz_te"][m], lw=0.5, color=figstyle.MUTED,
            alpha=0.65, zorder=1)
    pm = (d["pred_t"] - t0 >= 0) & (d["pred_t"] - t0 < span)
    for x in d["pred_t"][pm] - t0:
        ax.axvline(x, color=figstyle.SLATE, lw=1.1, ls=(0, (4, 3)), alpha=0.85, zorder=0)
    ax.plot([], [], color=figstyle.SLATE, lw=1.1, ls=(0, (4, 3)),
            label="forecasts, one per cycle:\nprecision %.2f (chance %.2f)"
            % (d["precision"], d["chance"]))
    om = d["obs_t"] - t0 < span
    hit = om & d["caught"]
    miss = om & ~d["caught"]
    ax.plot(d["obs_t"][hit & ~d["top"]] - t0, d["obs_e"][hit & ~d["top"]], "o",
            color=figstyle.ACCENT, ms=6, label="caught by forecast", zorder=3)
    ax.plot(d["obs_t"][hit & d["top"]] - t0, d["obs_e"][hit & d["top"]], "o",
            color=figstyle.ACCENT, ms=9, mec=figstyle.INK, mew=0.7, zorder=3)
    ax.plot(d["obs_t"][miss & d["top"]] - t0, d["obs_e"][miss & d["top"]], "o",
            mfc="none", mec=figstyle.RUST, mew=2.0, ms=10,
            label="largest decile: %d of %d caught" % (d["n_top_caught"], d["n_top"]),
            zorder=4)
    # typical large events not matched to a forecast: quiet dots (the forecaster
    # only aims at one event per cycle, so these are not failures)
    ax.plot(d["obs_t"][miss & ~d["top"]] - t0, d["obs_e"][miss & ~d["top"]], "o",
            color=figstyle.MUTED, ms=2.6, alpha=0.7, lw=0,
            label="other large events", zorder=2)
    ax.set_xlim(0, span)
    ymax = max(d["obs_e"][om].max(), d["sz_te"][m].max())
    ax.set_ylim(0, 1.14 * ymax)
    ax.set_xlabel("iteration into the unseen half")
    ax.set_ylabel("avalanche size")
    ax.legend(loc="upper left", fontsize=8, ncol=1)
    ax.set_title("Full knowledge of the mechanism:\nthe rhythm forecasts all but the largest")


# ----------------------------------------------------------------------------
def main():
    trio = panel_a_data()
    q_grid, DqA, DqA_sd, manna, btw = panel_b_data()
    ofc = panel_c_data()

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.4),
                             gridspec_kw=dict(width_ratios=[1.18, 0.95, 1.15]))
    draw_panel_a(axes[0], trio)
    draw_panel_b(axes[1], q_grid, DqA, DqA_sd, manna, btw)
    draw_panel_c(axes[2], ofc)
    for ax, letter in zip(axes, "abc"):
        figstyle.panel_label(ax, letter, dx=0.0, dy=1.06)
    fig.tight_layout(w_pad=2.2)
    p = os.path.join(SITE, "teaser.png")
    fig.savefig(p)
    plt.close(fig)
    print("saved %s" % os.path.relpath(p, ROOT))


if __name__ == "__main__":
    main()
