"""
ofc_predict.py -- E7: the Grand Challenge, earthquake prediction in the OFC model
(Charbonneau Exercise 6).

The nonconservative OFC model (alpha=0.15) produces quasi-periodic large avalanches
(E3).  The challenge: using the first half of an avalanche time series to learn the
recurrence rhythm and the characteristic large-event size, forecast the timing and
size of the large events (E > 20% of the training maximum) in the second half.  A
"good" forecast gets the timing within +/-100 iterations and the amplitude within
+/-25% of the observed value (book's criteria).  We track hits, misses, and false
alarms, and compare the skill to a chance baseline.

Forecaster (deliberately simple, time-series only -- no peeking at the lattice
state).  From the training half: detect the large events, estimate the recurrence
period P (median spacing of consecutive large events, cross-checked by
autocorrelation), and the mean large-event size.  In the test half we run a
phase-tracking predictor: maintain an expected next-large-event time = (time of last
observed large event) + P, predict a large event there with the training mean size,
and re-anchor on each observed large event.  This asks the real question: does the
rhythm learned in the first half persist into the second?
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from earthquake.ofc import run_ofc

os.makedirs('figures', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

N = 128
ALPHA = 0.15
WIN_TIGHT = 100         # book's timing tolerance (iterations)
WIN_LOOSE = 300
AMP_TOL = 0.25          # +/-25% amplitude tolerance


def detect_large(iters, sizes, thr):
    m = sizes > thr
    return iters[m], sizes[m]


def autocorr_period(iters, sizes, lo_iter, hi_iter, binw=200):
    nb = int((hi_iter - lo_iter) // binw)
    grid = np.zeros(nb)
    b = ((iters - lo_iter) // binw).astype(int)
    keep = (b >= 0) & (b < nb)
    np.add.at(grid, b[keep], sizes[keep])
    g = grid - grid.mean()
    ac = np.correlate(g, g, mode='full')[len(g) - 1:]
    ac = ac / ac[0]
    lo = max(1, int(800 // binw))
    hi = min(len(ac) - 1, int(20000 // binw))
    k = np.argmax(ac[lo:hi])
    return (lo + k) * binw


def evaluate(pred_times, pred_sizes, obs_times, obs_sizes, win, amp_tol):
    """Greedy match predictions to observations within +/-win iterations."""
    obs_times = np.asarray(obs_times, dtype=float)
    used = np.zeros(obs_times.size, dtype=bool)
    hits = 0
    amp_ok = 0
    for pt, ps in zip(pred_times, pred_sizes):
        d = np.abs(obs_times - pt)
        d[used] = np.inf
        j = np.argmin(d) if d.size else None
        if j is not None and d[j] <= win:
            used[j] = True
            hits += 1
            if abs(ps - obs_sizes[j]) <= amp_tol * obs_sizes[j]:
                amp_ok += 1
    false_alarms = pred_times.size - hits
    misses = obs_times.size - hits
    return hits, misses, false_alarms, amp_ok


def main():
    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out("E7: Grand Challenge -- OFC earthquake prediction (Exercise 6)")
    out("  N=%d  alpha=%.2f  delta_f=1e-4.  Forecast large events (>20%% of train max)." % (N, ALPHA))
    out("  Timing tolerance +/-%d iter (book), amplitude tolerance +/-%d%%." % (WIN_TIGHT, int(AMP_TOL * 100)))
    out("")

    # long stationary run, recording the iteration timeline
    r = run_ofc(N=N, alpha=ALPHA, n_events=400000, warmup_events=400000,
                seed=0, record_iter=True)
    it, sz = r['iters'], r['sizes']
    it = it - it[0]
    span = it[-1]
    out("  recorded %d events over %d iterations (%.1f cycles at the recurrence period)"
        % (sz.size, span, span / 4000.0))

    # split into train / test halves by iteration
    mid = span // 2
    tr = it < mid
    te = ~tr
    it_tr, sz_tr = it[tr], sz[tr]
    it_te, sz_te = it[te], sz[te]

    # learn from training half
    emax_tr = sz_tr.max()
    thr = 0.20 * emax_tr
    big_tr_t, big_tr_e = detect_large(it_tr, sz_tr, thr)
    P_spacing = np.median(np.diff(big_tr_t)) if big_tr_t.size > 3 else np.nan
    P_ac = autocorr_period(it_tr, sz_tr, it_tr[0], it_tr[-1])
    # The recurrence CYCLE is the raw autocorrelation period (~10000 iter at alpha=0.15,
    # matching the book's fig.-8.4 raw period ~10960); the inter-large-event spacing is
    # much shorter because several events occur per cycle.  We phase-lock the forecaster
    # to the cycle and predict the recurrent LARGEST event once per cycle.
    P = P_ac if np.isfinite(P_ac) else P_spacing
    # forcing-corrected period for comparison with the book (subtract avalanching fraction).
    # f_av over the RECORDING span (matched ranges): avalanching sweeps / total iterations.
    f_av = r['durations'].sum() / (r['iters'][-1] - r['iters'][0])
    P_corrected = P * (1 - f_av)
    mean_big = big_tr_e.mean()
    out("  training: max E=%d, %d large events, recurrence cycle P=%.0f iter (autocorr; book raw ~10960)"
        % (emax_tr, big_tr_t.size, P_ac))
    out("           forcing-corrected period=%.0f iter (book ~4002 at alpha=0.15); large-event spacing=%.0f"
        % (P_corrected, P_spacing))

    # observed large events in the test half
    big_te_t, big_te_e = detect_large(it_te, sz_te, thr)
    out("  test: %d large events to forecast" % big_te_t.size)
    out("")

    # phase-locked predictor across the test half: predict the recurrent LARGEST event
    # once per cycle, re-anchoring each cycle on the BIGGEST observed event in the window
    # (online forecasting -- we see events as they happen and use them to set the phase).
    pred_t, pred_s = [], []
    anchor = big_tr_t[-1]            # last large event seen in training
    t = anchor + P
    test_end = it_te[-1]
    obs_t, obs_e = big_te_t.astype(float), big_te_e.astype(float)
    while t <= test_end + P:
        pred_t.append(t)
        pred_s.append(mean_big)
        win_mask = np.abs(obs_t - t) <= 0.5 * P
        if win_mask.any():
            cand_t, cand_e = obs_t[win_mask], obs_e[win_mask]
            anchor = cand_t[np.argmax(cand_e)]      # lock onto the cyclic peak
        else:
            anchor = t
        t = anchor + P
    pred_t = np.array(pred_t); pred_s = np.array(pred_s)
    test_span = it_te[-1] - it_te[0]

    out("  forecaster predicts %d large events (one per ~%.0f-iter cycle) over %d test events"
        % (pred_t.size, P, big_te_t.size))
    for win in (WIN_TIGHT, WIN_LOOSE):
        hits, misses, fa, amp_ok = evaluate(pred_t, pred_s, big_te_t, big_te_e, win, AMP_TOL)
        prec = hits / max(1, pred_t.size)
        # chance precision: prob a random predicted time lands within +/-win of any large event
        chance_prec = min(1.0, big_te_t.size * 2.0 * win / test_span)
        skill = prec / chance_prec if chance_prec > 0 else np.nan
        out("  window +/-%3d iter: predictions=%d  hits=%d  false_alarms=%d | precision=%.2f (chance %.2f, skill %.1fx) | amplitude-correct=%d/%d"
            % (win, pred_t.size, hits, fa, prec, chance_prec, skill, amp_ok, hits))

    # recall of the genuinely LARGEST events (top decile by size) at the loose window
    top_thr = np.quantile(big_te_e, 0.90)
    top_t = big_te_t[big_te_e >= top_thr].astype(float)
    caught = 0
    for tt in top_t:
        if np.any(np.abs(pred_t - tt) <= WIN_LOOSE):
            caught += 1
    out("  largest events (top 10%% by size, n=%d): %d caught within +/-%d iter (recall %.2f)"
        % (top_t.size, caught, WIN_LOOSE, caught / max(1, top_t.size)))

    # ---------------------------------------------------------------- figure
    fig, ax = plt.subplots(figsize=(11, 4.5))
    show = it_te - it_te[0] < 80000
    ax.plot((it_te[show] - it_te[0]), sz_te[show], lw=0.6, color='gray', label='avalanche size')
    bm = big_te_t - it_te[0] < 80000
    ax.plot((big_te_t[bm] - it_te[0]), big_te_e[bm], 'o', color='tab:red', ms=5, label='large events (observed)')
    pm = (pred_t - it_te[0] >= 0) & (pred_t - it_te[0] < 80000)
    for x in (pred_t[pm] - it_te[0]):
        ax.axvline(x, color='tab:blue', lw=0.8, alpha=0.4)
    ax.axvline(np.nan, color='tab:blue', lw=0.8, alpha=0.4, label='forecast times')
    ax.set_xlabel('iteration into test half'); ax.set_ylabel('avalanche size E')
    ax.set_title('OFC earthquake prediction: phase-tracking forecast vs observed large events (alpha=%.2f)' % ALPHA)
    ax.legend(fontsize=8, loc='upper right'); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/ofc_predict.png', dpi=120)
    plt.close()
    out("")
    out("  --> figures/ofc_predict.png")

    with open('outputs/ofc_predict.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    out("  --> outputs/ofc_predict.txt")


if __name__ == '__main__':
    main()
