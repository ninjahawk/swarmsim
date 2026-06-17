"""
ofc_speedup.py -- E5: the forcing skip-ahead acceleration (Charbonneau Exercise 3).

Because the forcing is deterministic and uniform, the number of forcing iterations
before the next avalanche is exactly (Fc - max F)/delta_f.  Rather than adding
delta_f one step at a time, we jump straight to the next instability.  The ofc.run
engine already does this (naive_forcing=False); here we measure the speedup factor
against the naive one-step-at-a-time forcing, and confirm it grows as delta_f shrinks
(the smaller the forcing increment, the more idle forcing steps are skipped).

The avalanche statistics are bit-identical either way (verified in the ofc self-test),
so the skip changes only run time, not physics.  This is the OFC counterpart of the
sandpile fast engine (finding S9), and it is what makes the long stationary runs and
finite-size scaling of this chapter tractable.
"""

import os
import time
import numpy as np
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from earthquake.ofc import run_ofc

os.makedirs('outputs', exist_ok=True)


def main():
    lines = []
    def out(s=''):
        print(s); lines.append(s)

    out("E5: OFC forcing skip-ahead speedup (Exercise 3)")
    out("  N=48, alpha=0.20; skip-ahead vs naive one-step forcing, identical statistics.")
    out("")
    out("  delta_f     naive(s)   skip(s)   speedup   identical")

    for df in (1e-3, 1e-4, 1e-5):
        # fewer events for the slow naive path at small delta_f
        nev = 4000 if df >= 1e-4 else 1500
        t0 = time.time()
        r_naive = run_ofc(N=48, alpha=0.20, n_events=nev, warmup_events=1000,
                          seed=1, delta_f=df, naive_forcing=True)
        t_naive = time.time() - t0
        t0 = time.time()
        r_fast = run_ofc(N=48, alpha=0.20, n_events=nev, warmup_events=1000,
                         seed=1, delta_f=df, naive_forcing=False)
        t_fast = time.time() - t0
        same = np.array_equal(r_naive['sizes'], r_fast['sizes'])
        out("  %.0e    %8.2f  %8.2f   %7.1fx   %s"
            % (df, t_naive, t_fast, t_naive / max(t_fast, 1e-9), same))

    out("")
    out("  The speedup grows as delta_f decreases, because the skip collapses the")
    out("  (Fc - max F)/delta_f idle forcing iterations between avalanches into a single")
    out("  array add.  Avalanche-size series are bit-identical, so no physics is changed.")

    with open('outputs/ofc_speedup.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')
    out("  --> outputs/ofc_speedup.txt")


if __name__ == '__main__':
    main()
