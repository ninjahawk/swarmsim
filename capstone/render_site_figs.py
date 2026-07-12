"""Re-render the capstone site figures per the locked manifest (plan section 4).

Each source script runs untouched in a subprocess under the shared style
(_shim_run.py + figstyle.py); its figure writes are redirected into
docs/figures/ so the repo figures/ archive is never overwritten. Every script
gets one try with a hard 15-minute budget (the plan's fallback rule); on
timeout / error / missing output the committed archive PNG is copied into
docs/figures/ instead and the figure is harmonized by display treatment only.

Run from repo root:  python capstone/render_site_figs.py
Writes capstone/render_report.txt; per-script logs go to capstone/render_logs/.
"""

import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = os.path.join(ROOT, "docs", "figures")
ARCHIVE = os.path.join(ROOT, "figures")
LOGDIR = os.path.join(ROOT, "capstone", "render_logs")
SHIM = os.path.join(ROOT, "capstone", "_shim_run.py")
BUDGET_S = 900          # the locked >15-min fallback rule
WORKERS = 3

# (manifest #, source script, primary png, extra pngs also produced)
MANIFEST = [
    ("9  many-wrongs (F82)", "collective/many_wrongs.py", "many_wrongs_1.png", []),
    ("10 misinformation (F85)", "collective/misinformation.py", "misinformation_1.png", []),
    ("11 collective escape (F70)", "predator/collective_escape.py", "collective_escape_1.png", []),
    ("3  1-D anchor (S16)", "sandpile/geometry1d.py", "sandpile_geometry1d.png", []),
    ("14 OFC conservation FSS (E2)", "earthquake/ofc_fss.py", "ofc_fss.png", []),
    ("15 OFC domains (E3)", "earthquake/ofc_quasiperiodic.py", "ofc_domains.png", ["ofc_quasiperiodic.png"]),
    ("16 OFC prediction (E7)", "earthquake/ofc_predict.py", "ofc_predict.png", []),
    ("8  repose out-of-sample (S21)", "sandpile/repose_scaling_L1024.py", "sandpile_repose_scaling_L1024.png", []),
    ("1  filament geometry (S14)", "sandpile/geometry2d.py", "sandpile_geometry.png", []),
    ("7  Manna placement (S22)", "sandpile/manna.py", "sandpile_manna.png", []),
    ("12 escape invasion (F88)", "evolution/escape_invasion.py", "escape_invasion_1.png", []),
    ("13 alignment non-invasion (F94)", "evolution/alignment_invasion.py", "alignment_invasion_1.png", []),
    ("2  stochastic split (S15)", "sandpile/stochastic_split.py", "sandpile_stochastic.png", []),
    ("4  duration closure (S17)", "sandpile/duration_closure.py", "sandpile_duration_closure.png", []),
    ("5  area multifractality (S18)", "sandpile/area_multifractality.py", "sandpile_area_multifractal.png", []),
    ("6  filament fattening (S19)", "sandpile/filament_fattening.py", "sandpile_filament_fattening.png", []),
]


def warm_numba():
    """Compile the fast engines once, serially, so parallel workers hit the
    on-disk numba cache instead of racing to compile."""
    code = (
        "import sys, os; sys.path.insert(0, os.path.join(%r, 'sandpile'));"
        "from sandpile_fast import run_sandpile_fast, run_sandpile2d_fast;"
        "run_sandpile_fast(N=32, eps=0.1, Zc=5.0, n_iter=2000, seed=0);"
        "run_sandpile2d_fast(L=12, eps=0.1, Zc=5.0, n_iter=2000, seed=0);"
        "print('numba warm')" % ROOT
    )
    t = time.time()
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                       capture_output=True, text=True, timeout=600)
    print("numba pre-warm: %.0fs rc=%d %s"
          % (time.time() - t, r.returncode, r.stdout.strip().splitlines()[-1:]))


def fallback(primary, extras, note):
    copied = []
    for png in [primary] + extras:
        src = os.path.join(ARCHIVE, png)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(SITE, png))
            copied.append(png)
    return "FALLBACK (%s; archive PNG kept: %s)" % (note, ", ".join(copied) or "MISSING FROM ARCHIVE")


def run_one(entry):
    label, script, primary, extras = entry
    logp = os.path.join(LOGDIR, os.path.basename(script) + ".log")
    t0 = time.time()
    try:
        with open(logp, "w") as lf:
            r = subprocess.run([sys.executable, SHIM, script], cwd=ROOT,
                               stdout=lf, stderr=subprocess.STDOUT,
                               timeout=BUDGET_S)
        dt = time.time() - t0
        outp = os.path.join(SITE, primary)
        if r.returncode != 0:
            return label, fallback(primary, extras, "exit code %d" % r.returncode), dt
        if not (os.path.exists(outp) and os.path.getmtime(outp) >= t0):
            return label, fallback(primary, extras, "no fresh output"), dt
        return label, "RENDERED", dt
    except subprocess.TimeoutExpired:
        return label, fallback(primary, extras, ">15 min, per locked rule"), time.time() - t0
    except Exception as e:  # noqa: BLE001
        return label, fallback(primary, extras, "runner error %s" % e), time.time() - t0


def main():
    os.makedirs(SITE, exist_ok=True)
    os.makedirs(LOGDIR, exist_ok=True)
    warm_numba()
    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for res in ex.map(run_one, MANIFEST):
            results.append(res)
            print("  %-34s %-14s (%.0fs)" % (res[0], res[1].split(" (")[0], res[2]))
    lines = ["capstone site-figure render report  (budget %ds/script, %d workers)"
             % (BUDGET_S, WORKERS), ""]
    n_ok = 0
    for label, status, dt in results:
        n_ok += status == "RENDERED"
        lines.append("%-34s  %6.0fs  %s" % (label, dt, status))
    lines.append("")
    lines.append("%d/%d re-rendered, %d fallback; total wall %.0fs"
                 % (n_ok, len(results), len(results) - n_ok, time.time() - t0))
    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(ROOT, "capstone", "render_report.txt"), "w") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
