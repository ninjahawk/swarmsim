# Summer Research — Claude Context

## Project
PHY 351 Independent Summer Research  
Professor contact: ian@ianbeatty.com  
Textbook: *Natural Complexity: A Modeling Handbook* by Paul Charbonneau (Princeton University Press, 2017)

## Long-Running Nature of This Project
This is an entire summer-long research project, not a one-and-done task. The coursework
(Chapter 10 flocking model) is a starting point, not the finish line. After the formal
topic is exhausted, the work continues: new extensions, new questions, new experiments,
deeper analysis. Each session builds on the last. When one thread closes, we find the
next one to pull. Treat every conversation as a continuation of ongoing research, not a
fresh isolated task. Always look at findings.md and the open questions before suggesting
what to do next.

## Key Files
- `logs.html` — open in Chrome to view/copy the time log and research log. This is the main interface for logging. Do not replace it with a different solution.
- `README.md` — GitHub repo overview
- `research_log.md` — placeholder, actual log data lives in `logs.html`
- `flocking.py` — core simulation module: buffer zone, vectorized force function, run loop, metrics, animation
- `analysis.py` — validation limiting cases and parameter sweeps
- `predator.py` — single-predator extension with 4 experiments
- `phase_transition.py` — finite-size scaling of solid-to-fluid transition
- `geometry.py` — radius of gyration and aspect ratio analysis
- `multi_predator.py` — multi-predator experiments
- `findings.md` — running notes on all 10 findings
- `report_draft.md` — main research report (Markdown source)
- `build_report.py` — generates report_draft.pdf using reportlab
- `model.py` — **OOP foundation for all new experiments**: `Flock` and `Predator` classes, `flock.evolve()`, `simulate()` helper
- `figures/` — all output PNG figures from simulations

## GitHub
Remote: https://github.com/ninjahawk/Summer_Research.git  
Branch: main  
Share the repo URL with the professor for access — no collaborator invite needed.

## Logging Workflow
All logging is done by editing `logs.html` directly:
- **Time log** — add rows to `#timeBody` as `<tr data-date="YYYY-MM-DD">` with date, hours (decimal), summary
- **Research log** — add objects to the `researchData` array with `date` and `body` fields
- After any update, commit and push to GitHub

## Log Format
Time log hours are decimal rounded to nearest 0.25 (e.g. 30 min = 0.50).  
Research log entries are plain text, concise, first person.  
Header in research log: PHY 351 / Independent Summer Research — do not change.

## Rules from Professor
- Document all AI use in the research log (general purpose, not every prompt)
- The student must understand and own all decisions — "Claude recommended it" is not sufficient
- Share time log with professor via Google Sheets (copy from logs.html), research log via Google Docs

## Google Sheets / Docs
No API integration. Logs are copied from `logs.html` in Chrome and pasted manually into Google Sheets and Google Docs. Time log copies as a table (splits into columns on paste). Research log copies as plain text.

## Topic
**Flocking** — Chapter 10 of Charbonneau (2017), originally by Silverberg et al. (2013) for crowd dynamics.
N agents on a periodic 2D unit square under four forces: repulsion, velocity alignment, self-propulsion, random noise.

## Key Findings (16 total -- see findings.md for full details)
1. Equilibrium cruise speed v_eq = v0 + alpha/mu (exact, not just v0)
2. Solid-to-fluid transition is a smooth crossover, not a true phase transition
3. Flocking prey maintain Phi~1.0 under predator pressure; non-flocking scatter to Phi~0.1
4. Multiple naive predators co-localize at prey CoM -- self-undermining, counterintuitively helps flock
5. Coordinated predators (pred-pred repulsion) spread out but still can't break flock (Phi > 0.92)
6. Encirclement strategy (angular offset targets) achieves Phi=0.77 at n_pred=6 -- first real disruption
7. Encirclement works by flock DIVISION: predators compress flock into coherent sub-flocks that scatter
8. Disruption floor ~Phi=0.67 at n_pred=10 for both N=100 and N=350 -- angular coverage is the key variable

## Default Parameters
N=350, r0=0.005, eps=0.1, rf=0.1, alpha=1.0, v0=1.0, mu=10.0, ramp=0.5, dt=0.01

## Current Status (as of 2026-05-10)
This is week 1 of a 2-month project. Core model and book exercises are complete.
Active research thread: predator strategy hierarchy.

### Completed
- Core model + validation (flocking.py, analysis.py)
- Phase transition finite-size scaling (phase_transition.py)
- Fixed-compactness phase scaling (compactness_phase.py)
- Single-predator extension, 4 experiments (predator.py)
- Flock geometry -- Rg, AR (geometry.py)
- Multi-predator -- naive CoM-chasing, 1-4 predators (multi_predator.py)
- Evasion diagnostic -- predator co-localization mechanism (evasion_analysis.py)
- Coordinated predators -- predator-predator repulsion (coordinated_predators.py)
- Encirclement strategy -- angular offset targeting, radius sweep, threshold (encirclement.py)
- Encirclement scaling -- fixed count vs fixed ratio vs full sweep across N (encirclement_scaling.py)
- Fragmentation analysis -- sub-cluster detection, division vs dissolution (fragmentation.py)
- 16 findings documented (findings.md)
- Report draft written (report_draft.md, gitignored)

### Ready to run (scripts written, not yet executed)
- panic.py -- fraction of erratic agents in calm flock (book Section 10.5)
- predator_sensing.py -- limited sensing radius, search/attack phases

### Planned next
- Intermediate compactness phase search (prof Q2: find cooperative regime between caged and non-interacting)
- Minimum viable flock size (below what N does collective evasion fail?)
- Literature search for novelty assessment
- Active/passive segregation (book Section 10.4)

### Key architectural note
All **new** experiments import from `model.py` (Flock, Predator classes).
Old scripts (predator.py, multi_predator.py, encirclement.py, etc.) remain intact but should
not be imported by new code -- they have no __main__ guard and re-run on import.
`model.py` is self-contained and safe to import.

To regenerate the PDF after any changes to report_draft.md:
  python build_report.py

## .gitignore
Excludes: Natural Complexity Book/ (large PDFs), __pycache__/, *.pyc, Emails From Professor.txt
These files still exist locally but are not tracked by git.

## Git Commit Style
- Include `Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>` in roughly 50% of commits — not every one.
- Use it for significant AI-contributed work; omit for small fixes, logging, and minor tweaks.

## Notes
- All simulation code uses ASCII-only print statements (no Unicode) to avoid cp1252 errors on Windows
- Repulsion force uses masked computation to avoid negative^1.5 RuntimeWarning
- Buffer zone (ghost agents) handles periodic boundary forces; yb is indexed from 0, not N
- report_draft.pdf maps: Fig1=validate_3_flocking_only.png, Fig2+3=phase4_sweeps.png, Fig4=phase_transition_scaling.png, Fig5=predator_2_coherence.png, Fig6=geometry_2_alpha_sweep.png, Fig7=multi_pred_3_summary.png
