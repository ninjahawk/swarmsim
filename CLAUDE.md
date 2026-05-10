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

## Key Findings
1. Equilibrium cruise speed v_eq = v0 + alpha/mu (exact, not just v0)
2. Solid-to-fluid transition is a smooth crossover (not a true phase transition) — finite-size scaling shows no diverging susceptibility
3. Flocking prey maintain Phi~1.0 under predator pressure; non-flocking scatter to Phi~0.1
4. Multiple predators elongate the flock (AR up to 7.9) but coherence stays near 0.98; evasion distance counterintuitively improves

## Default Parameters
N=350, r0=0.005, eps=0.1, rf=0.1, alpha=1.0, v0=1.0, mu=10.0, ramp=0.5, dt=0.01

## Current Status (as of 2026-05-08)
All research is complete. The full pipeline is done:
- Simulation code written and validated (flocking.py)
- Parameter sweeps and validation figures generated (analysis.py)
- Phase transition finite-size scaling done (phase_transition.py)
- Predator-prey extension with 4 experiments done (predator.py)
- Flock geometry analysis done (geometry.py)
- Multi-predator experiments done (multi_predator.py)
- All 10 findings documented (findings.md)
- Professional lab report written in Markdown (report_draft.md)
- PDF version generated (report_draft.pdf via build_report.py)
- Repo cleaned up: .gitignore excludes book PDFs, __pycache__, email
- Everything committed and pushed to GitHub

To regenerate the PDF after any changes to report_draft.md or figures:
  python build_report.py

## .gitignore
Excludes: Natural Complexity Book/ (large PDFs), __pycache__/, *.pyc, Emails From Professor.txt
These files still exist locally but are not tracked by git.

## Notes
- All simulation code uses ASCII-only print statements (no Unicode) to avoid cp1252 errors on Windows
- Repulsion force uses masked computation to avoid negative^1.5 RuntimeWarning
- Buffer zone (ghost agents) handles periodic boundary forces; yb is indexed from 0, not N
- report_draft.pdf maps: Fig1=validate_3_flocking_only.png, Fig2+3=phase4_sweeps.png, Fig4=phase_transition_scaling.png, Fig5=predator_2_coherence.png, Fig6=geometry_2_alpha_sweep.png, Fig7=multi_pred_3_summary.png
