# Summer Research — Claude Context

## Project
PHY 351 Independent Summer Research  
Student: Nathan Langley  
Professor: Ian Beatty (ian@ianbeatty.com)  
Textbook: *Natural Complexity: A Modeling Handbook* by Paul Charbonneau (Princeton University Press, 2017)

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
Header in research log: PHY 351 / Independent Summer Research / Nathan Langley — do not change.

## Rules from Professor
- Document all AI use in the research log (general purpose, not every prompt)
- Nathan must understand and own all decisions — "Claude recommended it" is not sufficient
- Share time log with professor via Google Sheets (copy from logs.html), research log via Google Docs

## Google Sheets / Docs
No API integration. Nathan copies from `logs.html` in Chrome and pastes manually into Google Sheets and Google Docs. Time log copies as a table (splits into columns on paste). Research log copies as plain text.

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

## Notes
- All simulation code uses ASCII-only print statements (no Unicode) to avoid cp1252 errors on Windows
- Repulsion force uses masked computation to avoid negative^1.5 RuntimeWarning
- Buffer zone (ghost agents) handles periodic boundary forces; yb is indexed from 0, not N
