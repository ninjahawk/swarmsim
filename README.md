# PHY 351 — Independent Summer Research

**Student:** Nathan Langley  
**Advisor:** Prof. Ian Beatty, UNCG  
**Topic:** Emergent Flocking and Collective Evasion in a Force-Based Agent Model  
**Textbook:** *Natural Complexity: A Modeling Handbook* — Charbonneau (2017), Chapter 10

---

## Overview

Computational simulation of a force-based flocking model in which N agents on a periodic 2D unit square interact through four forces: repulsion, velocity alignment, self-propulsion, and random noise. Investigations include parameter sweeps, phase transition analysis, predator-prey extension, flock geometry, and multi-predator dynamics.

---

## Key Files

| File | Description |
|------|-------------|
| `logs.html` | Time log and research log — open in browser to view and copy |
| `data.html` | Raw numerical data from all experiments — open in browser, copy to Google Sheets |
| `report_draft.md` | Full written report (Markdown source) |
| `report_draft.pdf` | Compiled PDF report |
| `findings.md` | Running notes on all 12 findings |
| `flocking.py` | Core model: buffer zone, vectorized forces, run loop, metrics |
| `analysis.py` | Validation limiting cases and parameter sweeps |
| `predator.py` | Single-predator extension with 4 experiments |
| `phase_transition.py` | Finite-size scaling of solid-to-fluid transition |
| `geometry.py` | Radius of gyration and aspect ratio analysis |
| `multi_predator.py` | Multi-predator experiments (1–4 predators) |
| `evasion_analysis.py` | Predator co-localization diagnostic |
| `compactness_phase.py` | Fixed-compactness finite-size scaling |
| `build_report.py` | Generates report_draft.pdf via reportlab |
| `figures/` | All output figures (PNG) |

---

## Key Findings

1. Equilibrium cruise speed is v_eq = v0 + alpha/mu (exact analytical result)
2. Solid-to-fluid transition is a smooth crossover — no true phase transition at any tested compactness
3. Flock forms at very low alignment amplitude (alpha ~ 0.05–0.10)
4. Full model is robust to noise up to eta ~ 10
5. Flocking prey maintain Phi ~ 1.0 under predator pressure; non-flocking scatter to Phi ~ 0.1
6. Evasion distance saturates regardless of predator aggression (collective buffer effect)
7. Larger flocks expose smaller fractions to predator threat (dilution effect)
8. Fixed-compactness scaling confirms crossover persists in both dense and dilute regimes
9. Flock elongates with stronger alignment and under predator pressure
10. Multiple predators maintain coherence while increasing flock elongation
11. Multiple predators co-localize at prey CoM — combined repulsion explains evasion paradox
12. Crossover behavior is general across compactness values, not regime-specific

---

## Default Parameters

N=350, r0=0.005, eps=0.1, rf=0.1, alpha=1.0, v0=1.0, mu=10.0, ramp=0.5, dt=0.01

---

## Reproducing Results

```
python analysis.py          # validation and parameter sweeps
python phase_transition.py  # finite-size scaling
python predator.py          # single-predator experiments
python geometry.py          # flock shape analysis
python multi_predator.py    # multi-predator experiments
python evasion_analysis.py  # evasion diagnostic
python compactness_phase.py # fixed-compactness phase scaling
python build_report.py      # compile PDF report
```

---

## Tools

- **Language:** Python 3 (numpy, matplotlib, reportlab)
- **AI Assistance:** Claude (Anthropic) — used for code generation, debugging, and research guidance. All AI use is documented in the research log.
