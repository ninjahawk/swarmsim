<div align="center">

<img src="figures/demo.gif" width="480"/>

# Emergent Flocking & Collective Evasion

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-scientific_computing-013243?logo=numpy&logoColor=white)](https://numpy.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-visualization-11557c)](https://matplotlib.org)
[![Course](https://img.shields.io/badge/PHY_351-UNCG-gold)](https://uncg.edu)
[![Status](https://img.shields.io/badge/status-active-brightgreen)](https://github.com/ninjahawk/swarmsim)

**PHY 351 — Independent Summer Research &nbsp;|&nbsp; UNCG**

A computational study of force-based flocking agents on a periodic 2D domain, extended with predator-prey dynamics. Based on Charbonneau (2017), Ch. 10.

[**▶ Open Interactive Demo**](https://htmlpreview.github.io/?https://github.com/ninjahawk/swarmsim/blob/main/sim_demo.html) &nbsp;·&nbsp; [**📋 Research Log**](https://htmlpreview.github.io/?https://github.com/ninjahawk/swarmsim/blob/main/logs.html)

</div>

---

## Model

N agents move on a periodic unit square under four forces each timestep:

| Force | Description |
|-------|-------------|
| **Repulsion** | Short-range — keeps agents from overlapping (range 2r₀) |
| **Alignment** | Drives velocity toward mean of neighbors within r_f |
| **Self-propulsion** | Corrects speed toward target v₀ |
| **Noise** | Uniform random perturbation in [−η, η] |

**Key analytical result:** equilibrium cruise speed is `v_eq = v₀ + α/μ` — not `v₀`. Derived from force balance in an aligned flock; verified experimentally to within 0.002 across four α values.

**Default parameters:** N=350, r₀=0.005, ε=0.1, r_f=0.1, α=1.0, v₀=1.0, μ=10.0, η=0.5, dt=0.01

---

## Code Architecture

The codebase has two layers.

**Core layer (`model.py`)** — object-oriented foundation for all current and future experiments.

```
Flock          — holds prey positions, velocities, and all physics parameters
                 flock.evolve(predators=[...]) advances one timestep
Predator       — configurable predator agent; strategy='naive'|'encircle',
                 coord_alpha for predator-predator repulsion, enc_radius for encirclement
simulate()     — runs a full experiment, returns Phi and distance timeseries
```

New experiments import from `model.py`. To add a new experiment: import `Flock`, `Predator`, and `simulate` from `model.py`, set parameters, call `simulate()`.

**Legacy layer (`flocking.py`)** — procedural implementation that predates the OOP refactor. The earlier experiment scripts (`predator.py`, `multi_predator.py`, `encirclement.py`, etc.) import from here. They are retained intact for reproducibility and are not broken by the `model.py` refactor.

### Investigation Progression

The experiments follow a logical arc, each motivated by the result before it:

| Stage | Question | Scripts |
|-------|----------|---------|
| **1. Baseline** | Does the model reproduce expected flocking behavior? | `flocking.py`, `analysis.py` |
| **2. Phase transition** | Is the solid-to-fluid transition a true phase transition? | `phase_transition.py`, `compactness_phase.py`, `compactness_search.py` |
| **3. Geometry** | What shape does the flock take? | `geometry.py` |
| **4. Single predator** | Does flocking help prey survive? | `predator.py` |
| **5. Predator strategy hierarchy** | How many and how coordinated do predators need to be to break the flock? | `multi_predator.py` → `evasion_analysis.py` → `coordinated_predators.py` → `encirclement.py` → `encirclement_scaling.py` → `fragmentation.py` |
| **6. Internal disruption** | Does panic from within disrupt the flock more or less than a predator? | `panic.py` *(ready to run)* |
| **7. Realistic predator** | What changes when the predator has limited sensing range? | `predator_sensing.py` *(ready to run)* |

---

## Results

### Flock Formation & Coherence

<div align="center">
<img src="figures/phase4_sweeps.png" width="780"/>
</div>

Flock forms reliably above α ≈ 0.1. With all forces active, the order parameter Φ stays above 0.97 up to noise η ≈ 10, then collapses near η ≈ 20. The alignment force makes the system dramatically more noise-resistant than the repulsion-only case.

---

### Phase Transition Analysis

<div align="center">
<img src="figures/phase_transition_scaling.png" width="780"/>
</div>

Finite-size scaling across N = 25–200 (with compactness fixed via r₀ = √(C/πN)) shows KE/N curves are N-independent and susceptibility χ = N·Var(KE/N) increases monotonically at every tested compactness value (C = 0.10–0.78). No diverging peak appears anywhere. The solid-to-fluid transition is a **smooth crossover** throughout — a consequence of the soft repulsion potential used in this model. A true phase transition would require hard-core exclusion.

---

### Predator Strategy Hierarchy

<div align="center">
<img src="figures/predator_2_coherence.png" width="560"/>
</div>

Flocking prey maintain Φ ≈ 1.0 under sustained predator pressure; non-flocking prey scatter to Φ ≈ 0.1 almost immediately. A minimum evasion buffer distance (~0.10) persists regardless of predator aggression.

With multiple predators, flock coherence remains high (Φ > 0.97) because all predators independently target the same center of mass and co-localize (measured separation ~0.001). This self-undermining behavior was confirmed in `evasion_analysis.py`.

Adding predator-predator repulsion (coordinated predators) successfully spreads predators out — separation rises to 0.29 — but Φ never drops below 0.92. The flock absorbs distributed pressure from multiple directions without breaking.

<div align="center">
<img src="figures/encircle_3_breaking_threshold.png" width="640"/>
</div>

**Encirclement** (each predator assigned a fixed compass angle, targeting CoM offset by R_enc in that direction) is the first strategy to substantially disrupt the flock. At n_pred = 6, Φ drops to 0.77. The minimum predator-prey distance falls to 0.050 vs 0.105 for naive predators.

<div align="center">
<img src="figures/frag_2_cluster_stats.png" width="640"/>
</div>

This Φ = 0.77 does not represent dissolution. `fragmentation.py` shows that encirclement **divides** the flock: predators compress agents spatially (60 clusters → 24 larger clusters) while splitting them directionally — each sub-flock escapes through a gap between predators and remains internally coherent (sub-flock Φ = 0.997). This is analogous to wolf-pack herding.

---

## Key Findings

| # | Finding |
|---|---------|
| 1 | Cruise speed is v_eq = v₀ + α/μ (exact analytical result) |
| 2 | Repulsion-only solid-to-fluid transition appears continuous |
| 3 | Flock forms at very low alignment amplitude (α ≈ 0.05–0.10) |
| 4 | Full model robust to noise up to η ≈ 10; collapses near η ≈ 20 |
| 5 | Flocking prey maintain Φ ≈ 1.0 under predator pressure; non-flocking scatter to Φ ≈ 0.1 |
| 6 | Evasion floor: minimum predator-prey distance saturates regardless of predator speed |
| 7 | Dilution effect: larger flocks expose smaller fractions to predator threat |
| 8 | Fixed-compactness finite-size scaling confirms crossover — not phase transition — at both C=0.78 and C=0.10 |
| 9 | Flock elongates with stronger alignment force and under predator pressure |
| 10 | Multiple naive predators maintain high flock coherence while increasing elongation |
| 11 | Multiple predators co-localize at prey CoM — explains why evasion distance increases with more predators |
| 12 | Fixed-compactness crossover is identical in dense and dilute regimes — not regime-specific |
| 13 | Coordinated predators (with mutual repulsion) spread out spatially but still cannot break the flock (Φ > 0.92) |
| 14 | Encirclement — predators from equally spaced angles — achieves Φ = 0.77 at n=6, the first substantial disruption |
| 15 | Encirclement threshold does not scale simply with N; both N=100 and N=350 converge to Φ ≈ 0.67 at n_pred=10 |
| 16 | Φ = 0.77 reflects flock **division**, not dissolution — encirclement compresses agents into coherent sub-flocks escaping in different directions |
| 17 | No phase transition at any intermediate compactness (C = 0.15–0.60); crossover is universal in this model due to soft repulsion potential |

Full documentation, evidence, and figures for each finding: [`findings.md`](findings.md)

---

## Repository

### Core

| File | Description |
|------|-------------|
| `model.py` | **OOP foundation** — `Flock`, `Predator` classes; `flock.evolve()`; `simulate()` helper. Import this for new experiments. |
| `flocking.py` | Procedural core — buffer zone, vectorized forces, run loop, order parameter. Used by legacy experiment scripts. |

### Experiments (legacy — procedural, import from `flocking.py`)

| File | Investigation |
|------|--------------|
| `analysis.py` | Validation limiting cases and parameter sweeps (Findings 1–4) |
| `phase_transition.py` | Finite-size scaling of repulsion-only transition (Finding 8) |
| `compactness_phase.py` | Fixed-compactness finite-size scaling at C=0.10 and C=0.78 (Finding 12) |
| `compactness_search.py` | Phase transition search across C = 0.15–0.60 (Finding 17) |
| `geometry.py` | Radius of gyration and aspect ratio analysis (Finding 9) |
| `predator.py` | Single-predator extension — 4 experiments (Findings 5–7) |
| `multi_predator.py` | Multiple naive predators (Finding 10) |
| `evasion_analysis.py` | Predator co-localization diagnostic (Finding 11) |
| `coordinated_predators.py` | Predator-predator repulsion — coordination strength sweep and flock-breaking threshold (Finding 13) |
| `encirclement.py` | Encirclement strategy — radius sweep and flock-breaking threshold (Finding 14) |
| `encirclement_scaling.py` | Encirclement threshold vs flock size N (Finding 15) |
| `fragmentation.py` | Flock fragmentation analysis — cluster detection, sub-flock coherence (Finding 16) |

### Experiments (new — import from `model.py`)

| File | Investigation |
|------|--------------|
| `panic.py` | Panic dynamics — fraction of erratic agents disrupting a calm flock *(written, not yet run)* |
| `predator_sensing.py` | Limited predator sensing range — search/attack phases, hunting cycles *(written, not yet run)* |

### Supporting files

| File | Description |
|------|-------------|
| `make_demo.py` | Generates `figures/demo.gif` for this README |
| `build_report.py` | Generates `report_draft.pdf` from `report_draft.md` using reportlab |
| `sim_demo.html` | Interactive browser simulation (open locally or via htmlpreview link above) |
| `logs.html` | Time log and research log — open in browser |
| `findings.md` | Running notes on all 17 findings with figures |
| `report_draft.md` | Full research report in Markdown |

---

## Run

```bash
# Validation and parameter sweeps
python analysis.py

# Phase transition analysis
python phase_transition.py
python compactness_phase.py
python compactness_search.py

# Geometry
python geometry.py

# Predator strategy hierarchy (run in order)
python predator.py
python multi_predator.py
python evasion_analysis.py
python coordinated_predators.py
python encirclement.py
python encirclement_scaling.py
python fragmentation.py

# Upcoming experiments
python panic.py
python predator_sensing.py
```

Open `sim_demo.html` in a browser for a real-time interactive simulation with adjustable parameters.

---

## Tools

- **Language:** Python 3 — numpy, matplotlib, reportlab
- **AI assistance:** Claude (Anthropic) — code generation, debugging, research guidance. All AI use documented in research log.

---

*Charbonneau, P. (2017). Natural Complexity: A Modeling Handbook. Princeton University Press.*  
*Silverberg et al. (2013). Collective motion of humans in mosh and circle pits. Physical Review Letters, 110, 228701.*
