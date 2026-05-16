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

### Repository Layout

```
root/        flocking.py  model.py  predator.py  geometry.py  multi_predator.py  encirclement.py  analysis.py
predator/    predator strategy experiments
contagion/   panic, epidemic, and vaccination experiments
phase/       phase-transition and statistical-mechanics experiments
3d/          three-dimensional extension experiments
figures/     output PNG figures
outputs/     captured text/log output from runs
```

Experiment scripts in subfolders automatically add the project root to `sys.path` so they can import the core library files above.

### Investigation Progression

The experiments follow a logical arc, each motivated by the result before it:

| Stage | Question | Scripts |
|-------|----------|---------|
| **1. Baseline** | Does the model reproduce expected flocking behavior? | `flocking.py`, `analysis.py` |
| **2. Phase transition** | Is the solid-to-fluid transition a true phase transition? | `phase/phase_transition.py`, `phase/compactness_phase.py`, `phase/compactness_search.py` |
| **3. Geometry** | What shape does the flock take? | `geometry.py` |
| **4. Single predator** | Does flocking help prey survive? | `predator.py` |
| **5. Predator strategy hierarchy** | How many and how coordinated do predators need to be to break the flock? | `multi_predator.py` → `predator/evasion_analysis.py` → `predator/coordinated_predators.py` → `encirclement.py` → `predator/encirclement_scaling.py` → `predator/fragmentation.py` |
| **6. Encirclement limits** | Does the encirclement floor scale with N? Is it stable long-term? Does a gap help the flock? | `predator/large_N_encirclement.py` → `predator/renc_scaling.py` → `predator/long_encirclement.py` → `predator/encirclement_gap.py` |
| **7. Minimum viable size** | Below what N does collective evasion fail? | `predator/min_flock_size.py` |
| **8. Reversibility** | Does encirclement damage reverse after predator removal? | `predator/reunion.py` |
| **9. Realistic predator** | What changes with limited sensing range? | `predator/predator_sensing.py` |
| **10. Internal panic** | Does panic from within disrupt the flock? | `contagion/panic.py` |
| **11. Contagion** | What if panic spreads by contact? (SI and SIS models) | `contagion/panic_contagion.py` → `contagion/contagion_sis.py` |
| **12. Hybrid stressors** | How do predation and contagion interact? | `contagion/hybrid_stressors.py` → `contagion/hybrid_sis.py` → `contagion/critical_shift.py` → `contagion/outbreak_removal.py` |
| **13. Herd immunity** | How many immune agents are needed to quench an outbreak? | `contagion/herd_immunity.py` → `contagion/targeted_immunity.py` |
| **14. Segregation** | Does a mixed-speed population spatially separate? | `contagion/segregation.py` → `contagion/segregation_alpha.py` |
| **15. Adaptive predators** | Do predators that track flock geometry outperform fixed strategy? | `predator/adaptive_encirclement.py` |
| **16. Vaccination strategies** | Can targeting high-degree or spatially-distributed agents reduce herd-immunity threshold? | `contagion/targeted_immunity.py` → `contagion/spatial_vaccination.py` |
| **17. Phase transition mechanism** | Why does the model lack a true phase transition? Is it the soft repulsion, or something else? | `phase/hard_repulsion.py` → `phase/langevin_repulsion.py` → `phase/langevin_hexatic.py` |
| **18. 3D extension** | Does flocking and the v_eq result hold in three dimensions? Extended noise sweep | `3d/flocking3d.py` → `3d/flocking3d_noise.py` |

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

40 findings documented — full evidence and figures: [`findings.md`](findings.md)

**Selected highlights:**

| # | Finding |
|---|---------|
| 1 | Cruise speed is **v_eq = v₀ + α/μ** (exact analytical result, not just v₀) |
| 2 | Solid-to-fluid transition is a smooth crossover at all compactness values — no true phase transition |
| 5 | Flocking prey maintain Φ ≈ 1.0 under predator pressure; non-flocking scatter to Φ ≈ 0.1 |
| 11 | Multiple naive predators co-localize at CoM, self-undermining — paradoxically helps flock |
| 14 | Encirclement (fixed-angle targets) is the **only strategy** to substantially disrupt the flock: Φ = 0.77 at n=6 |
| 16 | Encirclement causes flock **DIVISION** (coherent sub-flocks heading in different directions), not dissolution |
| 22 | Encirclement damage is **fully reversible**: sub-flocks reunite within ~10 time units of predator removal |
| 25 | SIS contagion has a clean epidemic threshold at β/γ ≈ 1; flock coherence tracks it |
| 30 | Herd-immunity threshold in the flock is **~2× mean-field** prediction due to spatial clustering |
| 31 | Encirclement scaling collapses on R_enc/Rg; optimal at R_enc/Rg ≈ 0.5 — strategy is **size-invariant** |
| 32 | Long-time encirclement is **intermittent**: Φ oscillates 0.4–0.95 in a sustained merge/split cycle |
| 33 | Incomplete encirclement (1 gap) is **more disruptive** than full ring; no global escape-route detection |
| 34 | Predator removal after encirclement+SIS: kinematic damage reverses in ~10 tu, epidemic persists ~100+ tu |
| 35 | **Adaptive R_enc = 0.5×Rg** outperforms fixed radius: Φ 0.778→0.713, high-coherence dwell time −34% |
| 36 | **Targeted vaccination null**: hub-targeting fails; kinematic reorganization restores hub positions |
| 37 | **Spatial vaccination null**: farthest-point spatial sampling fails; kinematic mixing scrambles positions |
| 38 | **Repulsion exponent null**: sweeping n=1.5→12 gives identical crossover; non-equilibrium driving is cause |
| 39 | **Langevin thermalizes correctly** (KE/N=kT to 1%), but chi_KE cannot detect KTHNY structural melting |
| 40 | **Hexatic |ψ₆| confirms n=1.5 cannot crystallize**: flat at ~0.4 across all kT; soft repulsion prevents lattice formation |

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
| `phase_transition.py` | Finite-size scaling of repulsion-only transition |
| `compactness_phase.py` | Fixed-compactness finite-size scaling at C=0.10 and C=0.78 |
| `compactness_search.py` | Phase transition search across C = 0.15–0.60 |
| `geometry.py` | Radius of gyration and aspect ratio analysis |
| `predator.py` | Single-predator extension — 4 experiments |
| `multi_predator.py` | Multiple naive predators |
| `evasion_analysis.py` | Predator co-localization diagnostic |
| `coordinated_predators.py` | Predator-predator repulsion experiments |
| `encirclement.py` | Encirclement strategy — radius sweep and flock-breaking threshold |
| `encirclement_scaling.py` | Encirclement threshold vs flock size N |
| `fragmentation.py` | Flock fragmentation analysis — cluster detection, sub-flock coherence |

### Experiments (new — import from `model.py`)

| File | Investigation |
|------|--------------|
| `panic.py` | Panic dynamics — fraction of erratic agents (Finding 18) |
| `predator_sensing.py` | Limited predator sensing range (Finding 19) |
| `panic_contagion.py` | SI panic contagion — no recovery (Finding 20) |
| `contagion_sis.py` | SIS contagion with recovery rate γ; epidemic threshold sweep (Finding 25) |
| `reunion.py` | Sub-flock reunion after predator removal (Finding 22) |
| `min_flock_size.py` | Minimum N for collective coherence and evasion (Finding 21) |
| `hybrid_stressors.py` | Combined predation + SI contagion (Finding 23) |
| `hybrid_sis.py` | Sub-threshold SIS + encirclement; compression amplification (Finding 26) |
| `segregation.py` | Active/passive v0 contrast — null segregation result (Finding 24) |
| `segregation_alpha.py` | Alpha-contrast segregation with local-purity diagnostic (Finding 27) |
| `large_N_encirclement.py` | Encirclement at N=350, 700, 1000 (Finding 28) |
| `critical_shift.py` | Beta sweep with/without encirclement; threshold shift (Finding 29) |
| `herd_immunity.py` | Immune sub-population sweep at supercritical SIS (Finding 30) |
| `renc_scaling.py` | R_enc sweep at N=350 and N=1000; collapse on R_enc/Rg (Finding 31) |
| `long_encirclement.py` | 30000-step encirclement; merge/split dynamics (Finding 32) |
| `encirclement_gap.py` | Incomplete encirclement and gap detection (Finding 33) |
| `outbreak_removal.py` | Encirclement+SIS then predator removal; epidemic persistence (Finding 34) |
| `targeted_immunity.py` | Targeted vs random vaccination at supercritical SIS |
| `adaptive_encirclement.py` | Adaptive R_enc = 0.5×Rg vs fixed R_enc |

### Supporting files

| File | Description |
|------|-------------|
| `make_demo.py` | Generates `figures/demo.gif` for this README |
| `build_report.py` | Generates `report_draft.pdf` from `report_draft.md` using reportlab |
| `sim_demo.html` | Interactive browser simulation (open locally or via htmlpreview link above) |
| `logs.html` | Time log and research log — open in browser |
| `findings.md` | Running notes on all 34 findings with figures |
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

# Predator strategy hierarchy
python predator.py
python multi_predator.py
python evasion_analysis.py
python coordinated_predators.py
python encirclement.py
python encirclement_scaling.py
python fragmentation.py

# Panic and contagion
python panic.py
python panic_contagion.py
python contagion_sis.py

# Hybrid stressors
python hybrid_stressors.py
python hybrid_sis.py
python outbreak_removal.py

# Scaling and herd immunity
python large_N_encirclement.py
python renc_scaling.py
python critical_shift.py
python herd_immunity.py
python targeted_immunity.py

# Long-time and adaptive dynamics
python long_encirclement.py
python encirclement_gap.py
python adaptive_encirclement.py
```

Open `sim_demo.html` in a browser for a real-time interactive simulation with adjustable parameters.

---

## Tools

- **Language:** Python 3 — numpy, matplotlib, reportlab
- **AI assistance:** Claude (Anthropic) — code generation, debugging, research guidance. All AI use documented in research log.

---

*Charbonneau, P. (2017). Natural Complexity: A Modeling Handbook. Princeton University Press.*  
*Silverberg et al. (2013). Collective motion of humans in mosh and circle pits. Physical Review Letters, 110, 228701.*
