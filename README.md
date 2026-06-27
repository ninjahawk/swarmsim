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

## Project status — where we are

**94 findings (F1–F94), all written up in [`findings.md`](findings.md) and the [report](report_draft.md).** Last updated 2026-06-16.

**Threads, and whether they're closed:**

| Thread | Findings | State |
|--------|----------|-------|
| Baseline & validation | F1–F4 | ✅ closed |
| Phase transition (is the solid→fluid a true transition?) | F2, F8, F12, F17, F38–F40, F50 | ✅ closed — it's a crossover; no `base_r^n` potential crystallizes |
| Predator strategy (2D + 3D) | F5–F35, F43–F45, F49, F53, F65 | ✅ closed — only encirclement disrupts, and only in 2D |
| Predator–prey arms race (predictive encirclement vs collective escape) | F66–F71 | ✅ closed |
| Contagion / vaccination / kinematic mixing | F10–F37, F47–F48, F52, F54–F64 | ✅ closed — slow-recoverer targeting is the one strategy that beats random |
| Leadership / collective decision-making (incl. many-wrongs) | F72–F86 | ✅ closed (paused) |
| **Co-adaptation / evolution** (heritable traits under selection) | **F87–F94** | ✅ complete arc (newest work) |

**Most recent work — the co-adaptation thread (F87–F94):** prey evolve a heritable *escape weight* under capture/removal predation. The F70 "dangerous valley" is a strong evolutionary **brake** (F87): escape is stable once present but can't evolve from scratch — it's a barrier to *origination*, not invasion (F88), and the resulting escape equilibrium is a robust mixed strategy (F89). With a co-evolving predator the arms race is **asymmetric** (F90), a self-test corrected an over-reading of it (F91), and under an energy-budget fitness model the brake proves model-independent while escape's *persistence* does not (F92). A *second* heritable trait, the **alignment strength** α, is also bistable — high alignment is a protective ESS that can't evolve from low (F93) — but unlike escape it does **not** invade from a seeded minority either (F94), because alignment is a mutual coupling rather than a free-rideable shared signal. All of it runs on a validated, fast harness (`vectorized_predator.py`, `vectorized_predator_prey.py`).

**Where to pick up next:** see the end of [`findings.md`](findings.md) (*Open Questions / Next Directions*). The natural next experiments — a third fitness model (proximity-survival) or a two-trait predator (lead + aggression) — each rest on the same harness.

---

## Second chapter — Sandpiles & Self-Organized Criticality

A new topic (Charbonneau Ch. 5), distinct from the flocking work below. Code in
[`sandpile/`](sandpile/), findings in [`findings_sandpile.md`](findings_sandpile.md)
(S-series). The arc: validated 1-D core → rigorous critical exponents → the
chapter's 2-D "Grand Challenge" → a universality comparison against the canonical
abelian (Bak–Tang–Wiesenfeld) sandpile.

**Headline result:** the *phenomenon* of self-organized criticality (scale-free
avalanches as a dynamical attractor, no parameter tuning) is robust, but the
critical *exponents* are not — they depend on both dimension and toppling rule:

| model | avalanche exponent | same class? |
|-------|--------------------|-------------|
| 1-D slope sandpile | τ_E ≈ 1.03 (D_E ≈ 2.0) | — |
| 2-D slope sandpile | τ_E ≈ 0.87 | ✗ differs from 1-D (purely dimensional; local rule identical) |
| 2-D canonical BTW | τ_S ≈ 1.14 (lit. ~1.2) | ✗ differs from 2-D slope (τ_S ≈ 0.89) |

The 2-D slope-vs-BTW split is now confirmed by a *second, convention-free* measure:
the avalanche **duration** exponent, τ_T ≈ 0.56 (slope) vs ≈ 1.22 (BTW), stable
across lattices up to 512 a side (S10). That measurement needed an active-list,
numba-compiled engine (Exercise 5, S9) that reproduces the original engines
bit-for-bit and runs ~600× faster in 2-D.

And the criticality is *exact* only because the bulk is conservative: breaking
conservation (destroying a fraction of toppled sand) collapses the cutoff scaling
from N² to sub-N¹ and truncates the avalanche distribution at a dissipation-set
size — singling out conservation among the four SOC ingredients (the OFC
earthquake-model sensitivity, in the simplest sandpile).

Validated against the chapter: mass conservation to 3×10⁻¹¹, N-independent
power-law avalanche PDFs, initial-condition independence (the SOC attractor), and
a clarification of the angle-of-repose claim (the *mean* slope sits ~16% below Z_c
independent of grain size; only the *peak* pre-avalanche slope approaches Z_c as
grains shrink).

```
sandpile/
  sandpile1d.py    1-D slope model (eqs 5.1–5.10), avalanche measurement
  validate1d.py    conservation, SOC signatures, IC-independence, eps-sweep
  fss1d.py         1-D finite-size scaling: τ_E, D_E, D_T + data collapse
  repose_peak.py   angle-of-repose mean-vs-peak slope reconciliation
  sandpile2d.py    2-D bond-slope generalization (Grand Challenge)
  fss2d.py         2-D finite-size scaling + 1-D-vs-2-D comparison
  btw_compare.py   canonical 2-D abelian BTW, same pipeline (universality)
  dissipation.py   break bulk conservation -> is conservation necessary for SOC?
  dissipation2d.py 2-D version of the conservation test (transfers from 1-D)
  duration_compare.py  duration-exponent universality cross-check (self-test, S6)
  falloff.py       boundary-evacuation avalanches vs bulk avalanches (Exercise 2)
  sandpile_fast.py active-list, numba-compiled 1-D+2-D engine (Exercise 5, S9)
  duration_fss2d.py large-L duration FSS; resolves S6 (S10)
  moments.py       moment-scaling machinery + FSS self-test (S11)
  moment_slope.py  slope-model moment spectrum: filamentary area (S12)
  conditional.py   conditional avalanche exponents; ballistic front (S13)
  geometry2d.py    per-avalanche footprint geometry: the filament (S14)
  stochastic_split.py  tunable split toward Manna; the causal test (S15)
  geometry1d.py    1-D footprint geometry = the dimensional anchor (S16)
  equilibrate2d.py reusable warmup to a verified-stationary L>256 (S16)
  duration_closure.py  single-scale exact in space, residual in time (S17)
  area_multifractality.py  area multifractality is asymptotic, not a corona (S18)
```

Exercises addressed: 2 (falloff avalanches, S8), 3 (initial-condition
independence, S1), 4 (eps-dependence of the angle of repose, S2), 5 (fast
active-list engine, S9), 6 (the 2-D Grand Challenge, S4) — plus the conservation
study (S5/S7) and a self-test that was inconclusive (S6) then resolved (S10).

**The scaling-theory arc (S11–S18).** Beyond the exercises, the chapter builds one
extended argument about *what a 2-D slope avalanche actually is*. Moment-spectrum
analysis (S11–S12) finds the avalanche **footprint is filamentary** — its area grows only
linearly with system size (D_area ≈ 1), against the compact D ≈ 2 of canonical BTW through
the same pipeline — while the total toppling activity is space-filling (≈ L²): a thin front
that sweeps its own bonds ≈ L times. Conditional exponents (S13) and a direct footprint
recording (S14) confirm it geometrically — a constant-width, one-bond-wide, **ballistic**
filament (topple time ≈ distance from the seed), thinner than the exactly-solvable directed
sandpile (D = 3/2). A tunable stochastic split toward the Manna class (S15) shows the
filament is *caused* by the deterministic rule (its dimension climbs 1 → 1.87 as randomness
is switched on), yet the model never collapses onto simple Manna scaling. A 1-D dimensional
anchor (S16) proves D ≈ 1 is intrinsic to a slope avalanche, not a 2-D artifact, and ships
an equilibration tool that reaches a verified-stationary L = 512. With it, the last two
findings pin down *how* single-scale the model is: the **spatial** relations become
*exactly* single-scale as L → ∞ (size ∝ area², area ∝ duration), but **duration** stays a
permanently loose proxy for spatial extent (S17), and the avalanche **area distribution**
stays **multifractal** asymptotically rather than healing to simple scaling (S18) — the
model is single-scale in its *means* (the typical avalanche) and multifractal in its *tails*
(the largest avalanches). It sits in its own place in the SOC landscape: more filamentary
than the directed sandpile, outside the stochastic Manna class, a distinct anomalous class
from BTW. The whole arc is drawn together in a standalone write-up,
[`report_sandpile.md`](report_sandpile.md); full per-finding detail (S-series) is in
[`findings_sandpile.md`](findings_sandpile.md).

---

## Third chapter — Earthquakes (the Olami–Feder–Christensen model)

A new topic (Charbonneau Ch. 8), the direct continuation of the sandpile
conservation thread. Code in [`earthquake/`](earthquake/), findings in
[`findings_earthquake.md`](findings_earthquake.md) (E-series). The OFC model is a 2-D
lattice of "force" values driven uniformly until a node hits threshold and topples,
passing a fraction α to each neighbor (0 ≤ α ≤ 0.25). At **α = 0.25 redistribution is
conservative**; for **α < 0.25 the bulk dissipates** (1−4α) per topple — so α is a
continuous knob on the exact ingredient (bulk conservation) that finding S5 singled
out for SOC.

**Headline result:** OFC reproduces the Gutenberg–Richter law, and conservation tunes
criticality. The avalanche-size cutoff **grows with system size when (near‑)conservative
and saturates to a dissipation‑set scale when not** — the S5 result made quantitative on
the canonical earthquake model. And unlike the sandpile, the nonconservative model is
deterministic and develops **synchronized domains** that drive *quasi‑periodic* recurrent
avalanching — temporal structure the sandpile entirely lacks.

| α (conservation) | PDF slope (book) | cutoff vs L | regime |
|---|---|---|---|
| 0.25 (conservative) | −1.19 (−1.19) | grows steeply (D ≈ 3.2) | **critical** |
| 0.20 | −1.87 (−1.92) | ≈ L² (D ≈ 1.95) | critical‑like |
| 0.10 (60% dissipated) | −3.52 (−3.34) | flat (D ≈ −0.15) | **subcritical** |

Validated against the chapter: the fig.-8.7 PDF slopes (all three match), the
fig.-8.6 synchronization domains forming from a random start, the fig.-8.4 recurrent
avalanching (raw period ~10,600 at α=0.15, matching the book's ~10,960; forcing-corrected
~3,982 vs the book's 4,002; vanishing at the conservative α=0.25). The size–duration relation steepens with conservation
(E ~ T^γ, γ = 1.47 → 1.89 as α: 0.15 → 0.25). A self-test (Exercise 4) confirms the
quasi‑periodicity is fragile: a ±0.01 per‑topple jitter in α halves the recurrence
peak, because synchronization needs the redistribution to preserve equality exactly.

**Grand Challenge (earthquake prediction):** a forecaster that phase-locks to the
recurrence rhythm beats chance only modestly (1.5–1.7×) and **misses the largest
events** (top‑decile recall 0.09) — reproducing, in a fully-known noise-free model, the
book's point that large events sit in an unpredictable tail.

```
earthquake/
  ofc.py                core OFC model + 6 validation self-tests (open/periodic BC,
                        per-topple stochastic α, Exercise-3 skip-forcing, bit-identical)
  ofc_gr.py             E1: Gutenberg–Richter validation (fig. 8.7)
  ofc_fss.py            E2: finite-size scaling / conservation test (cf. S5/S7)
  ofc_quasiperiodic.py  E3: recurrence period + synchronization domains (figs. 8.4/8.6)
  ofc_et.py             E4: avalanche size–duration relation (Exercise 5)
  ofc_speedup.py        E5: forcing skip-ahead speedup (Exercise 3)
  ofc_stochastic_alpha.py  E6: self-test — does stochastic α break periodicity? (Ex. 4)
  ofc_predict.py        E7: Grand Challenge — earthquake prediction (Exercise 6)
```

Exercises addressed: 2 (finite-size scaling, E2), 3 (skip-forcing, E5), 4 (stochastic-α
self-test, E6), 5 (size–duration relation, E4), 6 (the Grand Challenge prediction, E7) —
plus the validation (E1) and the recurrence/synchronization study (E3).

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
root/
  model.py          ← current API. New experiments import from here.
  geometry.py       ← shared helpers (Rg, AR) — used by both old and new scripts.

  flocking.py       ← legacy: procedural baseline.        [keep for reproducibility]
  analysis.py       ← legacy: validation sweeps.          [keep for reproducibility]
  predator.py       ← legacy: single-predator API.        [keep for reproducibility]
  multi_predator.py ← legacy: multi-predator helpers.     [keep for reproducibility]
  encirclement.py   ← legacy: encirclement scaffolding.   [keep for reproducibility]

predator/    predator strategy experiments
contagion/   panic, epidemic, and vaccination experiments
phase/       phase-transition and statistical-mechanics experiments
3d/          three-dimensional extension experiments
figures/     output PNG figures
outputs/     captured text/log output from runs
```

The five legacy scripts at root are intentionally retained: every numbered finding F5
through F16 was generated by them, and rerunning them must produce identical figures.
New experiments should not import from them — they have no `if __name__ == "__main__"`
guard and will side-effect on import. Use `model.py` instead.

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
| **19. 3D predators** | Does encirclement transfer to 3D? Radius, count, adaptive, sphere/planar arrangement | `3d/flocking3d_predator.py` → `3d/flocking3d_predator_scaling.py` → `3d/flocking3d_adaptive.py` → `3d/flocking3d_strategy.py` |
| **20. 3D contagion & segregation** | Do vaccination nulls and α-contrast segregation transfer to 3D? Does 3D mix faster than 2D? | `3d/flocking3d_vaccination.py` → `3d/flocking3d_segregation.py` → `contagion/mixing_dimension.py` |
| **21. Section 5 self-tests** | Does topological alignment slow mixing? Does freezing contacts rescue targeting? | `contagion/topological_mixing.py` → `contagion/contact_freezing.py` |
| **22. Phase-transition closure** | Does hard repulsion crystallize? | `phase/langevin_hexatic_hard.py` |
| **23. Agent memory** | Does fatigue make encirclement damage irreversible? Do heterogeneous recovery rates shift the SIS threshold? | `predator/fatigue.py` → `contagion/recovery_heterogeneity.py` |
| **24. Heterogeneity & targeted vaccination** | Does infectiousness spread shift the threshold? Can vaccinating slow recoverers preferentially beat random — in 2D, 3D, under continuous γ, with noisy estimates, and for rare reservoirs? | `contagion/infectiousness_heterogeneity.py` → `contagion/slow_recoverer_vaccination.py` → `contagion/het_recovery_spatial.py` → `3d/flocking3d_slow_vaccination.py` → `contagion/continuous_gamma_vaccination.py` → `contagion/noisy_gamma_vaccination.py` → `contagion/rare_reservoir_vaccination.py` |

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

92 findings documented — full evidence and figures: [`findings.md`](findings.md)

**Selected highlights:**

| # | Finding |
|---|---------|
| 1 | Cruise speed is **v_eq = v₀ + α/μ** (exact analytical result, not just v₀) |
| 8 | Solid-to-fluid transition is a smooth crossover at all compactness values — no true phase transition |
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
| 40 | **Hexatic \|ψ₆\| confirms n=1.5 cannot crystallize**: flat at ~0.4 across all kT; soft repulsion prevents lattice formation |
| 41 | **3D flocking validates**: v_eq = v₀ + α/μ exact in 3D; Φ = 0.841 at η=10; 3D less noise-robust than 2D |
| 43 | **3D encirclement fails entirely**: Φ ≈ 1.0 at every R_enc, every n_pred — encirclement is strictly 2D-specific (point predators cannot seal a closed surface) |
| 46 | **3D vaccination null transfers**: random / spatial / degree-targeted vaccination all identical in 3D |
| 47 | **Section 5 self-test #1 FALSIFIED**: k-NN alignment does NOT slow kinematic mixing (Jaccard 0.036 vs 0.037) |
| 48 | **Section 5 self-test #2**: Freezing contacts does NOT rescue degree-targeting → the null is **structural** (no hubs), not kinematic |
| 50 | **Hard repulsion null** (F40 falsified): exponent n=12, 24 also flat \|ψ₆\| — `base_r^n` cannot crystallize at ANY exponent; raising n shrinks the core, doesn't harden it |
| 52 | **"3D mixes faster" theme FALSIFIED**: at matched contact degree, 3D rewires at ~0.56× the 2D rate — 3D mixes ~1.8× SLOWER |
| 53 | **Prey fatigue null**: encirclement damage stays reversible even with per-agent fatigue Q; align-mode fatigue deepens during-attack disruption (F27 echo), speed-mode doesn't (F24 echo) |
| 54 | **Heterogeneous SIS recovery**: spread in per-agent γ at fixed mean lowers β_c by ~2.5× (0.385 → 0.155); slow recoverers are reservoirs (1.5–2× more panic than fast); reframes vaccination toward slow recoverers |
| 55 | **Heterogeneous infectiousness does NOT shift the threshold**: super-spreaders source 74–97% of transmissions but γ stays homogeneous, so no stock asymmetry — the vaccination target is γ_i, not β_i |
| 56 | **Slow-recoverer vaccination beats random by 2–3×** — the *first* targeting strategy in this study (of ~15) to beat random. At p=0.40 slow-targeting eradicates (0.000) vs random 0.095. The hub label lives in per-agent γ_i, which kinematic mixing cannot scramble |
| 58 | **Slow-targeting transfers to 3D** unchanged: at p=0.50 slow eradicates (0.000) vs random 0.282; the per-agent-rate mechanism is dimension-independent |
| 59 | **Slow-targeting survives continuous (lognormal) γ** — no bimodal class structure needed; "vaccinate the bottom X% by γ_i" works for any plausible distribution |
| 60 | **Slow-targeting tolerates noisy γ estimates**: identical to perfect knowledge up to σ_obs ≈ 0.8, still beats random at σ_obs = 2.0 |
| 61 | **Slow-targeting works for rare reservoirs**: even a 5% slow class sustains the epidemic under random vaccination; targeting it at a matched budget eradicates |
| 62 | **Slow-targeting needs a *durable* recovery-rate label**: if γ_i drifts (state, not trait), mild drift erodes the advantage and fast drift eradicates the epidemic outright by self-averaging γ to its mean (undoing F54's threshold cut). The policy is valid exactly as long as the slow *class* persists on the epidemic timescale (~1/γ_slow) |
| 63 | **Combined β_i + γ_i heterogeneity**: targeting slow recoverers (the reservoir) is the *robust* vaccine across all correlations; targeting super-spreaders (the transmission engine) is equally good only when they aren't anti-correlated with slow recovery — when supers are fast recoverers, only slow-targeting eradicates. β-targeting *is* effective for removal, refining F55's "target γ not β" to a robustness statement |
| 64 | **Reservoir-targeting reverses the predator+contagion damage asymmetry**: F34's "contagion is the worst stressor" (epidemic outlasts predator removal) holds only while the reservoir survives. Vaccinating the slow class at a budget matching the reservoir fraction (p≥f_slow) eradicates the epidemic *and* lets the flock reunite to Φ≈1.0 — fully reversible combined damage. Below that budget the epidemic persists for every strategy |
| 65 | **3D flocks are robust to *all* point-predator strategies, not only sealing**: a fast transecting predator that punches through the core gives Φ=1.000 at every count and speed up to 40× prey speed — same as encirclement (F43). The 3D flock fills the box near-uniformly (Rg=0.43 of ~0.5 max), so it has no perimeter to seal AND no interior to transect; finite-range predators perturb a vanishing fraction while the alignment graph heals the wake. Refines F43's mechanism: it's "no spatial localization," not "no surface to seal." Disrupting 3D needs a per-agent alignment attack (contagion does this) |
| 66 | **Predictive encirclement** (predators target CoM + lead_time · v̄) is the *first predator-side adaptation* in this study to substantially beat F14: Φ drops to **0.530** at lead=2 tu (vs F14's 0.77 and F35-adaptive's 0.713), non-monotonic with optimum near lead ~ R_enc/v̄ where the lead distance matches the encirclement radius. Inverts the F33 asymmetry — the flock can't detect global escape directions, but predators can detect v̄. Adapting *position* (F66) is independent of adapting *radius* (F35) |
| 67 | **Predictive + adaptive predators do NOT compose**: combined predictive-adaptive Φ=0.535, within noise of predictive-fixed (0.530). Under encirclement the flock compresses to Rg~0.05–0.10, so adaptive R_enc=0.5·Rg shrinks the ring to ~0.03 while predictive shifts it 0.24 ahead — predators degenerate into a near-point block in the heading direction, yet are as effective as a proper ring. **Placement is the dominant lever; angular spread becomes secondary once the heading is blocked** |
| 68 | **Predictive encirclement is *less* noise-tolerant than slow-targeting**: noisy v̄ estimates degrade Φ monotonically and gracefully — but *graded from σ=0*, no plateau (0.530→0.629→0.709→0.804 as σ_obs goes 0→25%→100%→400% of \|v̄\|). Unlike F60's slow-targeting, which is identical to perfect knowledge up to σ ≈ half the signal separation. Statistical reason: per-agent rankings are buoyed by N-sample averaging; global summary statistics (one vector per step) have no averaging buffer |
| 69 | **Predictive encirclement is *far* more sensitive to delay than noise**: acting on a *stale* v̄ destroys the advantage fast — a 0.25 tu lag loses ~83%, by 1 tu it's gone (Φ≥F14). Delay is a *systematic* directional bias on a forward-projected quantity (vs noise's zero-mean error), and v̄ decorrelates sub-tu under disruption. Dual of F60: a per-agent invariant is both noise-robust *and* delay-free; a global heading is both noise- *and* delay-sensitive. Closes the predator-learning thread (F66–F69) |
| 70 | **The arms-race step: collective escape intelligence beats predictive encirclement — if committed**. Giving prey the dual global signal (flee the predator centroid, weight w) is *non-monotonic*: weak escape (w=0.25) is *worse* than none (Φ=0.275, competes with alignment), but strong escape (w≥2) fully restores Φ=1.000 — a unified flee reinforces alignment and outruns the trap. Threshold ≈ alignment strength. Elegantly, the counter works *only because* predictive predators mass ahead (defining the escape direction) — **the predator's own intelligence creates the prey's opening; predictive encirclement is self-defeating vs committed escape-intelligent prey** |
| 71 | **But the F70 escape needs a globally *shared* direction, not just escape info**: with realistic *local* per-prey sensing (flee predators within r_sense), escape only partially works — Φ peaks at ~0.83 near the ring scale, never the full 1.000, even at global range. Per-prey escape vectors point different ways and *compete* with alignment instead of reinforcing it; too-large r_sense sees the symmetric ring and cancels (F33 echo). Honest caveat on F70: full escape is partly an artifact of a shared signal; the flock acts collectively only on already-shared signals |
| 72 | **A tiny informed minority steers the whole flock — and it works for the *same* reason F70 did**. Couzin-style leadership: a fraction ρ of agents carry a preferred direction g, the rest are naive followers. Just **ρ=0.05 (18 of 350)** already drives accuracy 0.63–0.83; ρ=0.10 reaches 0.87–0.96; saturates at ρ≈0.20. Stronger leaders reach a given accuracy at smaller ρ *and* with lower seed variance. **Cohesion is never lost** (Φ≥0.995 throughout) — the opposite of the predator case, where steering costs coherence. The minority's power is *agreement, not numbers*: all leaders share one common vector g, which alignment amplifies — the **constructive** half of the F70/F71 shared-signal rule (F70 escape, leadership, and flocking itself are one phenomenon: a globally shared direction is amplified, a locally heterogeneous one is not) |
| 73 | **Conflicting leaders: the flock compromises, then votes — reproducing Couzin's decision dynamics**. Two informed subgroups pull in different directions. At small angular conflict (θ≤90°) the flock **compromises**, traveling almost exactly the midpoint (θ=90°→ heading 45°), tight across seeds. Past a critical angle (~90–120°) it switches to **consensus**: cross-seed heading spread explodes (22°→74°) as different seeds commit to one goal or the other — averaging two near-opposed directions isn't a viable heading, so symmetry breaks. It **picks one, it does not split** (Φ≥0.96, split fraction ≤0.16 even at 180°). And the **majority wins**: at direct opposition a 21:14 informed-agent margin already biases the flock +0.42 toward the majority goal, rising to +0.93 — group direction is an effective majority vote among the informed. Alignment doesn't just propagate one shared direction, it *arbitrates* among several (average-when-compatible, vote-when-not) |
| 74 | **Numbers vs conviction: the decision is set by total *pull* (count × strength), not headcount**. At equal numbers (18 vs 18), the more strongly committed subgroup wins, more so with the conviction ratio (+0.20 tie → +0.71 at 5×). And a **small strong minority beats a large weak majority**: 10 strongly-committed agents tie then overtake 26 weak ones as the minority's total pull crosses the majority's — accuracy crosses zero near pull balance (−0.27 at pull 10 → ≈0 at 26 → +0.07 at 50). The zero-crossing sits *slightly above* equal pull, a mild residual numbers bonus (more agents nucleate the direction in more places). The flock is a **weighted** majority integrator: each vote scaled by commitment, decided by summed directed force — the quantitative form of the F72/F73 alignment-arbitration rule, echoing F70's force-vs-alignment escape threshold |
| 75 | **Time-resolved decisions: leadership is fast & noise-robust, and the flock shows *critical slowing* at the decision boundary**. Recording the full heading time series (not just the steady state): strong leadership is **both faster and more accurate** — settle time collapses from ~4.5 tu at ρ=0.20 to 0.46 tu at ρ=0.50, accuracy →1.0; no real speed-accuracy tradeoff (the "fast but wrong" ρ=0.02 case is just under-led). Noise barely matters — across a 20× noise range accuracy stays ≥0.99 and commitment slows only 8.9→11.3 tu (cf. F4). **Headline:** commitment time *peaks at θ=90°* (12 tu), falling off on both sides — the dynamical signature of a **bistable system slowing near its bifurcation**, sitting exactly at the F73 compromise→consensus boundary. The transition is a genuine bifurcation, and the flock takes longest to decide precisely when the decision is hardest |
| 76 | **Leadership is a *signal*, not an *identity*: rotating which agents are informed never hurts — and faster rotation *helps***. Keep the goal fixed but re-draw the informed set every τ: accuracy equals or beats the fixed-leader baseline at every τ, and *fast* rotation raises accuracy (0.86→0.94 at ρ=0.05) while collapsing seed variance (std 0.14→0.07; at ρ=0.10, 0.04→0.007). The same total pull smeared uniformly over all agents steers more reliably than a fixed subset that must propagate its bias. The **exact opposite of F62**, where drifting the per-agent label *destroyed* the advantage: contagion targeting needs a durable per-agent identity, leadership needs only a durable shared *direction*. Closes the "what makes a target exploitable" arc — fragile if it rests on a persistent label, robust (even helped) by turnover if it rests on a shared global quantity |
| 77 | **The flock has a *steering bandwidth***: leaders can drive a turn only below a critical rate. With a goal rotating at ω, tracking is near-perfect at ω=0 but degrades (54° lag at ω=0.05) and *fails* by ω=0.10 at ρ=0.10 (the goal laps the heading). The bandwidth **scales with the informed fraction as 1/(F75 response time)** — ≈0.11 rad/tu at ρ=0.10, ≈0.22 at ρ=0.20 (doubling leaders ≈ doubles ω_crit); F75 and F77 are the time- and frequency-domain views of one timescale. Below it the flock is a **low-pass steering filter** (lags, then attenuates). And **over-steering costs cohesion**: a turn faster than the flock can follow drops Φ to 0.78 — unlike cohesion-free fixed steering. Steerability and coherence are the same resource, mediated by the alignment response time — the same tension as predation |
| 78 | **Leadership counters encirclement — the two biggest threads meet**. Encirclement (F14) is the *one* predator strategy that breaks 2D coherence (Φ→0.79). Give a fraction ρ of prey a shared *goal* direction (not fleeing — just a heading) while 6 predators encircle: leaders both **restore the coherence the predators destroyed** (Φ 0.79→0.94 at ρ=0.40) *and* **steer the flock through the ring** (accuracy 0→0.98). Encirclement *raises the leadership threshold* (ρ=0.05 steers freely with no predators but needs ρ≈0.2–0.4 under the ring) — the quantitative cost. **Generalizes F70**: it's not the *content* of the shared signal (flee vs goal) that counters the predator but the *presence* of any strong shared heading. Encirclement wins by erasing the common direction; leadership (even predator-oblivious) is its antidote. Predation and leadership pull on opposite ends of one lever — shared alignment |
| 79 | **Spreading panic severs the *rudder*, not the *hull* — the complement of encirclement**. SIS panic makes agents erratic and a panicked leader can't lead. As β rises, **steerability collapses** (accuracy 0.94→~0 by β/γ≈2) but **coherence is untouched** (Φ stays ≥0.98 even at 98% panic — noise below the F4 melting point). The flock stays a tight flock flying a random, leaderless heading. Mechanism = the **F74 pull law taxed by panic**: panic saturates immediately (spatial R0≫1, cf. F18–F25), so what matters is the *depth* — at any instant a fraction *f* of leaders is silenced, giving effective ρ=(1−f)·ρ; steering tracks it exactly. Encirclement attacks coherence (leadership fixes it); panic attacks steerability (leadership *can't* fix it — the disease disables the leaders). Explains why **contagion is the study's most durable stressor**: the shared heading can't repair itself once its carriers are panicking |
| 80 | **Adversarial leadership: *denial is cheaper than capture***. A saboteur minority pushes toward a "trap" against true leaders (ρ_true=0.10 toward goal). **Denial** (deadlock the flock, deny its goal) needs only *parity* — at ρ_sab=ρ_true goal-accuracy collapses 0.92→0.11, and half-parity already halves it. **Capture** (drive the flock to the trap) needs a *majority* — accuracy crosses zero only past parity and decisive capture (acc<−0.5) takes ρ_sab≈2× the true leaders. Between sits a *deadlock band*. The zero-crossing at pull parity is exactly the F74 product law; the new content is the **threshold gap** (denial ≈1×, capture ≈2×). Security asymmetry: an adversary can *paralyze* a led flock at parity but must *dominate* to hijack it — disrupting the shared heading is easier than commandeering it. Φ only dips to 0.88 (the fight is over heading, not integrity) |
| 81 | **(self-test) Steering is set by the informed *fraction*, not the absolute number — correcting F72's offhand Couzin attribution**. F72 cited Couzin (2005)'s "informed *number* needed drops as the group grows" but only ever swept ρ at one N. Varying N at *fixed* leader count n_lead, accuracy *falls* with N at every fixed n_lead (n_lead=20 → 0.96/0.76/0.48 at N=100/250/500); re-indexed by fraction the data collapses with no residual N trend. Steering delivers exactly the F74 product law **per capita** — total injected force n_lead·w divided by *all* N — so growing N at fixed n_lead dilutes the per-capita pull. The literature's number-suffices scaling comes from a *many-wrongs* preferred-direction-averaging amplification that **linear velocity alignment does not have**; this model delivers per-capita pull with no amplification. The fraction-based frame (F72/F78/F79/F80) is correct; the one number-based intuition is the exception removed here. A many-wrongs follower rule would be needed to recover the literature scaling — an open direction. Fourth self-test (cf. F47/F48/F52) |
| 82 | **Many-wrongs navigation: noisy *private* goal estimates average to a 1/√N wisdom of crowds — recovering exactly the amplification F81 predicted but found absent for exact-shared vectors**. Give *every* agent its own goal estimate at per-agent angular error σ_pref and bias it toward its own: the flock's RMS heading error falls 16°→2° as N grows 30→250 (log-log slope **−0.52**, predict −0.5) and accuracy *rises* with N (0.96→0.999) — the clean inverse of F81. Each agent alone would err ~57°, but a flock of 250 navigates to within 2°. No contradiction with F81: alignment averages whatever signal the agents carry — an exact shared vector has no error to average (per-capita pull, F81), heterogeneous noise averages down as 1/√N (F82). Two limits: a ~2° **floor** (size-independent dynamical jitter) and a **noise ceiling** — past σ_pref≈1.3 rad the pooled estimate magnitude (~exp(−σ²/2)) collapses below the spontaneous-heading threshold and accuracy falls 0.99→0.58 with exploding seed scatter while Φ stays ~0.90 (the many-wrongs form of the F72/F74 pull threshold). Alignment is a directional averager: per-capita for shared signals, √N-amplifying for noisy ones |
| 83 | **Correlated estimates: F81 and F82 are the two ends of *one* axis (error correlation ρ_c), and shared sensing error caps the wisdom of crowds**. Build each agent's goal error as √ρ_c·(shared) + √(1−ρ_c)·(private): at ρ_c=0 error falls 1/√N (F82), at ρ_c=1 it's *exactly* N-independent (68° at every N, Φ=1.0 — the F81 limit, all agents confidently agree on the same wrong heading). Any ρ_c>0 imposes an N-independent floor ≈σ√ρ_c that **no flock size can beat**: a mere 10% correlation pins accuracy at ~0.92 whether N=30 or 500. Bonus: correlation buys *coherence* (Φ 0.88→1.0) at the cost of *accuracy* (0.99→0.44) — a perfectly coherent flock can be unanimously wrong (consensus ≠ correctness, cf. F73). Ties to F80: an attacker who can't out-number the leaders can instead inject a single shared false cue and cap the collective's accuracy regardless of size — **common-mode deception is cheaper than majority capture**. Closes the many-wrongs sub-thread (F81–F83) |
| 84 | **The noisy minority: informed-minority steering and the wisdom of crowds are *distinct* mechanisms that don't combine in a minority**. A fixed *number* of *noisy*-informed agents (rest naive) still fails as N grows — exactly F81, not F82: at n_lead=20 accuracy falls 0.86→0.43 as N=100→500, and noisy sits ≤ exact everywhere (the pooled-direction penalty σ/√n_lead — leaders agree on a heading ~18° off, which followers can't fix). Only growing n_lead recovers accuracy (0.22→0.91 at N=250). The 1/√N many-wrongs amplification (F82) needs the informed *fraction* to grow; confined to a fixed cadre it gives per-capita-diluted steering toward a fixed-accuracy pooled target — strictly worse than an exact minority. Separates Couzin (informed-minority) from Simons/Codling (many-wrongs) within one model. Completes the leadership mechanistic map (F72–F84): alignment is a directional averager whose accuracy = per-capita pull (fraction×strength) toward a target set by the estimates' correlation (F83) and sample size (F82/F84) |
| 85 | **Misinformation: a crowd averages out *noise* but flips to a coordinated *falsehood* of the same size**. A fraction f of agents are misinformed — either *lost* (uniform-random) or *adversarial* (coordinated toward a false goal). Lost is near-harmless: accuracy stays **0.998 even at f=0.5**, only 0.983 at f=0.7 (random unit-votes cancel). Adversarial is decisive: accuracy crosses **zero at parity f=0.5** (0.62→0.11→−0.62 across f=0.4–0.6), Φ bottoming at 0.78 at the deadlock point (F80 product law + F73/F75 critical-slowing dip). Same number of equally-wrong agents — only the **correlation** of their error differs (ties F83). Lost-robustness is scale-free (N-independent at fixed f). The lesson across F80/F83/F85: what threatens a collective's heading is never the *amount* of error but its *correlation* — alignment averages out everything independent and is moved only by what's shared. Caps the many-wrongs arc (F81–F85) |
| 86 | **(self-test) A noisy crowd tracks a *moving* goal at the same bandwidth as a sharp leader — spatial (many-wrongs) and temporal (F77 bandwidth) averaging are *independent***. With a goal rotating at ω and every agent carrying a fixed private offset, the tracking-vs-ω curves for σ_pref=0/0.5/1.0 **overlay** (bandwidth flat at ~0.20 rad/tu) even as the averaged-bias magnitude falls to 0.61. My pre-registered prediction (bandwidth ∝ exp(−σ²/2)) is **falsified**: the static offsets rotate *with* the goal and add no lag, and the reduced magnitude stays above the steering threshold. Noise is *free* below the F82 ceiling and *catastrophic* above it (σ=1.5 collapses, not a graceful bandwidth drop) — binary, not graded. Shows the directional average is recomputed each timestep, not integrated. 5th self-test (cf. F47/F48/F52/F81); closes the many-wrongs arc (F81–F86) |
| 87 | **Evolution: the F70 escape-weight "dangerous valley" is a strong evolutionary *brake*, not a barrier** *(opens the co-adaptation thread)*. Making the collective-escape weight a **heritable** per-agent trait under **capture/removal** selection against the predictive predator (F66): escape (w≥α) is evolutionarily **stable and near-cost-free once present** — seeded at w₀=2 it stays at 2.0 with **~6 captures, Φ=1.0** — but does **not** evolve from the no-escape state. Selection on w is directional-*upward* from every start, yet the valley throttles the climb so hard that w₀=0 crawls to only **w~0.5 in 400 tu (never reaching w=1)** while taking the heaviest predation (captures **peak in the valley**: ~1800–1900 at w₀=0–0.25 vs ~6 at w₀=2). Strong evolutionary **hysteresis / first-mover problem**: escape is easy to keep, hard to evolve de novo, because the path runs through a region where partial commitment is selected against (the F16/F24/F70 domination-not-blending theme, read at the evolutionary level). Fitness model = capture/removal (a deliberate scientific choice). Predator side still fixed — next: co-evolve it |
| 88 | **The F87 brake is a barrier to *origination*, not invasion** — escape spreads once present. Seeding a **5% escaper founder group** (w=2) into a no-escape flock carries the mean weight into the escape regime (w_end **1.18**, escaper fraction 0.05→0.55, captures halved); larger seeds settle the same. Escape establishes at a **mixed ~60% equilibrium**, not fixation — a **free-rider / herd-protection** effect of the *shared* escape signal (once enough flee, the predator is outrun and low-w agents ride along protected; the public-good face of the F70/F72 shared-direction rule). From a uniform w=0 start, the F87 mutation step (σ=0.10) never reaches w=1, but **σ=0.30–0.60 cross in 100%** of seeds (σ=1.0 noisier, 50% — an intermediate sweet spot). So whether escape evolves is **mutation-/standing-variation-limited**, not selection-limited: selection favours escape once the valley is cleared, but variation must deliver an agent past it in one step or a pre-adapted founder group must arrive |
| 89 | **The escape free-rider equilibrium is *robust* — escape never fixes**. Mapping the F88 mixed equilibrium against predation pressure: sweeping the capture hazard across an **8× range** moves the steady escaper fraction only **0.56 → 0.66**, and escape never approaches fixation — even under the heaviest predation **~⅓ of the flock rides the shared escape as protected free-riders** (mean w stays in the escape regime; runs from different starting fractions settle into the same band). Intensifying predation erodes free-riding only weakly, confirming the **public-good** reading: a shared, non-excludable escape direction supports a persistent non-contributing fraction, so escape stabilises as a **durable mixed strategy** rather than sweeping to fixation — the population-level face of the F70/F72 shared-direction rule. Carries the F87/F88 hysteresis signature (mild start-dependence) |
| 90 | **Two-sided arms race is *asymmetric*** *(closes the co-adaptation thread)*. The predator co-evolves a heritable predictive **lead time**, selected on capture success (replace-worst-with-mutated-best). The predator climbs to its optimum from any start, but the prey counter is **origination-limited** (F87/F88), so de-novo co-evolution **favours the predator** (prey stall at w~0.7); from *seeded* escape, escape **fixes** (w~2.0, frac 0.99) and the predator's lead **drifts** (no captures → no selection signal). Reconciles F89: the ~60% free-rider equilibrium needs a *persistently effective* predator; once the predator decays, escape fixes. Collective escape is a **powerful but evolutionarily fragile** defence — hard to originate, easy to lose to drift, decisive once present. (The predator evolved a lead ~3, *above* the F66 disruptive ~2 — I read this as "capture ≠ disruption," but **F91 falsified that**; see below. 2–3-seed noise; the asymmetry is robust) |
| 91 | **(self-test) The capture optimum *coincides* with the disruption optimum — correcting F90**. Measuring capture rate and coherence directly against a *fixed* predator lead (frozen prey, no evolution): the **capture rate peaks at lead~2** (9.6/tu) — *exactly* where coherence is lowest (Φ=0.584) — and collapses by lead 4 (predators overshoot). So the lead that catches the most prey **is** the lead that most disrupts the flock; the F90 evolved lead~3 was **not** a distinct capture optimum but a **small-population selection artifact** (at lead~3 the capture rate is wildly variable, ±5.6, and replace-worst-with-best chases that lucky tail). F90's "capture ≠ disruption" claim is **withdrawn**; the F90 asymmetry stands. Lesson, inverted: a tightly-converged evolved trait under noisy small-population selection **need not sit at its fitness optimum** — check against a direct measurement. 6th self-test (cf. F47/F48/F52/F81/F86), the only one correcting this session's own thread |
| 92 | **Robustness: the origination *brake* is model-independent, the *persistence* of escape is not**. Re-running the thread under an **energy-budget** fitness model (an explicit metabolic death hazard ∝ w, paid even when safe) splits the earlier results. The brake **survives** — escape still can't evolve from no-escape (w₀=0 → 0.05), so the F70-valley barrier is general, not an artifact of capture/removal. But F87's "escape is free & stable once present" **fails**: at cost c=0.5 *every* start collapses to w~0.05 (even a w₀=2 seed), and the ESS falls **sharply** with cost (c=0 → w=1.28, but **c=0.25 → 0.11**) — nearly all-or-nothing, no interior optimum. Mechanism: the cost is paid by *every* escaper *continuously* while predation threatens only the few near a predator at any instant, so a modest per-capita cost outweighs the diffuse benefit. Collective escape is viable only where it's **essentially free** — even more evolutionarily fragile than F90/F91 implied |
| 93 | **A *different* heritable trait — alignment strength α — is also bistable: an origination brake mirroring F87**. Evolving the flocking force itself (heritable per-agent α, **no escape force**) under the same capture/removal predation: the outcome is **history-dependent**. A high-alignment flock is a strong protective ESS — α₀=2 stays at **2.00, Φ=1.0, just 6 captures** (a tight flock evades the predictive encircler) — but alignment does **not** climb from below: α₀=0.2 → 0.19, α₀=0.5 → 0.39, and α₀=1.0 *drifts down* to 0.85 (Φ≈0.55, **~1400 captures**, ~230× the ESS toll). Selection clearly favours the high state, yet the population can't reach it de novo — the basin boundary sits near α~1.5. A collective defence requiring the *whole group* to act together (coherent flocking) is hard to originate, exactly like the escape valley — same brake, different trait and force |
| 94 | **But unlike escape (F88), a seeded high-alignment minority does *not* invade — alignment is a mutual coupling, not a free-rideable signal**. Seeding f=0.05–0.50 of high-α agents (α=2) into a low-α flock: the minority is **diluted at every fraction** (f=0.05/0.10 → ~0, mean back to 0.2; even f=0.50 only *holds* at frac~0.45, mean~1.0 at the boundary, Φ=0.65 — the protective ESS never forms, captures stay ~550–650). No mutation step jumps the basin either (σ up to 1.0 reaches only 0.74; **0% of seeds** reach α=1, vs F88's σ=0.3–0.6 clearing the escape valley 100%). So the alignment brake blocks **both origination *and* invasion** — *stronger* than F88's. The reason is a clean contrast: escape is a **public good** carried by a shared signal (one agent's flee protects all → free-rider spread, F88), but alignment is a **reciprocal coupling** — a high-α agent aligns to neighbours who don't align back, so a minority can't nucleate a coherent core and is dragged by the indifferent majority. Tight flocking is a **coordination/quorum trait** that only pays off near a majority a founder group can't reach. Whether a group-defence trait can invade depends on whether it's free-rideable |

Full documentation, evidence, and figures for each finding: [`findings.md`](findings.md)

---

## Repository

### Core

| File | Description |
|------|-------------|
| `model.py` | **OOP foundation** — `Flock`, `Predator` classes; `flock.evolve()`; `simulate()` helper. Import this for new experiments. |
| `flocking.py` | Procedural core — buffer zone, vectorized forces, run loop, order parameter. Used by legacy experiment scripts. |
| `vectorized_predator.py` | Vectorized predator→prey repulsion force; reproduces `model.Predator.force_on_prey` exactly (self-test asserts 1e-17 agreement) at ~250× the speed. Additive helper. |
| `vectorized_predator_prey.py` | Fast predictive-encirclement (F66) + collective-escape (F70) episode runner, `run_episode()`. Verified bit-identical to the legacy predator scripts; supports a per-agent escape weight and returns final state for downstream analysis. |

### Experiment scripts

Experiments live in four theme subfolders. Each script is self-contained, has an `if __name__ == "__main__"` guard, and imports the root-level core library (`flocking.py`, `model.py`, `geometry.py`) via a `sys.path` insert at the top of the file.

| Folder | Theme | Representative scripts |
|--------|-------|------------------------|
| `predator/` | predator strategies, encirclement, fragmentation, reunion, sensing, adaptive radius, prey fatigue, predictive encirclement, collective/local escape (F5–F16, F19, F21–F22, F28, F31–F35, F53, F66–F71) | `encirclement_scaling.py`, `adaptive_encirclement.py`, `fragmentation.py`, `long_encirclement.py`, `fatigue.py` |
| `contagion/` | panic, SI/SIS contagion, vaccination, segregation, mixing, heterogeneous recovery, slow-recoverer targeting (F18–F37, F47–F48, F52, F54–F64) | `contagion_sis.py`, `targeted_immunity.py`, `spatial_vaccination.py`, `recovery_heterogeneity.py`, `slow_recoverer_vaccination.py` |
| `phase/` | finite-size scaling, hard repulsion, Langevin, hexatic order parameter (F2, F8, F12, F17, F38–F40, F50) | `phase_transition.py`, `langevin_repulsion.py`, `langevin_hexatic.py`, `langevin_hexatic_hard.py` |
| `3d/` | three-dimensional flocking, predators, vaccination, segregation, slow-recoverer targeting, transect-predator robustness (F41–F46, F49, F51, F58, F65) | `flocking3d.py`, `flocking3d_predator.py`, `flocking3d_vaccination.py`, `flocking3d_slow_vaccination.py` |
| `collective/` | collective decision-making — informed-minority leadership, conflicting leaders, compromise vs consensus, numbers-vs-conviction product law, time-resolved decisions & critical slowing, signal-vs-identity rotation, steering bandwidth, leadership-vs-encirclement, leadership-under-panic, adversarial leadership, fraction-not-number scaling self-test, many-wrongs wisdom-of-crowds navigation, correlated-estimate ceiling, noisy-minority mechanism split, misinformation robustness, moving-goal crowd tracking (F72–F86; `leadership.py` lives at root) | `conflicting_leaders.py`, `conviction.py`, `decision_time.py`, `rotate_leaders.py`, `moving_goal.py`, `led_encirclement.py`, `panic_leadership.py`, `adversarial_leaders.py`, `leader_scaling.py`, `many_wrongs.py`, `correlated_estimates.py`, `noisy_minority.py`, `misinformation.py`, `moving_goal_crowd.py` |
| `evolution/` | co-adaptation — heritable per-agent escape weight under capture/removal selection vs the predictive predator; the F70 valley as an evolutionary brake (F87) that blocks origination but not invasion (F88); robust free-rider equilibrium (F89); two-sided arms race with a co-evolving predator (F90); direct capture-vs-lead diagnostic (F91 self-test, correcting F90); energy-budget robustness check (F92). Builds on the validated `vectorized_predator.py` + `vectorized_predator_prey.py` harness. | `escape_evolution.py`, `escape_invasion.py`, `escape_freerider.py`, `escape_coevolution.py`, `escape_capture_curve.py`, `escape_energy.py` |

A complete file-by-file index lives in the per-finding evidence sections of [`findings.md`](findings.md). A predator-force sign bug in the 3D scripts was found and fixed in May 2026 (commit `30ead1c`); F43/F44/F45/F49 were rerun with the corrected sign and write-ups updated.

### Supporting files

| File | Description |
|------|-------------|
| `make_demo.py` | Generates `figures/demo.gif` for this README |
| `build_report.py` | Generates `report_draft.pdf` from `report_draft.md` using reportlab |
| `sim_demo.html` | Interactive browser simulation (open locally or via htmlpreview link above) |
| `logs.html` | Time log and research log — open in browser |
| `findings.md` | Running notes on all 92 findings with figures |
| `report_draft.md` | Full research report in Markdown |

---

## Run

All experiment scripts are invoked from the repository root and write figures into `figures/` and logs into `outputs/`.

```bash
# Validation, phase transition, baseline (root-level legacy scripts)
python analysis.py

# Phase transition (subfolder)
python phase/phase_transition.py
python phase/compactness_search.py
python phase/langevin_hexatic_hard.py

# Predator strategy (legacy roots + new subfolder)
python predator.py
python multi_predator.py
python encirclement.py
python predator/encirclement_scaling.py
python predator/adaptive_encirclement.py
python predator/long_encirclement.py
python predator/fatigue.py
python predator/predictive_encirclement.py
python predator/predictive_adaptive_encirclement.py
python predator/predictive_noisy_encirclement.py
python predator/predictive_delayed_encirclement.py
python predator/collective_escape.py
python predator/local_escape.py

# Contagion, vaccination, mixing
python contagion/contagion_sis.py
python contagion/targeted_immunity.py
python contagion/spatial_vaccination.py
python contagion/topological_mixing.py
python contagion/contact_freezing.py
python contagion/mixing_dimension.py
python contagion/recovery_heterogeneity.py

# Heterogeneity and slow-recoverer vaccination (F55-F61)
python contagion/infectiousness_heterogeneity.py
python contagion/slow_recoverer_vaccination.py
python contagion/het_recovery_spatial.py
python contagion/continuous_gamma_vaccination.py
python contagion/noisy_gamma_vaccination.py
python contagion/rare_reservoir_vaccination.py
python contagion/drifting_gamma_vaccination.py
python contagion/het_beta_gamma_vaccination.py
python contagion/predator_slow_vaccination.py

# 3D extension
python 3d/flocking3d.py
python 3d/flocking3d_predator.py
python 3d/flocking3d_vaccination.py
python 3d/flocking3d_segregation.py
python 3d/flocking3d_slow_vaccination.py
python 3d/flocking3d_transect.py

# Collective decision-making / leadership (F72-F86)
python leadership.py
python collective/conflicting_leaders.py
python collective/conviction.py
python collective/decision_time.py
python collective/rotate_leaders.py
python collective/moving_goal.py
python collective/led_encirclement.py
python collective/panic_leadership.py
python collective/adversarial_leaders.py
python collective/leader_scaling.py
python collective/many_wrongs.py
python collective/correlated_estimates.py
python collective/noisy_minority.py
python collective/misinformation.py
python collective/moving_goal_crowd.py
```

For the full set of scripts and what each tests, see the per-finding sections of [`findings.md`](findings.md).

Open `sim_demo.html` in a browser for a real-time interactive simulation with adjustable parameters.

---

## Tools

- **Language:** Python 3 — numpy, matplotlib, reportlab
- **AI assistance:** Claude (Anthropic) — code generation, debugging, research guidance. All AI use documented in research log.

---

*Charbonneau, P. (2017). Natural Complexity: A Modeling Handbook. Princeton University Press.*  
*Silverberg et al. (2013). Collective motion of humans in mosh and circle pits. Physical Review Letters, 110, 228701.*
