# PHY 351 Independent Summer Research
## Flocking Dynamics and Predator-Prey Interactions in a Force-Based Agent Model
### Nathan Langley

---

## 1. Introduction

Collective motion -- flocks of birds, schools of fish, herds of mammals -- is one
of the most visually striking examples of emergent behavior in nature. Complex,
coordinated global patterns arise from simple local rules followed by each individual,
with no central control. Understanding how this happens, and what governs the
transition between ordered and disordered motion, is a core problem in the
physics of complex systems.

This report presents computational simulations of a flocking model based on
Chapter 10 of Charbonneau (2017). The model was originally developed to study
human crowd dynamics at rock concerts (Silverberg et al., 2013) but applies equally
to biological flocking. Each agent follows four simple force rules: avoid neighbors
who are too close, align velocity with nearby neighbors, maintain a target speed,
and respond to random perturbations.

I replicated the baseline model, validated it against limiting cases, explored
the parameter space systematically, and extended it with a predator agent to study
collective evasion. Several unexpected findings emerged, including a relationship
between model parameters and equilibrium speed, and the counterintuitive result
that flock coherence is maintained -- and even enhanced in some respects -- under
predator pressure.

---

## 2. Model Specification

### 2.1 Domain and Agents

N agents move on a periodic unit square (x, y in [0,1]), implemented as a torus
(agents exiting one edge reappear on the opposite edge). Positions and velocities
are updated with forward Euler integration at timestep dt = 0.01.

### 2.2 Forces

Each agent j is subject to four forces at each timestep:

**Repulsion** (Eq. 10.1): Short-range force preventing overlap. Acts within
range 2*r0, intensity proportional to (1 - r/(2*r0))^1.5.

    F_rep = eps * sum_k [ (1 - r_jk/(2*r0))^1.5 * r_hat_jk ]   for r_jk <= 2*r0

**Flocking** (Eq. 10.2-10.3): Aligns velocity with the mean velocity of neighbors
within range r_f.

    F_flock = alpha * V_bar / |V_bar|
    V_bar = sum of v_k for all k within r_f of j

**Self-propulsion** (Eq. 10.4): Drives agent toward target speed v0 along its
current velocity direction.

    F_prop = mu * (v0 - |v_j|) * v_hat_j

**Random** (Eq. 10.7): Uniform random force in [-ramp, ramp] per component.

    F_rand = eta_j   (each component uniform in [-ramp, ramp])

Total force: F_j = F_rep + F_flock + F_prop + F_rand
Acceleration: a_j = F_j / M  (M = 1 for all agents)

### 2.3 Default Parameters (Table 10.1)

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Agents | N | 350 | Number of agents |
| Timestep | dt | 0.01 | Integration step |
| Repulsion radius | r0 | 0.005 | Core repulsion range |
| Repulsion amplitude | eps | 0.1 | Repulsion force strength |
| Flocking radius | rf | 0.1 | Neighbor detection range |
| Flocking amplitude | alpha | 1.0 | Alignment force strength |
| Target speed | v0 | 1.0 | Self-propulsion target |
| Propulsion amplitude | mu | 10.0 | Speed correction strength |
| Random amplitude | ramp | 0.5 | Noise level |

### 2.4 Periodic Boundary Implementation

Forces near the domain boundaries require special handling. Agents within range rf
of a boundary are replicated as ghost agents on the opposite side, so that flocking
and repulsion forces wrap correctly. This "buffer zone" approach follows Charbonneau
Fig 10.2 exactly.

### 2.5 Order Parameter

Flock coherence is measured by the order parameter:

    Phi = | mean(v_hat_j) |

where v_hat_j is the unit velocity vector of agent j. Phi = 1 means all agents
move in exactly the same direction; Phi = 0 means velocities are randomly oriented.

---

## 3. Validation

Three limiting cases were run to verify the implementation before trusting any results.

### 3.1 Pure Random Walk (all physical forces off)

With eps=0, alpha=0, mu=0, v0=0, ramp=1.0: agents perform a pure random walk.
**Expected:** Phi near 0 (no preferred direction), positions spread uniformly.
**Result:** Phi = 0.04, x std = 0.277, y std = 0.303 (uniform gives 0.289). PASS.

### 3.2 Repulsion and Noise Only (alpha=0, v0=0)

With the flocking force off and self-propulsion acting as a brake: agents should
pack into a hexagonal crystal at low noise, and disorder into a fluid at high noise.
**Expected:** Reproduces Fig 10.5 from Charbonneau.
**Result:** At eta=1, agents form a close-packed quasi-hexagonal structure. At eta=30,
agents move freely in a fluid state. Phi stays low throughout (no alignment force),
consistent with expectation. PASS.

### 3.3 Flocking Only (eps=0, v0=0)

With only the alignment force active: agents should spontaneously align into a
coherent streaming flock.
**Expected:** Phi rising from ~0 to near 1.0 as flock forms.
**Result:** Final Phi = 0.998 after t=30. PASS.

---

## 4. Results

### 4.1 Equilibrium Speed

**Finding:** The equilibrium cruise speed of aligned agents is v_eq = v0 + alpha/mu,
not v0.

**Derivation:** In a perfectly aligned flock, the flocking force contributes alpha * v_hat
(a force in the direction of motion). Self-propulsion equilibrates when the net
forward force is zero: alpha + mu*(v0 - v_eq) = 0, giving v_eq = v0 + alpha/mu.

**Verification:** Measured across alpha = 0, 0.5, 1.0, 2.0 with v0=1, mu=10.
Predicted vs measured agreement within 0.002 in all cases.

**Implication:** The parameters v0 and alpha are not independent in controlling
cruise speed. To achieve a target cruise speed v_cruise, set v0 = v_cruise - alpha/mu.

### 4.2 Solid-to-Fluid Transition (Repulsion-Only System)

In the repulsion-only system (alpha=0, v0=0), KE/N rises with noise amplitude eta.
Finite-size scaling across N = 25, 50, 100, 200 shows:

- KE/N curves are essentially independent of N (all four N values give identical results)
- The susceptibility chi = N * var(KE/N) rises monotonically with eta and shows no peak
  within the tested range (eta = 0.5 to 20)
- The transition is a smooth crossover, not a sharp phase transition

The N-independence of KE/N means each agent responds to noise independently, like
a harmonic oscillator at its lattice site. This is consistent with a high-compactness
system (C ~ 0.78) where each agent is effectively trapped by neighbors and vibrates
about a fixed equilibrium. A true phase transition would require diverging susceptibility
at a finite critical eta, which is not observed here.

### 4.3 Flock Formation Threshold

The order parameter Phi rises sharply from near zero to ~0.7 between alpha=0 and
alpha=0.05. Above alpha~0.2, Phi reliably exceeds 0.89 across all random initializations.
The transition is sharp but has large run-to-run variance near threshold (std ~0.12-0.19
at alpha=0.10), indicating sensitivity to initial conditions near the critical point.

### 4.4 Noise Tolerance of the Full Model

With all forces active (default parameters), flock coherence remains above Phi=0.99
up to eta=3, above Phi=0.97 at eta=5, and above Phi=0.91 at eta=10. Coherence
collapses at eta~20 (Phi=0.42). The flocking force makes the system dramatically more
resistant to noise than the repulsion-only case.

### 4.5 Flock Geometry

Flock shape was characterized using two metrics:
- Radius of gyration Rg: sqrt(mean squared distance from center of mass)
- Aspect ratio AR: ratio of major to minor eigenvalue of the covariance matrix
  (AR=1 means circular, AR>>1 means elongated)

Without predator: Rg=0.215, AR=2.61 (moderately elongated in direction of motion).
With predator:    Rg=0.274, AR=2.76 (slightly more spread and elongated).

Stronger flocking amplitude alpha produces more elongated flocks: AR increases from
2.09 at alpha=0.2 to 7.27 at alpha=2.0. This makes physical sense -- stronger flocking
forces agents into tighter velocity alignment, producing a more needle-like flock.

### 4.6 Predator-Prey Dynamics

**Setup:** A single predator agent chases the prey center of mass using a flocking-like
force (alpha_pred=5), moves at target speed v0_pred=0.05, and generates long-range
repulsion (r0_pred=0.1) on nearby prey.

**Finding 1 -- Flock coherence under pressure:**
Flocking prey (alpha=1) maintain Phi=0.998 throughout the simulation under predator
pressure. Non-flocking prey (alpha=0) scatter to Phi=0.096 almost immediately.
The flock absorbs predator disturbance without breaking apart.

**Finding 2 -- Evasion distance saturates:**
Mean predator-to-nearest-prey distance drops from 0.24 (passive predator, alpha_pred=0)
to ~0.10 (alpha_pred >= 1) and then saturates -- further increasing predator aggression
does not bring the predator closer. The collective repulsion response establishes a
minimum buffer distance.

**Finding 3 -- Multiple predators increase flock elongation:**
With 1 to 4 predators, flock coherence (Phi) stays near 0.975-0.991 -- remarkably
stable. Aspect ratio increases from AR=2.83 (1 predator) to AR=7.91 (3 predators),
suggesting flocks elongate to present a smaller cross-section or thread between predators.
Minimum predator-prey distance actually increases slightly with more predators (0.093
to 0.106), suggesting the flock becomes better at maintaining a buffer with more pressure.

**Finding 4 -- Dilution effect:**
The fraction of the flock within predator threat range decreases with flock size:
N=10: 49%, N=25: 21%, N=100: 11%. Consistent with geometric dilution (safety in numbers).

---

## 5. Discussion

The central result of the predator-prey experiments is that flocking is not primarily
about maximizing individual distance from a predator. Non-flocking agents maintain
slightly more individual distance (0.127 vs 0.112), but they lose all collective
structure (Phi drops from 1.0 to 0.1). The flocking flock, by contrast, maintains
perfect velocity alignment under sustained predator pressure.

This suggests the adaptive value of flocking may be less about evasion mechanics
and more about information sharing and coordinated response. A coherent flock can
collectively sense and respond to a predator -- all agents aligned means all agents
can shift direction simultaneously if one is perturbed.

The finding that flock elongation increases with both alpha and number of predators
is consistent with the book's prediction of arched, thinning flocks under predator
pressure. Higher alpha produces more needle-like flocks (AR up to 7.27), which could
represent a strategy of threading through narrow gaps or presenting a smaller profile.

The equilibrium speed finding (v_eq = v0 + alpha/mu) is a direct consequence of the
force equations and represents an exact analytical result, not just an empirical
observation. It has practical implications for anyone using this model: the two
"independent" parameters v0 and alpha actually jointly determine cruise speed.

---

## 6. Conclusions

1. The Charbonneau flocking model produces coherent collective motion emerging from
   four simple local force rules -- no global coordination required.

2. Equilibrium cruise speed is v_eq = v0 + alpha/mu, not v0, due to the forward
   acceleration contributed by the flocking force in aligned motion.

3. The solid-to-fluid transition in the repulsion-only system is a smooth crossover,
   not a sharp phase transition -- finite-size scaling shows no N-dependent critical point.

4. Under predator pressure, flocking prey maintain perfect velocity alignment (Phi~1)
   while non-flocking prey scatter completely (Phi~0.1).

5. Multiple predators cause flock elongation (AR increasing with n_pred) while
   coherence and evasion distance remain approximately constant -- the flock adapts
   its shape to maintain its collective properties.

---

## References

Charbonneau, P. (2017). *Natural Complexity: A Modeling Handbook*.
Princeton University Press.

Silverberg, J.L., Bierbaum, M., Sethna, J.P., and Cohen, I. (2013).
"Collective motion of humans in mosh and circle pits at heavy metal concerts."
*Phys. Rev. Lett.*, 110, 228701.

---

## Appendix: Code

All simulation code is available at:
https://github.com/ninjahawk/Summer_Research

Key files:
- flocking.py    -- core model (buffer, force, run, analysis utilities)
- analysis.py    -- validation and parameter sweep experiments
- predator.py    -- predator-prey extension
- geometry.py    -- flock shape analysis
- multi_predator.py -- multiple predator experiments
- phase_transition.py -- finite-size scaling analysis
- findings.md    -- running findings summary
