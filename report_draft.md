# Emergent Flocking and Collective Evasion in a Force-Based Agent Model

**PHY 351 — Independent Summer Research**
Nathan Langley
May 2026
Advisor: Prof. Ian Beatty

---

## Abstract

I present computational simulations of a force-based flocking model (Charbonneau, 2017)
in which N agents on a periodic two-dimensional domain interact through repulsion,
velocity alignment, self-propulsion, and random noise. After validating the implementation
against three analytically tractable limiting cases, I characterize the parameter space
through systematic sweeps and identify an exact analytical result: the equilibrium cruise
speed of an aligned flock is v_eq = v0 + alpha/mu, not the nominal target speed v0.
Finite-size scaling of the repulsion-only system shows no diverging susceptibility,
indicating a smooth crossover rather than a true phase transition. I then extend the model
with a predator agent and find that flocking prey maintain near-perfect velocity alignment
(Phi ~ 1.0) under sustained predator pressure while non-flocking prey scatter completely
(Phi ~ 0.1). With multiple predators, flock coherence remains intact while flock
elongation increases substantially, suggesting shape adaptation as a collective evasion
strategy. These results demonstrate that the primary function of flocking under predation
is coherence maintenance, not distance maximization.

---

## 1. Introduction

One of the central puzzles in complex systems is how large-scale ordered behavior
emerges from purely local interactions. Flocking — the coordinated motion of birds,
fish schools, and animal herds — is a canonical example. Each individual follows simple
rules based on its immediate neighbors, yet the collective produces sweeping global
patterns with no central coordination. Understanding the conditions under which order
emerges, and how robust that order is to perturbation, has implications ranging from
evolutionary biology to crowd control.

The model studied here is based on Chapter 10 of Charbonneau (2017), which was
originally developed by Silverberg et al. (2013) to describe crowd dynamics in mosh pits
at heavy metal concerts. Each agent in the model is subject to four forces: short-range
repulsion, velocity-aligning flocking force, self-propulsion toward a target speed, and
random noise. The interplay of these four forces produces a rich behavioral phase space,
including crystalline order, disordered fluid motion, and coherent streaming flocks.

This report covers four main investigations. First, I validate the implementation and
establish baseline behavior through limiting cases. Second, I sweep the noise and
alignment parameters to characterize the transition to flocking. Third, I examine whether
the repulsion-only transition constitutes a true phase transition using finite-size scaling.
Finally, I extend the model with a predator agent and characterize collective evasion
behavior, including the effect of multiple simultaneous predators on flock geometry and
coherence.

---

## 2. Model

### 2.1 Setup

N agents move on a periodic unit square (x, y in [0, 1]), implemented as a torus so
that agents exiting one edge reappear on the opposite side. Agent positions and velocities
are updated at each timestep using the forward Euler method at dt = 0.01. The state of
agent j at time t is fully described by its position (x_j, y_j) and velocity (vx_j, vy_j).

### 2.2 Forces

The total force on agent j is a sum of four contributions (Charbonneau Eqs. 10.1-10.8):

**Repulsion.** A short-range force prevents agents from overlapping. It acts on pairs
within distance 2r0 and grows in intensity as agents approach:

    F_rep,j = eps * SUM_k [ (1 - r_jk / 2r0)^(3/2) * r_hat_jk ]    for r_jk <= 2r0

where r_jk is the distance between agents j and k, and r_hat_jk is a unit vector
pointing from k toward j.

**Flocking.** An alignment force drives the velocity of agent j toward the mean velocity
of its neighbors within a flocking radius r_f:

    F_flock,j = alpha * V_bar / |V_bar|,    V_bar = SUM_{k: r_jk <= r_f} v_k

The normalized form ensures the flocking force has constant magnitude alpha regardless
of how many neighbors are present.

**Self-propulsion.** A speed-correcting force drives agent j toward a target speed v0
along its current direction of motion:

    F_prop,j = mu * (v0 - |v_j|) * v_hat_j

where v_hat_j is the unit vector along v_j. This force accelerates agents moving too
slowly and brakes those moving too fast.

**Random noise.** Each component of the random force is drawn independently from a
uniform distribution on [-ramp, ramp] at each timestep.

With unit mass for all agents, Newton's second law gives a_j = F_j, and the equations
of motion are integrated as:

    x_j(t + dt) = x_j(t) + v_j(t) * dt
    v_j(t + dt) = v_j(t) + F_j(t) * dt

### 2.3 Periodic Boundary Implementation

Force calculations near domain boundaries require special handling. Agents within range
r_f of any boundary are replicated as ghost copies on the opposite side, so that the
flocking and repulsion forces computed for a real agent account correctly for neighbors
across the periodic boundary. This buffer zone approach follows Charbonneau Fig. 10.2.

### 2.4 Metrics

The primary measure of collective order is the **order parameter**:

    Phi = | (1/N) SUM_j v_hat_j |

Phi = 1 corresponds to perfect velocity alignment; Phi = 0 to randomly oriented motion.
I also track total kinetic energy KE = (1/2) SUM_j |v_j|^2 and, for flock geometry,
the **radius of gyration** Rg (root-mean-square distance from center of mass) and the
**aspect ratio** AR (ratio of the major to minor eigenvalue of the spatial covariance
matrix, measuring elongation).

### 2.5 Default Parameters

Unless otherwise noted, simulations use the parameters from Charbonneau Table 10.1:
N = 350, r0 = 0.005, eps = 0.1, r_f = 0.1, alpha = 1.0, v0 = 1.0, mu = 10.0,
ramp = 0.5, dt = 0.01.

---

## 3. Validation

Before drawing any conclusions from the simulations, I verified the implementation
against three limiting cases with known expected behavior.

**Case 1: Pure random walk.** With all physical forces disabled (eps = alpha = mu =
v0 = 0, ramp = 1), agents should perform a pure random walk with no preferred direction.
The measured order parameter was Phi = 0.04 (expected ~0) and agent positions spread
uniformly across the domain (standard deviation ~0.29, consistent with uniform
distribution). This confirms the integration and boundary conditions are working.

**Case 2: Repulsion and noise only.** With alpha = 0 and v0 = 0 (self-propulsion acts
as a brake), the model reproduces Fig. 10.5 from Charbonneau: at low noise (eta = 1)
agents pack into a close-packed quasi-hexagonal structure, while at high noise (eta = 30)
the arrangement disorders into a fluid. Phi remains near zero throughout (no alignment
force), as expected.

**Case 3: Flocking only.** With eps = 0 and v0 = 0, the alignment force alone should
drive agents into a coherent stream. The final order parameter was Phi = 0.998 after
t = 30, confirming that a single coherent flock forms from random initial conditions,
consistent with Fig. 10.6 from Charbonneau (Fig. 1).

---

## 4. Results

### 4.1 Equilibrium Cruise Speed

An exact result follows directly from the force equations. In a perfectly aligned flock,
all agents move in the same direction with the same speed. The flocking force then acts
purely in the forward direction with magnitude alpha. The self-propulsion force balances
this when:

    alpha + mu * (v0 - v_eq) = 0    =>    v_eq = v0 + alpha/mu

With the default parameters (alpha = 1, mu = 10, v0 = 1), this predicts v_eq = 1.10.
I verified this prediction by measuring steady-state mean speed across four values of
alpha with v0 = 1, mu = 10 fixed. Measured speeds agreed with the prediction to within
0.002 in all cases (Fig. 2). The implication is that v0 and alpha are not independent
knobs for cruise speed: to achieve a target cruising speed v_c, one must set
v0 = v_c - alpha/mu.

### 4.2 Flock Formation

Sweeping the flocking amplitude alpha with noise fixed at ramp = 0.1 (5 seeds per
point, error bars represent standard deviation) shows a sharp onset of flocking near
alpha ~ 0.05. At alpha = 0, Phi = 0.09 +/- 0.01. By alpha = 0.05, Phi = 0.40 +/- 0.12,
and by alpha = 0.20, Phi = 0.89 +/- 0.03. The large run-to-run variance near the
threshold (std ~ 0.1-0.2 for 0.05 <= alpha <= 0.15) indicates sensitivity to initial
conditions near the onset. Above alpha ~ 0.2, flocks form reliably (Fig. 3).

With all forces active and the default alpha = 1, flock coherence is robust: Phi exceeds
0.99 up to noise amplitude ramp = 3, exceeds 0.97 at ramp = 5, and drops below 0.5
only at ramp ~ 20. The alignment force makes the system dramatically more resistant to
noise disruption than the repulsion-only case.

### 4.3 Nature of the Solid-to-Fluid Transition

In the repulsion-only system (alpha = 0, v0 = 0), kinetic energy rises with noise
amplitude, suggesting a transition from a solid-like crystalline state to a fluid-like
disordered state. To test whether this constitutes a true phase transition, I performed
finite-size scaling across N = 25, 50, 100, and 200 (Fig. 4).

A true phase transition would produce KE/N curves that depend on N, with a critical
point (susceptibility peak) that converges to a finite eta_c as N increases. Instead,
the KE/N curves are essentially identical for all four system sizes, and the
susceptibility chi = N * var(KE/N) increases monotonically with eta with no peak.

This indicates a smooth crossover rather than a true critical phenomenon. The physical
picture is consistent with the high compactness of the system (C = pi*N*r0^2 ~ 0.78
for the parameters used): each agent is effectively caged by its neighbors and oscillates
harmonically around a fixed lattice site. This produces KE proportional to eta^2,
independent of N — behavior characteristic of uncoupled harmonic oscillators, not a
correlated system approaching criticality.

**Fixed-compactness scaling.** The original finite-size scaling held r0 fixed, meaning
compactness C = pi*N*r0^2 grew with N (C = 0.196 for N = 25 up to C = 1.57 for N = 200).
To isolate the effect of compactness, I repeated the analysis holding C fixed by scaling
r0 = sqrt(C/(pi*N)) for each N. Testing both C = 0.78 (dense) and C = 0.10 (dilute),
the result in both cases is the same: KE/N curves are N-independent and the susceptibility
chi = N * var(KE/N) increases monotonically to the top of the sweep (eta = 30) with no
peak at finite eta. The KE/N values at the two compactness levels are also nearly
identical to each other (Fig. 9).

The crossover is therefore not an artifact of high compactness alone. In the dilute
regime, agents barely interact (mean spacing exceeds repulsion range), so they behave
as essentially independent random walkers; KE/N is set solely by the noise amplitude
and is N-independent for the same reason as in the dense case — just via a different
physical mechanism. Both extremes suppress cooperative behavior: too dense means
agents are caged; too dilute means they never interact enough to form a solid. A genuine
critical point would require an intermediate regime where a solid phase can form and
agents can rearrange cooperatively. Whether such a regime exists in this model at
compactness values between 0.10 and 0.78, or at noise amplitudes above eta = 30,
remains an open question.

### 4.4 Predator-Prey Dynamics

I extended the model with a predator agent that chases the prey center of mass via a
strong alignment force (alpha_pred = 5) and generates a long-range repulsive force on
nearby prey (r0_pred = 0.1). Prey parameters were set to the slow-walking regime
(v0 = 0.02, alpha = 1.0, ramp = 0.1) to match the concert crowd context from
Silverberg et al.

**Flock coherence under pressure.** Comparing flocking prey (alpha = 1) versus
non-flocking prey (alpha = 0) across 10 random initializations shows a striking
divergence. Flocking prey maintain Phi ~ 0.998 throughout the simulation despite
continuous predator pressure. Non-flocking prey scatter almost immediately, reaching
Phi ~ 0.096 in steady state (Fig. 5). Non-flocking agents maintain marginally more
individual distance from the predator (0.127 vs. 0.112), but they lose all collective
structure. The flock absorbs the disturbance while remaining coherent.

**Evasion distance saturates.** Sweeping predator aggression alpha_pred (which sets
effective predator speed as v_eq,pred = v0_pred + alpha_pred/mu_pred) reveals that
the mean predator-to-nearest-prey distance drops from 0.24 with a passive predator to
~0.10 for alpha_pred >= 1 and then saturates — the collective repulsion response
establishes a minimum buffer distance that persists regardless of predator aggression.

**Flock geometry.** The flock is not just a point moving through space; its shape
matters. Without a predator, the steady-state aspect ratio is AR = 2.61 and radius of
gyration Rg = 0.215. With a predator, these shift modestly to AR = 2.76 and Rg = 0.274.
Stronger flocking (larger alpha) produces substantially more elongated flocks: AR
increases from 2.09 at alpha = 0.2 to 7.27 at alpha = 2.0. These highly elongated
configurations resemble the arched, thinning flocks predicted by Charbonneau Exercise 6
(Fig. 6).

**Multiple predators.** With 1 to 4 simultaneous predators, flock order parameter
stays near 0.975-0.991 — coherence is maintained across the entire range (Fig. 7).
Aspect ratio rises substantially with predator count (AR = 2.83 with one predator,
AR = 7.91 with three), while Rg increases modestly. Counterintuitively, the minimum
predator-to-prey distance increases slightly as the number of predators grows
(0.093 for one predator, 0.106 for three). The flock under multiple predators is
more elongated but no more accessible to any individual predator.

**Why evasion distance increases with predator count.** To diagnose the counterintuitive
evasion result, I measured the predator-predator separation and the orientation of the
flock major axis relative to the predator centroid direction across 8 random initializations
(evasion_analysis.py). The predator-predator mean distance was approximately 0.001 for
all multi-predator runs — effectively zero. Because every predator independently targets
the flock center of mass using the same rule, they all converge to the same location and
pile up on top of one another. This co-localization means multiple predators do not
approach from different directions; they compete for the same point. Flock orientation
relative to the predator centroid was 43-46 degrees across all conditions — consistent
with the 45-degree expectation for random alignment — confirming that the flock does not
systematically orient its narrow or broad side toward the predator. The evasion distance
improvement therefore arises mechanically: co-localized predators collectively apply
repulsion from a single point, and this concentrated repulsion is stronger than what a
single predator can produce, pushing the flock slightly farther away. This is a
model-specific artifact of the naive "chase CoM" predator strategy: multiple independent
pursuers using identical rules undermine each other rather than coordinating (Fig. 8).

---

## 5. Discussion

The most striking result of the predator simulations is that flocking is not primarily
a strategy for maximizing distance from a predator. Non-flocking agents individually
maintain slightly greater separation from the predator, yet the flocking agents clearly
have a more robust collective response: they remain coordinated, move in concert, and
hold a consistent buffer distance. This distinction — coherence versus distance — may
reflect something real about the function of biological flocking. A coherent flock can
mount a coordinated escape response; scattered individuals cannot.

The increasing aspect ratio under multiple predators is interesting. When pressure
arrives from multiple directions, the flock elongates rather than fragmenting. The
diagnostic analysis (Section 4.4) shows this is not because predators actually approach
from multiple directions — they converge to the same point — but rather because the
stronger combined repulsion from co-localized predators drives a more intense elongation
response. The elongated shape is a real emergent effect, but its cause is the
concentration of force at one point, not a strategic response to multi-directional threat.

The equilibrium speed result (v_eq = v0 + alpha/mu) is an exact consequence of the
force equations that Charbonneau does not explicitly note. It means a researcher using
this model who sets v0 = 1 and alpha = 1 expecting agents to cruise at speed 1 will
find them consistently at speed 1.1. For simulations where absolute speed matters
(e.g., comparing flocking and non-flocking agents under identical predator pressure),
this correction is necessary.

The phase transition result extends the original finding. Fixing compactness properly via
r0 = sqrt(C/(pi*N)) and testing a dilute regime (C = 0.10) shows the same behavior as
the dense regime: N-independent KE/N and monotone susceptibility with no finite-eta peak.
The absence of a critical point is not a consequence of high compactness alone. Instead,
both extremes fail for different reasons — too dense means caged oscillators, too dilute
means non-interacting walkers. A true phase transition in this model, if it exists,
would require an intermediate compactness where a solid phase can form and cooperative
rearrangements are possible. The model's smooth crossover may be a general feature of
this force-based formulation rather than a regime-specific artifact.

---

## 6. Conclusions

This study produced four main results:

1. **Equilibrium speed:** The cruise speed of an aligned flock is v_eq = v0 + alpha/mu,
   exactly. This is a direct consequence of the force equations and must be accounted
   for when comparing simulations at different parameter values.

2. **Phase transition:** The solid-to-fluid transition in the repulsion-only system is a
   smooth crossover, not a true phase transition. Finite-size scaling shows no N-dependent
   critical point, consistent with independent harmonic oscillator behavior in a
   high-compactness system.

3. **Flock coherence under predation:** Flocking prey maintain near-perfect velocity
   alignment under predator pressure while non-flocking prey scatter. Evasion distance
   saturates at a minimum buffer value regardless of predator aggression.

4. **Collective geometry:** Flocks become more elongated under both stronger internal
   alignment and greater predator pressure. Multiple predators elongate the flock without
   degrading coherence; evasion distance counterintuitively improves.

Taken together, these results suggest that the primary function of the alignment force
in this model — and possibly in biological flocking — is not to keep individuals far
from threats, but to maintain coordinated collective response.

---

## References

Charbonneau, P. (2017). *Natural Complexity: A Modeling Handbook*. Princeton University Press.

Silverberg, J. L., Bierbaum, M., Sethna, J. P., and Cohen, I. (2013). Collective motion
of humans in mosh and circle pits at heavy metal concerts. *Physical Review Letters*,
110, 228701.

---

## Appendix: Code

All simulation code is available at https://github.com/ninjahawk/Summer_Research

| File | Description |
|------|-------------|
| flocking.py | Core model: buffer zone, vectorized force function, run loop, metrics |
| analysis.py | Validation limiting cases and parameter sweeps |
| predator.py | Single-predator extension with 4 experiments |
| phase_transition.py | Finite-size scaling of solid-to-fluid transition |
| geometry.py | Radius of gyration and aspect ratio analysis |
| multi_predator.py | Multi-predator experiments |
| evasion_analysis.py | Predator co-localization and evasion distance diagnostic |
| compactness_phase.py | Fixed-compactness finite-size scaling across C=0.10 and C=0.78 |
