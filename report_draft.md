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
against three analytically tractable limiting cases, I derive an exact analytical result:
the equilibrium cruise speed of an aligned flock is v_eq = v0 + alpha/mu, not the nominal
target speed v0. Finite-size scaling at multiple compactness values (C = 0.10 to 0.78)
shows no diverging susceptibility, indicating a smooth crossover rather than a true phase
transition. I then extend the model in two directions: external predation and internal
panic. Under sustained predator pressure, flocking prey maintain near-perfect alignment
(Phi ~ 1.0) while non-flocking prey scatter (Phi ~ 0.1). Three predator strategies form
a clear hierarchy: naive (chase CoM) and coordinated (mutual repulsion) predators fail
to disrupt the flock; only explicit encirclement — assigning each predator a fixed
compass angle — substantially reduces coherence (Phi = 0.77 at n_pred = 6). Cluster
analysis reveals this disruption is flock DIVISION into coherent sub-flocks rather than
dissolution, and the division is fully transient: all sub-flocks reunite within ~10 time
units of predator removal. Internal stressors behave differently: statically labeled
panicked agents fail to disrupt the calm sub-flock (calm_Phi ~ 1.0 at 20% panic), but
adding contact-mediated panic contagion drives the entire population to panic at any
non-zero contagion rate, collapsing global coherence. Predation produces reversible
damage; contagion produces absorbing damage. The primary function of flocking under
stress is coherence maintenance, and the collective's principal vulnerabilities are
multi-angle pressure and contagious internal state change.

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

This report covers eight investigations. First, I validate the implementation and
establish baseline behavior through limiting cases. Second, I sweep the noise and
alignment parameters to characterize the transition to flocking. Third, I examine whether
the repulsion-only transition constitutes a true phase transition using finite-size
scaling at multiple compactness values. Fourth, I extend the model with a predator agent
and characterize collective evasion. Fifth, I compare three predator strategies (naive,
coordinated, encircling) and identify encirclement as the only one capable of
substantial disruption. Sixth, I diagnose the encirclement disruption as transient flock
division and demonstrate full recovery after predator removal. Seventh, I bound the
minimum flock size at which collective behavior can form. Eighth, I implement internal
stressors — static panic and contagious panic — and contrast them with the external
predator results.

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

### 4.5 Predator Coordination and Encirclement

The naive-predator result (Section 4.4) raised an obvious question: if multiple predators
fail to disrupt the flock because they co-localize, what happens when they are forced apart?
I added an explicit predator-predator repulsion with strength alpha_coord and ran three
experiments (coordinated_predators.py, 8 seeds). At low coordination (alpha_coord < 5) the
predators still pile up at the prey center of mass — the shared target overwhelms the
mutual repulsion. Once alpha_coord >= 5 a real separation emerges (mean predator-predator
distance rises to 0.14 at alpha_coord = 5 and 0.29 at alpha_coord = 20). The predators are
spatially distributed and on average slightly closer to the flock (min_dist drops from
0.105 to ~0.08). But the flock order parameter never falls below 0.92 across all tested
predator counts and coordination strengths. The collective evasion response scales with
the number of approaching predators: the flock just builds a slightly tighter shell.

I then tested a fundamentally different strategy: encirclement. Each predator k is
assigned a fixed compass angle theta_k = 2*pi*k/n_pred and chases the point
CoM + R_enc*(cos theta_k, sin theta_k) rather than the CoM itself. With n_pred predators
this places them at n_pred equally spaced angles around the flock at radius R_enc. A
radius sweep at n_pred = 3 identifies R_enc = 0.15 as the optimal offset — just inside
the typical flock radius Rg ~ 0.25. Smaller R_enc and the predators still co-localize;
larger R_enc and they orbit outside the flock entirely. At n_pred = 6, the encirclement
strategy achieves Phi = 0.769 +/- 0.093 — the first substantial coherence reduction
across all predator experiments. Neither naive nor coordinated predators had ever pushed
Phi below 0.92.

### 4.6 Fragmentation Mechanism

To understand what the Phi drop under encirclement physically represents, I added
connected-components clustering (fragmentation.py): two agents are placed in the same
cluster if they are within the flocking radius r_f of one another. A naive multi-predator
setup gives Phi = 0.997 with 60 clusters — many small spatial groups all moving in the
same direction. The low cluster count of a normal flock is somewhat misleading: it
reports spatial structure, not directional structure. Under encirclement, the picture
inverts: cluster count drops to 24 (FEWER, LARGER groups) but global Phi falls to 0.72.
Crucially, each individual cluster has local order parameter ~0.997 — internally
coherent. The flock has fragmented into a small number of sub-flocks each heading in a
different direction. As n_pred increases, cluster count decreases further and the
largest cluster grows: predators compress the flock spatially while splitting it
directionally. This is flock DIVISION, not DISSOLUTION — a herding effect analogous to
wolf-pack predation or dolphin bait-ball formation, achieved purely through
multi-directional pressure without any explicit cooperative behavior between predators.

Scaling the protocol with N (encirclement_scaling.py) finds no simple law. At fixed
n_pred = 6, larger flocks are more resistant (Phi rises from 0.69 at N = 50 to 0.90 at
N = 350) — the dilution effect from Section 4.4 partially protects them. But at fixed
predator-to-prey ratio (n_pred = 9, N = 500), Phi falls to 0.654 — the worst result in
the entire project. The relevant variable is not predator count, not predator-to-prey
ratio, but ANGULAR COVERAGE. Both N = 100 and N = 350 converge to a common floor
Phi ~ 0.67 at n_pred ~ 10, suggesting that ten equally spaced angles is sufficient to
overwhelm the alignment force regardless of flock size.

### 4.7 Reversibility of Encirclement Damage

Finding that encirclement causes flock division raised an immediate follow-up question:
is the division permanent? On a periodic torus, sub-flocks heading in different
directions inevitably re-encounter each other; the alignment force would then re-couple
them. To test this I ran a three-phase protocol (reunion.py): a 1500-step pure-flock
warm-up, a 4000-step attack with n_pred = 10 encircling predators, and a 6500-step
recovery with predators removed. During the attack, Phi falls to 0.72 with 4.5 clusters
and largest cluster fraction 0.41 — substantial fragmentation. Immediately after
predator removal, the sub-flocks merge. Across 6 seeds, every single run recovers to
Phi >= 0.95 within 4.5 to 16.5 time units (mean 10.3) — much shorter than the 4000-step
attack that caused the damage. By the end of the recovery window, Phi has settled to
1.000 +/- 0.001 with a single coherent cluster.

This result establishes that encirclement is a fundamentally REVERSIBLE form of
disruption. The flock's topological state — a single connected population — is
preserved through the attack and re-emerges as soon as the stressor lifts. The damage
is purely kinematic, not structural.

### 4.8 Minimum Viable Flock Size

The dilution result of Section 4.4 implied that larger flocks are more resistant, but
the smallest tested case was N = 10. Below what N does collective evasion fail entirely?
A sweep from N = 3 to N = 100 (min_flock_size.py, 8 seeds) shows that flock formation
itself is the binding constraint: in the no-predator control, Phi = 0.49 at N = 3, rises
to 0.69 at N = 8, crosses 0.9 between N = 18 and N = 25, and reaches 0.99 at N = 100.
Below N ~ 12 the flock is unreliable (std across seeds 0.13-0.20). Predator presence
does not substantially shift this threshold: with a naive predator, Phi(N) follows the
same curve. With two opposing encircling predators, very small flocks (N = 3-8) are
slightly stabilized — the predator pressure forces nominal alignment — but the
underlying coherence threshold remains near N ~ 18-25. The interpretation is that
flocking is a collective effect with a minimum participant count: below ~12 agents in a
unit domain, the local spatial density is too sparse for the flocking force to dominate
noise. Real prey populations near this threshold should be most vulnerable to predators,
though the absolute capture rate in this simulation remains low across all sizes.

### 4.9 Sensing-Limited Predators

Up to this point all predators have had perfect global knowledge of the flock center of
mass. To test the more biologically realistic case of bounded perception, I added a
sensing radius r_sense within which the predator can locate the nearest prey
(predator_sensing.py). When the nearest prey is within r_sense, the predator chases
normally; outside r_sense it executes a slow random walk. Sweeping r_sense from 0.05 to
infinity produces a sharp transition near r_sense ~ flock radius Rg (0.10-0.15). Below
the threshold the lock-on fraction drops to 12% and the predator only finds the flock
occasionally; above r_sense = 0.20 the lock-on fraction reaches 100% and the result is
indistinguishable from perfect sensing. For single naive predators, bounded perception
makes no difference whenever r_sense > 0.20. Interestingly, for encirclement with n = 6
predators, limited sensing actually WORSENS the outcome for the flock (Phi = 0.79 vs
0.85): when an encircling predator loses the flock and re-enters search mode, its
subsequent re-approach is from a random angle, adding unpredictable multi-directional
pressure on top of the structured encirclement pattern.

### 4.10 Internal Stressors: Panic and Contagion

Charbonneau Section 10.5 ("Why You Should Never Panic") introduces agents with reduced
flocking and elevated noise — panicked agents that should disrupt the surrounding crowd.
I implemented this directly (panic.py) by labeling a fraction f of agents as panicked
(alpha_panic = 0.1, ramp_panic = 10) and sweeping f from 0 to 20%. The global order
parameter drops smoothly: Phi = 1.000 at f = 0% to Phi = 0.853 at f = 20%. But this drop
is pure dilution. Computing the order parameter using only the calm agents reveals that
calm_Phi remains at 0.999 across the entire sweep. The calm flock effectively ignores its
panicked neighbors — the alignment force is strong enough that calm agents maintain
coherence even when 1 in 5 of their neighbors is moving erratically.

A natural objection is that real panic is contagious: panicked agents make their
neighbors panic too. I added this mechanism (panic_contagion.py). Each calm agent within
a contagion radius r_cont = 0.05 of a panicked agent transitions to the panicked state
with probability per timestep p = 1 - exp(-beta * k * dt) where k is the local count of
panicked neighbors and beta is the contagion rate. Without recovery (an SI dynamics),
any non-zero beta drives the entire flock to f = 1.0 — saturation occurs in roughly
4/beta time units. With f = 1, all agents have the panicked alignment and noise
parameters, and the global Phi collapses to ~0.10. The "calm-flock immunity" from
panic.py was an artifact of treating panic as a fixed label: once the pool of calm
agents can be drained, it always is. The book's claim that panic is collectively
dangerous is recovered, but only conditional on contact-mediated propagation. Whether
a true epidemic threshold beta_c exists in an SIS model with recovery (calm <-> panic)
is an open question for future work.

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

This study produced eight main results:

1. **Equilibrium speed:** The cruise speed of an aligned flock is v_eq = v0 + alpha/mu,
   exactly. This is a direct consequence of the force equations and must be accounted
   for when comparing simulations at different parameter values.

2. **Phase transition is a crossover:** The solid-to-fluid transition in the
   repulsion-only system is a smooth crossover at all tested compactness values
   (C = 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.78), not a true phase transition.
   Susceptibility never peaks at a finite noise amplitude; KE/N is N-independent in
   every regime tested. The absence of a critical point is a universal feature of this
   model's soft repulsion, not an artifact of any density regime.

3. **Flock coherence under predation:** Flocking prey maintain near-perfect velocity
   alignment under sustained predator pressure while non-flocking prey scatter. Evasion
   distance saturates at a minimum buffer value regardless of predator aggression.

4. **Predator co-localization and elongation:** Multiple predators using the "chase CoM"
   rule converge to the same location (measured separation ~0.001), producing combined
   repulsion that paradoxically increases evasion distance and drives flock elongation.
   The elongation is real but its cause is force concentration, not strategic shape
   adaptation.

5. **Predator-strategy hierarchy:** Three predator strategies form a clear hierarchy
   of effectiveness against the flock. Naive predators co-localize and fail (Phi > 0.97
   for n_pred = 4). Coordinated predators with mutual repulsion spread out but still
   fail (Phi > 0.92 at n_pred = 10). Encirclement — explicit assignment of fixed
   compass angles to each predator — is the only strategy in this model capable of
   substantial disruption (Phi = 0.77 at n_pred = 6, Phi ~ 0.67 floor at n_pred = 10).

6. **Encirclement causes transient division:** Under encirclement, the flock divides
   into a small number of internally coherent sub-flocks heading in different
   directions (flock DIVISION, not flock DISSOLUTION). When predators are removed, all
   sub-flocks reunite within ~10 time units. The damage is purely kinematic and fully
   reversible.

7. **Internal stressors require contagion to matter:** Statically labeled panicked
   agents do not propagate disruption — the alignment force keeps calm neighbors
   coherent even at 20% panic fraction. Once a contact-mediated contagion mechanism is
   added, any non-zero contagion rate drives the population to total panic and the
   flock collapses. Predation produces reversible damage; contagion produces absorbing
   damage. The two stressor classes are qualitatively different.

8. **Minimum viable flock size:** Flock coherence requires N ~ 18-25 in a unit domain.
   Below N ~ 12 the flock is unreliable even without a stressor; the dilution-based
   "safety in numbers" hypothesis has a lower limit below which collective evasion
   cannot work because the collective itself cannot form.

Taken together, these results suggest that the primary function of the alignment force
in this model — and possibly in biological flocking — is to maintain a single
coordinated collective response to whatever stressor is encountered. Flock coherence is
remarkably robust to most disruption modes (noise, aggressive predators, internal
panic without contagion), and even when broken by encirclement it reconstitutes
spontaneously. The two ways to truly damage the collective in this model are
multi-angle predator pressure (transient) and contact-mediated panic contagion
(absorbing).

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
| compactness_search.py | Intermediate-compactness phase search (C=0.15..0.60) |
| coordinated_predators.py | Predator-predator repulsion experiments |
| encirclement.py | Encirclement strategy, radius sweep, flock-breaking threshold |
| encirclement_scaling.py | Encirclement threshold vs flock size N |
| fragmentation.py | Sub-cluster detection and division-vs-dissolution analysis |
| reunion.py | Sub-flock reunion after predator removal (recovery test) |
| min_flock_size.py | Minimum N for collective coherence and evasion |
| predator_sensing.py | Limited sensing radius, search/attack phases |
| panic.py | Static panic fraction sweep |
| panic_contagion.py | SI panic contagion (no recovery) |
| contagion_sis.py | SIS panic contagion with recovery rate gamma |
| hybrid_stressors.py | Combined predation + contagion |
| segregation.py | Active/passive segregation (mixed v0 populations) |
| model.py | OOP foundation: Flock and Predator classes for new experiments |
