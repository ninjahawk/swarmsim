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
dissolution. Over long time (300 time units), the encircled flock enters an intermittent
merge/split cycle rather than a steady fragmented state. Encirclement damage is fully
reversible: sub-flocks reunite within ~10 time units of predator removal. Internal
stressors obey different rules: static panic fails to disrupt the calm sub-flock (calm
Phi ~ 1.0 at 20% panic fraction), but SIS contagion with recovery rate gamma has a
clean epidemic threshold at beta/gamma ~ 1. Encirclement shifts this threshold by ~4%
by compressing local contact density, and the spatial herd-immunity threshold (~0.46) is
more than twice the mean-field prediction due to clustering in the contact network.
Critically, when encirclement drives a sub-threshold contagion into a transient endemic
state, removing the predators reverses kinematic fragmentation (in ~10 time units) but
leaves the epidemic intact for hundreds of time units. Predation produces reversible
kinematic damage; contagion produces lasting epidemic damage; the combination produces
damage that outlasts the predation event itself by an order of magnitude. An
adaptive encirclement strategy — predators that continuously track live flock Rg and
maintain R_enc = 0.5*Rg — cuts the fraction of time the flock spends in a highly
coherent state by 34% compared to a fixed-radius strategy, validating the universality
of the R_enc/Rg ~ 0.5 optimum across dynamical fluctuations. Neither degree-targeted nor spatially-targeted vaccination outperforms random: both require
p_immune ~ 0.46, because kinematic reorganization restores hub positions after high-degree
agents are immunized, and kinematic mixing scrambles the spatial distribution of immune
agents before the epidemic runs — defeating both targeting strategies by the same mechanism.
Sweeping the repulsion exponent from n = 1.5 (soft) to n = 12 (near hard-core) produces
identical finite-size scaling at all exponents. The smooth crossover is not caused by
potential softness; it is a consequence of non-equilibrium driving (uniform random kicks
without viscous dissipation). A follow-up Langevin simulation confirms that proper thermal
equilibration (FDT satisfied; KE/N = kT at equilibrium) is achievable within this model
family, but the kinetic energy susceptibility chi = N*Var(KE/N) is not sensitive to the
structural hard-disc melting transition. A final experiment measuring the hexatic order
parameter |psi_6| directly reveals that the n = 1.5 soft repulsion cannot crystallize at
any accessible temperature: |psi_6| ≈ 0.4 across the entire kT range from 0.001 to 5.0
with no N-dependent susceptibility peak. The KTHNY transition is absent because the soft
potential allows agents to pass through each other's cores at any finite kT, preventing
hexagonal lattice formation.

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

This report covers twenty-three investigations, producing forty-one numbered findings.
The first four sections establish the baseline: implementation validation, parameter
sweeps, finite-size scaling to test for a true phase transition, and flock geometry.
Sections five through nine develop the predator-strategy hierarchy — from naive
co-localization to coordinated spreading to encirclement — and characterize encirclement
as the only strategy capable of substantial disruption, acting through transient flock
division rather than dissolution. Sections ten through twelve turn to internal stressors:
minimum viable flock size, static and contagious panic, and segregation by agent
heterogeneity. Sections thirteen through sixteen examine the coupling between predator
and contagion stressors: hybrid-stressor interaction, epidemic-threshold shifts under
compression, spatial-network herd immunity, and the long-time dynamics of encirclement
including intermittent merge/split behavior and incomplete encirclement. Section
sixteen follows epidemic persistence after predator removal, revealing a two-timescale
asymmetry between kinematic recovery (~10 time units) and epidemic decay (~100+ time
units). Section seventeen validates the universal R_enc/Rg ~ 0.5 optimum through an
adaptive encirclement strategy, sections eighteen and twenty test whether degree-targeted
and spatially-targeted vaccination reduce the herd-immunity threshold (both null results),
section nineteen tests whether harder repulsion can produce a true phase transition (null),
section twenty-one tests whether Langevin dynamics recover the hard-disc structural
melting transition (thermal equilibration achieved; structural metric needed to detect it),
section twenty-two measures the hexatic order parameter directly (confirming it is the
correct diagnostic, but finding that n = 1.5 soft repulsion cannot crystallize at any
accessible temperature), and section twenty-three extends the model to three spatial
dimensions, confirming that flocking and the v_eq analytical result generalize cleanly to 3D.

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
is addressed below.

### 4.11 SIS Contagion and the Epidemic Threshold

Adding a recovery rate gamma (panicked agents return to calm with rate gamma per unit
time) restores the classical SIS phase structure (contagion_sis.py). A 2D sweep over
(beta, gamma) reveals a clean threshold along the diagonal beta = gamma. Below this
line, the seed outbreak dies out and the flock stays at Phi = 1.0. Above the line, an
endemic steady state emerges and the flock degrades smoothly with beta/gamma. At beta =
2, varying gamma from 0.1 to 10 takes the system from f_ss = 0.978 (outbreak persists,
Phi = 0.20) all the way down to f_ss = 0.000 (recovery wins, Phi = 1.0). The crossover
between gamma = 2 and gamma = 5 is sharp.

The location of the threshold beta_c ~ gamma is consistent with the standard mean-field
SIS prediction beta_c * <k> = gamma when the effective local contact count is <k> ~ 1.
For a uniform-density flock at N = 350 with r_cont = 0.05, the geometric estimate
gives <k> = pi * r_cont^2 * N ~ 2.7, but the flock is a spatially extended structure
whose effective contact rate at the contagion scale is closer to one.

This result reframes Finding 4.10. The SI claim that "any contagion is fatal" was
specifically a no-recovery limit (gamma = 0). With any finite recovery rate, the flock
can in principle contain a panic outbreak provided the contagion rate is below the
recovery rate. The flock's own alignment force may play the role of an effective gamma
in a fully physical model: panicked agents re-entering a coherent neighborhood feel
the alignment force pulling them toward the local mean velocity, which corresponds to
a return-to-calm process. In this sense, the kinematic flock-disruption problem maps
onto the classical contact-process problem of epidemiology via a single dimensionless
ratio beta/gamma.

### 4.12 Hybrid Stressors and Active/Passive Mixing

Two additional experiments tested how the various disruption mechanisms compose.

In the hybrid-stressor experiment (hybrid_stressors.py), I ran four conditions side-by-
side: no stressor, encirclement only, contagion only, and both. The encirclement-only
condition gave Phi = 0.71 (matching the Section 4.6 result). Contagion-only gave Phi =
0.05. Combined, Phi = 0.05 — identical to contagion alone. Encirclement adds nothing
because the population is fully panicked before encirclement-induced fragmentation
matures. The two disruption modes do not compose in the supercritical contagion regime:
the absorbing process simply wins.

In the segregation experiment (segregation.py), I tested whether mixed-speed populations
spatially segregate. With f_active = 0.5 and v0 contrast ranging from 0 (no contrast) to
0.9 (active 10x faster than passive), the segregation index — defined as the
along-heading position difference between groups in Rg units — stays at 0 +/- 0.05 in
every case. The alignment force homogenises group speed: an aligned flock with mixed
self-propulsion targets cruises at a population-weighted compromise speed, so individual
agents do not segregate into leading and trailing layers. To obtain spatial segregation
in this model would likely require asymmetric alpha values between groups, not v0
contrast alone.

A follow-up experiment (segregation_alpha.py) confirms this. Replacing v0 contrast with
alpha (alignment-strength) contrast produces real spatial segregation — but in a form
the along-heading index still misses. A local-purity diagnostic, defined as the
fraction of an agent's rf-neighbors that share its type, rises from 0.50 (random, no
contrast) to 0.73 at maximum contrast. The snapshot at alpha_passive=0 shows tight
clusters of high-alpha agents amid scattered low-alpha particles. The segregation is
isotropic in the flock frame, not preferentially at the leading or trailing edge.
Charbonneau's segregation result is therefore recovered, conditional on asymmetric
alignment fidelity rather than asymmetric speed, and the appropriate diagnostic is
local same-type purity rather than bulk position relative to heading.

### 4.13 Refinements: Sub-Threshold Coupling and Large-N Scaling

Two further experiments refine results from earlier sections.

The hybrid SIS experiment (hybrid_sis.py) tests whether encirclement can push a
contained SIS contagion over its critical threshold by raising the local contact count.
At beta = 1.0, gamma = 3.0 (beta/gamma = 0.33, well below the Section 4.11 threshold),
contagion alone fizzles: panic peaks at f_max = 0.13 and dies out, f_ss = 0. Adding 6
encircling predators triples the local contact count (k from 8.9 to 30.2) and doubles
the panic peak (f_max = 0.27), but the outbreak still dies out. The mechanism is
confirmed — compression raises effective beta — but the amplification is bounded:
well-subcritical contagion is not rescued by external mechanical pressure alone. The
combined-stressor Phi (0.73) is measurably worse than encirclement alone (0.86), so the
transient outbreak adds non-negligible kinematic disruption even when it eventually
collapses.

The large-N encirclement experiment (large_N_encirclement.py) tests whether the
Section 4.6 conjecture that the Phi ~ 0.67 floor at n_pred = 10 is N-independent
holds at very large flock sizes. At N = 350, 700, 1000 with R_enc = 0.15 held fixed,
the floor at n_pred = 10 rises with N: Phi = 0.667 (N = 350), 0.700 (N = 700),
0.740 (N = 1000). At n_pred = 14: 0.637, 0.727, 0.790. The encirclement strategy
becomes less effective at large N because R_enc was calibrated to N = 350's flock
radius Rg ~ 0.15; at N = 1000 the flock is broader and R_enc = 0.15 places the
predators inside the flock rather than at its boundary, where their repulsion is
absorbed by surrounding prey. The encirclement geometry must match the flock geometry
to disrupt — angular coverage alone is insufficient. A predator strategy that works
at intermediate flock sizes does not scale up to large prey aggregations without
re-calibration.

### 4.14 Near-Critical Coupling and Universal Encirclement Scaling

Three further experiments tighten the quantitative results above.

To directly measure the threshold shift hypothesised in Section 4.13, a sweep of beta
at fixed gamma=2.0 was repeated with and without 6 encircling predators
(critical_shift.py). The threshold beta_c (where f_ss crosses 0.5) was beta_c=1.93
without encirclement, 1.85 with — a leftward shift of 0.077, about 4% of the bare
value. The shift is largest just below threshold: at beta=1.5, f_ss rises from 0.34
(no encirclement) to 0.43 (with). Above threshold the two curves converge. The
modest size of the shift, despite a tripling of the local contact count under
encirclement (Section 4.13), is explained by redundancy: compressed sub-flocks have
panicked agents mostly surrounded by other panicked agents, so the new contacts are
not new infections. Encirclement is therefore a near-critical amplifier rather than
a general one; it can tip a contagion that is already close to its threshold but
cannot bridge a large gap.

A separate sweep tests the flock's resilience to immune sub-populations
(herd_immunity.py). At supercritical SIS (beta=2.5, gamma=2.0; R0=1.25), mean-field
theory predicts a herd-immunity threshold p_c = 1 - 1/R0 = 0.20. The measured
threshold in the flock model is p_c ~ 0.46, more than twice the mean-field value.
The cause is spatial clustering: panicked sub-clusters move together so their members
predominantly contact each other, inflating the effective local contact count within
clusters. Random immunity is then less effective than targeted immunity at breaking
transmission chains. This is a known feature of spatial epidemic models, and the
flock model reproduces it.

Finally, the apparent N-dependence of the encirclement floor (Section 4.13) is
resolved by sweeping R_enc at both N=350 and N=1000 (renc_scaling.py). When plotted
against R_enc / Rg the two Phi curves collapse: both show optimal disruption at
R_enc / Rg ~ 0.5, with Phi_min = 0.67 (N=350) and 0.73 (N=1000). The previous
N-dependence was almost entirely an R_enc / Rg mismatch — R_enc was tuned to
N=350's flock radius and was undersized for N=1000. With proper rescaling
encirclement is size-invariant. The strategy that wraps the flock at half its
radius works universally. The optimum at R_enc / Rg ~ 0.5 places each predator
within the bulk of the flock but not at the center, where it can push prey on all
sides toward neighboring predators — the kinematic geometry that drives the flock
division of Section 4.6.

### 4.15 Long-time Dynamics and Incomplete Encirclement

Two further experiments probe the limits of the encirclement strategy.

All encirclement experiments to this point ran for 4000 timesteps (40 time units).
To test whether the fragmented state is a true steady state or a transient on the way
to a different long-time attractor, I ran 30000 steps (300 time units) under constant
encirclement (long_encirclement.py, 4 seeds, n_pred=10, R_enc=0.15). The result is
neither sustained fragmentation nor recovery: instead, Phi oscillates continuously.
Within a single run, Phi excursions span from ~0.4 to ~0.95, with a temporal standard
deviation of ~0.21 per seed. The seed-to-seed variation is only 0.061. The temporal
fluctuations within each run dominate the between-seed variability, indicating that the
dynamics are intrinsically intermittent rather than converging to a steady state. Cluster
count and largest-cluster fraction oscillate in concert.

The mechanism is a positive feedback cycle. When the flock briefly re-coalesces
(Phi -> 0.9), all predators chase one CoM target offset by their respective angles —
maximum multi-directional pressure — and the flock quickly fragments again. Once
fragmented (Phi -> 0.5), the predators distribute among multiple sub-flocks rather than
all pressuring one, reducing the effective attack. The dominant sub-flock can
reassemble. The cycle repeats indefinitely; the long-time average Phi = 0.751 matches
the short-time value of 0.72 from Section 4.6, but the short-time measurement was
misleadingly quiet.

To test whether leaving a gap in the encirclement creates an escape route, I ran a
complementary experiment (encirclement_gap.py). Starting from a full 6-predator ring
(60-degree spacing), I removed predators one at a time and measured the flock's Phi
and whether its center-of-mass drift aligned with the gap direction. The result is
non-monotone and the gap direction is irrelevant. Full encirclement (n_active=6) gives
Phi = 0.918. Removing one predator to create a 120-degree gap drops Phi to 0.833 --
WORSE than the full ring. Removing two or three predators raises Phi back to 0.91-0.96.
The flock center-of-mass drift direction has no systematic alignment with the gap (mean
angular difference 70-110 degrees from gap orientation), confirming that agents have no
global awareness of where the ring is open.

The non-monotone result arises because one-predator removal creates a persistent
asymmetry: the five remaining predators keep re-encircling the shifted CoM, generating
irregular multi-directional pressure that is more disruptive than balanced 6-fold
pressure. Removing two or more predators reduces the angular complexity and lets the
flock partially escape. Counterintuitively, a near-complete encirclement with a single
gap is harder to escape than a full encirclement, and both are harder to escape than a
3- or 4-predator arc.

### 4.16 Outbreak Persistence After Predator Removal

Section 4.7 showed that pure encirclement damage reverses within ~10 time units once
predators are removed. Section 4.12 showed that supercritical SI contagion dominates
and eliminates the encirclement effect entirely. A regime between these extremes exists:
a contagion that is below the bare epidemic threshold (beta = 1.5, gamma = 2.0,
beta/gamma = 0.75 < 1.93) but is pushed into a transient endemic state by the
compression from encirclement. Does removing the predators allow both the kinematic
damage AND the contagion to reverse, or does the contagion outlast the kinematic stressor?

Running the three-phase protocol (outbreak_removal.py: warmup -> 4000 steps of
encirclement + SIS -> 5000 steps of recovery) answers this directly. During the
encirclement phase, local contact count rises by ~3x (Section 4.13) and the effective
transmission exceeds the recovery rate, establishing an endemic panic fraction of
f = 0.450 and suppressing Phi to 0.185. When predators are removed, kinematic recovery
begins -- Phi rises to 0.266 within 50 time units -- but the contagion does not collapse.
After 50 time units without predators, f = 0.413, barely changed from its peak. The
timescales are starkly asymmetric: kinematic damage heals in ~10 time units (Section 4.7);
contagion-driven damage persists for hundreds of time units as the system slowly evolves
back toward the bare endemic state (~0.34 panic fraction at these parameters).

The contrast illuminates the qualitative difference between the two damage types
introduced in Section 4.10. Pure encirclement damage is kinematic: sub-flocks that were
pushed apart simply re-merge on the periodic torus when the directional pressure
disappears. SIS contagion damage is a population state: the distribution of panicked vs.
calm agents evolves on epidemic timescales set by beta and gamma, not on kinematic
timescales set by flock size and swimming speed. A predator group that seeds a panic
cascade even transiently — even one that would not sustain itself in an unencircled flock
— inflicts damage that outlasts the predation event itself by an order of magnitude.

---

## 4.17 Adaptive Encirclement: Tracking Flock Geometry Increases Disruption (Finding 35)

Finding 31 established that encirclement performance collapses on the dimensionless
ratio R_enc/Rg and that the optimum is at R_enc/Rg ~ 0.5 for both N = 350 and N = 1000.
That finding was obtained with a fixed R_enc set at initialization. A natural follow-up
is whether a predator group that continuously tracks flock size and adjusts R_enc
accordingly can sustain that optimum across the long-time merge/split dynamics (Finding
32), where Rg fluctuates substantially.

I compared fixed R_enc = 0.150 against adaptive R_enc = 0.5 * live_Rg, where Rg is
recomputed from all prey positions at every timestep. Both conditions used n_pred = 10,
N = 350, slow prey, over 15000 steps (150 time units), with 5 seeds (adaptive_encirclement.py).

The adaptive strategy is more disruptive across all metrics:

| Condition | mean Phi | frac time Phi > 0.85 | mean R_enc/Rg |
|-----------|----------|----------------------|---------------|
| Fixed     |   0.778  |         0.56         |     0.485     |
| Adaptive  |   0.713  |         0.37         |     0.500     |

Mean coherence falls from 0.778 to 0.713 (an 8% reduction), and the fraction of time
the flock spends in a highly coherent state (Phi > 0.85) drops from 56% to 37% — a
34% relative reduction. The mean R_enc/Rg for the fixed condition is 0.485, only
slightly below the 0.5 target, but during the merge/split fluctuations of Finding 32
the instantaneous ratio drifts; adaptive removes those deviations. The lower temporal
standard deviation for adaptive (0.219 vs 0.233) confirms that the fluctuations into
high-Phi recovery states are suppressed.

The mechanism follows directly from Finding 32's dynamics. During consolidation phases
(sub-flocks merging, Rg shrinking), fixed R_enc becomes too large relative to the now-
smaller flock, placing predators too far from the bulk and reducing their effectiveness.
Adaptive R_enc shrinks with the flock, maintaining maximum directional pressure exactly
when the flock is most vulnerable to being re-fragmented. The combined effect cuts the
dwell time in coherent configurations by a third.

The effect size is modest, consistent with Finding 31's observation that the
performance function is relatively flat near the optimum: both conditions achieve
similar mean disruption, and the fixed condition was already close to optimal. The
primary advantage of adaptation is not unlocking a qualitatively different regime but
rather eliminating the off-optimum excursions that arise from the natural fluctuations
in a dynamical flock.

One limitation: the adaptive strategy uses global Rg (over all prey), which inflates
during fragmented phases as scattered sub-flocks span the domain. This causes adaptive
to briefly overshoot R_enc during high-fragmentation states, slightly limiting its
advantage. A predator group tracking the largest individual sub-flock's Rg would be
more effective still.

---

## 4.18 Targeted Vaccination Provides No Advantage Over Random: The Flock Contact Network Is Not Hub-Dominated (Finding 36)

Finding 30 showed that the herd-immunity threshold in the flock is approximately twice
the mean-field prediction (~0.46 vs. 0.20 at R0 = 1.25), attributed to spatial
clustering of panicked sub-groups. A natural follow-up is whether targeting
high-connectivity agents for immunity can reduce this inflated threshold. On a scale-free
network (power-law degree distribution), immunizing hub nodes first can dramatically
lower the required immune fraction because hubs are disproportionately responsible for
spreading contagion. If the flock contact network has hub nodes, targeting them should
outperform random vaccination.

I compared two vaccination protocols at the same supercritical SIS parameters as Finding
30 (beta = 2.5, gamma = 2.0, R0 = 1.25), across p_immune = 0 to 0.70 and 6 seeds
(targeted_immunity.py). In the targeted condition, each seed's settled flock was
analyzed for contact degree (number of agents within r_cont = 0.05), and the top
p_immune fraction by degree was immunized before the first panicked agent was seeded.

The result is negative: targeted vaccination provides no statistically significant
advantage over random vaccination at any tested immune fraction:

| p_immune | f_ss (random) | f_ss (targeted) | difference |
|----------|---------------|-----------------|------------|
|   0.00   | 0.593 ± 0.008 |  0.587 ± 0.006  |  −0.006    |
|   0.10   | 0.491 ± 0.010 |  0.495 ± 0.006  |  +0.004    |
|   0.20   | 0.393 ± 0.008 |  0.395 ± 0.010  |  +0.002    |
|   0.30   | 0.304 ± 0.013 |  0.301 ± 0.014  |  −0.003    |
|   0.40   | 0.204 ± 0.016 |  0.177 ± 0.080  |  −0.027    |
|   0.46   | 0.101 ± 0.072 |  0.084 ± 0.065  |  −0.016    |
|   0.50   | 0.046 ± 0.066 |  0.043 ± 0.053  |  −0.003    |
|   0.60–0.70 | 0.000      |   0.000         |   0.000    |

Differences at p_immune ≤ 0.30 are within noise (|diff| ≤ 0.006). Near the threshold
(p = 0.40–0.46), targeted shows a small numerical edge, but the variance in the targeted
condition is 4–5x higher than in the random condition (std = 0.080 vs. 0.016 at p =
0.40), and the difference is not statistically robust. Both strategies cross the quench
threshold (f_ss < 0.1) at p_immune ≈ 0.46, matching the random threshold from Finding 30.

The contact degree distribution across all seeds has mean = 9.02, median = 8, std = 6.17,
and max = 31. This is moderately heterogeneous (coefficient of variation CV = 0.68)
but the maximum degree is only 3.4 times the mean — far below the ratios of 10–1000
seen in social and internet networks where hub-targeting is effective. The second panel
of the figure (targeted_immunity_1.png) shows the degree distribution; it is approximately
unimodal and bounded, not power-law.

There is a deeper reason why degree-targeting fails even given moderate heterogeneity:
the flock contact network is spatially embedded and kinematically reconfigurable. Hub
agents are high-degree because they occupy the dense interior of the flock, where many
neighbors are in proximity. When those agents are immunized, the alignment and repulsion
forces redistribute the remaining agents, and new interior positions form. The network
topology reorganizes to restore a similar degree structure even with the specific hub
agents removed.

This result clarifies the mechanism behind Finding 30's inflated threshold. The 2x
inflation arises from spatial co-movement of panicked agents (sub-clusters that travel
together have redundant contacts with each other), not from degree heterogeneity. These
are distinct properties: degree-targeted vaccination addresses the latter but not the
former. A strategy that disperses immune agents spatially across the flock extent, rather
than concentrating them in the high-degree core, would more directly disrupt spatial
co-clustering and might outperform random vaccination — this hypothesis is tested in
Section 4.19.

---

## 4.19 Repulsion Hardness Does Not Rescue the Phase Transition: A Non-Equilibrium Diagnosis (Finding 38)

Finding 17 showed no diverging susceptibility at any tested compactness value (C = 0.10
to 0.78) using the standard soft repulsion (exponent n = 1.5). The natural follow-up
question is whether a harder repulsion potential would produce a true phase transition.
In equilibrium statistical mechanics, 2D hard discs at compactness C ~ 0.40 form a
hexagonal solid at low effective temperature and undergo the well-known KTHNY melting
transition at a finite critical temperature, so a hard-core potential should, in
principle, produce the diverging susceptibility that the soft model lacks.

I tested this by sweeping the repulsion exponent n = 1.5, 3.0, 6.0, 12.0 in a
repulsion-only simulation (no flocking, no self-propulsion) with finite-size scaling at
N = 25, 50, 100, 200, C = 0.40, 8 seeds per point, and the same noise sweep eta = 0.5
to 30 as in Finding 17 (hard_repulsion.py).

The result is unambiguous: the chi-peak (susceptibility chi = N * Var(KE/N)) falls at
eta = 30 — the top of the sweep — for every combination of exponent and system size.
The KE/N curves are essentially identical across n = 1.5 to n = 12:

| Exponent n | N = 25 chi peak | N = 50 chi peak | N = 100 chi peak | N = 200 chi peak |
|------------|-----------------|-----------------|------------------|------------------|
|     1.5    |     2607        |     7311        |      3785        |      3858        |
|     3.0    |     2609        |     7305        |      3766        |      3854        |
|     6.0    |     2615        |     7285        |      3770        |      3851        |
|    12.0    |     2615        |     7289        |      3773        |      3853        |

All peaks at eta = 30; no N-dependence of the peak location; chi values nearly
identical within each N column across all exponents.

The absence of a phase transition at n = 12 (near hard-core) requires a different
explanation from softness. The root cause is the non-equilibrium nature of the driving.
The current model uses uniform random kicks (each velocity component += eta * U[-1,1]
at every timestep), which do not satisfy the fluctuation-dissipation theorem (FDT). The
only velocity-limiting mechanism in the repulsion-only variant is confinement by
neighboring agents; there is no viscous friction (-mu*v) to produce true thermal
equilibration. Without FDT, there is no well-defined temperature, the system cannot
reach the Boltzmann distribution in configuration space, and the cooperativity required
for a phase transition cannot emerge. The crossover is instead set by the competition
between random kick energy and the scale of positional confinement, which depends on
the force range 2r0 but not on the force profile (exponent n). This explains the
complete insensitivity to n.

To observe the hard-disc phase transition in this model family, one would need to
replace the self-propulsion speed regulator with genuine Langevin dynamics: viscous
damping F_damp = -mu * v and noise amplitude sqrt(2*mu*kT/dt) * randn, which together
satisfy FDT and recover the Boltzmann distribution in the long-time limit. The KTHNY
transition would then appear at the appropriate area fraction and temperature.

This result clarifies the relationship between Findings 2, 8, 12, 17, and the present
one: the smooth crossover is a property of the entire model class (self-propulsion
regulation + independent random kicks), not of any particular parameter regime. No
modification of the repulsion potential within this model class will produce a true
phase transition.

---

## 4.20 Spatial Vaccination Also Fails: Kinematic Mixing Defeats All Targeting Strategies (Finding 37)

Section 4.18 showed that degree-targeted vaccination — immunizing the highest-contact-degree
agents first — provides no advantage over random vaccination because the flock contact network
lacks the fat-tailed heterogeneity required for hub-targeting to work, and kinematic
reorganization restores hub positions after high-degree agents are immunized. That result
identified the true mechanism behind the 2x mean-field herd-immunity inflation as SPATIAL
CLUSTERING of panicked sub-groups (agents within a panicked cluster contact primarily each
other, creating localized transmission chains). This suggests a different targeting hypothesis:
geographically distribute the immune agents to break the clustering, rather than targeting
high-degree individuals.

To test this, I compared three vaccination strategies (spatial_vaccination.py, beta = 2.5,
gamma = 2.0, R0 = 1.25, 5 seeds, same parameters as Findings 30 and 36):

- **Random**: immune agents chosen uniformly at random (baseline)
- **Spatial**: immune agents chosen by farthest-point (maxmin) sampling — each successive
  agent is the one farthest from all already-selected agents on the torus, maximizing
  spatial coverage
- **Targeted**: highest-contact-degree agents first (Finding 36 reference)

| p_immune | f_ss random | f_ss spatial | f_ss targeted |
|----------|-------------|--------------|---------------|
| 0.00 | 0.594 ± 0.009 | 0.589 ± 0.006 | 0.586 ± 0.006 |
| 0.10 | 0.490 ± 0.010 | 0.492 ± 0.010 | 0.493 ± 0.006 |
| 0.20 | 0.391 ± 0.007 | 0.391 ± 0.013 | 0.398 ± 0.009 |
| 0.30 | 0.301 ± 0.011 | 0.297 ± 0.009 | 0.298 ± 0.013 |
| 0.40 | 0.202 ± 0.016 | 0.170 ± 0.085 | 0.168 ± 0.085 |
| 0.46 | 0.088 ± 0.073 | 0.125 ± 0.063 | 0.072 ± 0.064 |
| 0.50 | 0.056 ± 0.068 | 0.088 ± 0.045 | 0.052 ± 0.055 |
| 0.60 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |

All three strategies are statistically indistinguishable at every immune fraction. At
p = 0.10-0.30, mean f_ss values agree within ±0.007, well within the ±0.009-0.013 standard
deviations. Near the threshold (p = 0.40-0.50), the spatial and targeted strategies show
substantially higher variance (std ~ 0.085) than random vaccination (std ~ 0.016-0.068),
making their slightly lower mean values unreliable. At p = 0.46, the spatial strategy
mean (0.125) is actually larger than random (0.088). This is a null result.

The mechanism is kinematic mixing. Immune agents are selected at warmup time (t = 0),
before the SIS dynamics begin at t = 1500 steps (15 time units). The flock's constant
spatial reorganization under alignment, repulsion, and self-propulsion forces scrambles
the initial agent positions between warmup and the epidemic run. By the time the epidemic
begins, the spatial arrangement of immune agents is statistically uncorrelated with the
warmup positions used for selection, making farthest-point spatial sampling equivalent
to random in its practical effect on the contact network.

A related effect explains the higher variance of spatial and targeted strategies near the
threshold. Random vaccination distributes immune agents uniformly in expectation, producing
a stable average contact-network coverage independent of the epidemic seed location.
Spatial and targeted vaccinations place agents in specific positions that may or may not
overlap with the epidemic's eventual spreading region — a source of additional variability
with no accompanying mean improvement.

Taken together, Findings 36 and 37 establish that the flock's kinematic reconfigurability
defeats all static vaccination strategies. No agent-selection rule outperforms random
vaccination in a continuously-mixing flocking collective, whether the selection criterion
is contact-degree, spatial coverage, or random chance. The 2x herd-immunity threshold
inflation is a structural property that cannot be exploited without real-time tracking and
dynamic vaccination during the epidemic.

---

## 4.21 Langevin Thermostat Confirms FDT Diagnosis but KE/N Does Not Detect Structural Melting (Finding 39)

Section 4.19 concluded that the smooth crossover is caused by non-equilibrium driving, not
by potential softness, and proposed that Langevin dynamics — viscous damping plus
FDT-satisfying thermal noise — would recover the hard-disc phase transition. This section
tests that prediction directly.

The Langevin simulation replaces the uniform random kick with viscous friction
F_damp = -mu * v (mu = 10) and thermal noise amplitude sqrt(2 * mu * kT * dt) * N(0,1)
per component per timestep. This satisfies the fluctuation-dissipation theorem: at thermal
equilibrium the Maxwell-Boltzmann distribution is stationary, velocities satisfy
equipartition (KE_x/N = kT/2 per component), and the system can in principle form an
ordered phase at low kT that melts at a finite critical temperature. Two compactness values
bracket the 2D hard-disc melting region: C = 0.60 (below the KTHNY transition) and
C = 0.70 (at or above it). Parameters: N = 25, 50, 100, 200; kT = 0.001 to 5.0 (10 values);
8 seeds; n = 1.5 repulsion; mu = 10 (langevin_repulsion.py).

The equipartition check confirms correct thermalization: KE/N at kT = 0.1 is 0.1047-0.1054
across all N and both compactnesses, within 1% of the target kT = 0.100. The Langevin
thermostat functions as designed.

However, the susceptibility chi = N * Var_seeds(time-avg KE/N) still peaks at kT = 5.0
— the top of the sweep — for every combination of N and C. Moreover, both compactnesses
give nearly identical chi values:

| N | C = 0.60 chi_peak | C = 0.70 chi_peak | kT of peak |
|---|-------------------|-------------------|------------|
| 25  | 0.0817 | 0.0817 | 5.000 |
| 50  | 0.0406 | 0.0406 | 5.000 |
| 100 | 0.0507 | 0.0507 | 5.000 |
| 200 | 0.0440 | 0.0440 | 5.000 |

The identity of C = 0.60 and C = 0.70 chi values at kT = 5.0 is explained by the ratio
of thermal to interaction energy: eps = 0.1 while kT = 5.0, so the repulsion is only 2%
of the thermal energy. Both compactnesses act as non-interacting gases at kT = 5.0, and
their chi values reflect free-gas velocity fluctuations rather than structural correlations.

The root cause is that chi = N * Var(KE/N) is the wrong diagnostic for the KTHNY
transition. KTHNY melting is a transition in positional order — the proliferation of
disclination-antidisclination pairs that destroys long-range hexatic bond-angle order
(Kosterlitz and Thouless, 1973; Halperin and Nelson, 1978). This transition shows up in the
hexatic order parameter |psi_6| = |mean_neighbors exp(6 i theta)| and the bond-angle
correlation function g6(r), not in kinetic energy fluctuations. Below KTHNY transition kT_c,
|psi_6| is large (hexagonal positional order); above it, |psi_6| collapses. With chi based
on KE/N, both the solid and fluid phases give small seed-to-seed variance in the time-averaged
KE/N (all seeds agree on KE/N ~ kT), so the metric cannot distinguish phases.

This result refines the chain of findings on the phase-transition question. Finding 17
showed no transition in the original model. Finding 38 ruled out potential softness as the
cause, identifying non-equilibrium driving as the culprit. The present finding confirms that
Langevin dynamics thermalizes the system correctly (the FDT diagnosis was right) but
demonstrates that the KE/N observable is insensitive to the structural melting transition.
To observe KTHNY, two additions are required: (1) a positional order metric such as the
hexatic order parameter, and (2) a harder repulsion to push the transition to a kT where
equilibration is practical within the simulation timescale.

---

## 4.22 Hexatic Order Parameter Detects No Crystallization: Soft Repulsion Cannot Produce KTHNY Melting (Finding 40)

Section 4.21 identified that chi = N * Var(KE/N) cannot detect the KTHNY structural melting
transition, and proposed the hexatic order parameter as the correct diagnostic. This section
tests that proposal directly with the same Langevin simulation (langevin_hexatic.py), replacing
KE/N measurements with a bond-angle analysis. At each measurement step, the bond angle
theta_{jk} is computed for every pair of neighbors within distance 3*r0, and

|psi_6,j| = |(1/k_j) * sum_{k: r_jk <= 3*r0} exp(6*i*theta_{jk})|

is averaged over all agents. The susceptibility is chi_psi6 = N * Var_seeds(time-avg |psi_6|).
For a KTHNY transition, chi_psi6 should peak at a finite kT_c that shifts with N, and the peak
magnitude should grow with N. Parameters: C = 0.60 and C = 0.70; N = 25, 50, 100, 200; kT =
0.001 to 5.0; 8 seeds.

| C | N | chi_psi6 peak | kT at peak | psi6(kT=0.001) | psi6(kT=5.0) |
|---|---|---------------|------------|----------------|--------------|
| 0.60 | 25  | 0.0115 | 0.001 | 0.416 | 0.425 |
| 0.60 | 50  | 0.0044 | 0.001 | 0.423 | 0.423 |
| 0.60 | 100 | 0.0024 | 0.005 | 0.421 | 0.422 |
| 0.60 | 200 | 0.0042 | 0.001 | 0.416 | 0.421 |
| 0.70 | 25  | 0.0044 | 0.005 | 0.372 | 0.387 |
| 0.70 | 50  | 0.0023 | 0.001 | 0.378 | 0.386 |
| 0.70 | 100 | 0.0010 | 0.005 | 0.377 | 0.386 |
| 0.70 | 200 | 0.0027 | 0.001 | 0.376 | 0.385 |

This is a null result with three diagnostic signatures:

**(1) |psi_6| is flat across all temperatures.** For C = 0.60, |psi_6| ranges from 0.416 to
0.425 across the entire kT sweep — a 2% variation with no systematic trend. For C = 0.70 the
range is 0.372–0.387 (4%). A KTHNY transition would show |psi_6| near 1.0 in the solid phase
and collapsing toward 0 in the fluid phase, with a sharp crossover at kT_c. Here |psi_6| is
constant at ~0.4 at both the coldest (kT = 0.001) and hottest (kT = 5.0) tested temperatures.
No solid phase exists at low kT; no fluid phase exists at high kT in the KTHNY sense.

**(2) chi_psi6 peaks at the bottom of the sweep.** All chi_psi6 peaks occur at kT = 0.001 or
kT = 0.005 — the lowest tested temperatures. The susceptibility is a monotonically decreasing
function of kT with no interior maximum. There is no temperature at which the system is near
a structural phase boundary.

**(3) chi_psi6 does not grow with N.** For C = 0.60, chi_psi6_peak falls from 0.0115 at
N = 25 to 0.0024 at N = 100 and 0.0042 at N = 200 — no systematic increase. A diverging
susceptibility would scale as N^(gamma/nu) in finite-size scaling theory. The observed trend
is the opposite.

The failure lies not in the metric but in the potential. For a hexagonal crystal to form,
agents need six well-defined nearest neighbors at a fixed lattice spacing with no overlap.
The n = 1.5 repulsion force F ~ (1 - r/2r0)^{1.5} is a smooth contact-avoidance: it
approaches zero at d = 2r0 and allows agents to overlap substantially before exerting
significant force. At any finite kT, agents can pass through each other's soft cores, so
no rigid hexagonal lattice can lock in. The flat |psi_6| ≈ 0.4 at all temperatures reflects
fixed short-range geometric correlation from the initial repulsive packing — not a
thermally-ordered crystalline phase.

A notable secondary result is that C = 0.70 produces lower |psi_6| than C = 0.60 at every
temperature (0.38 vs. 0.42). In equilibrium hard-disc systems, higher compactness drives more
hexagonal ordering until the KTHNY melting point. With soft repulsion, the opposite occurs:
at C = 0.70, agents are closely packed enough to exert overlapping repulsive forces in many
simultaneous directions, creating a frustrated amorphous arrangement that is less hexagonally
ordered than C = 0.60. The soft potential frustrates crystallization precisely where hard
discs would crystallize most strongly.

This finding completes the four-step phase-transition thread. Finding 17 showed no
susceptibility peak at any compactness in the original model. Finding 38 ruled out soft
repulsion as the cause, identifying non-equilibrium driving as the culprit. Finding 39
confirmed that Langevin dynamics thermalize correctly but that KE/N is the wrong
observable. The present finding shows that even with the correct observable (|psi_6|),
the n = 1.5 soft repulsion cannot produce a hexagonal solid at any accessible temperature.
The correct metric has been identified; the missing ingredient is a harder repulsion
potential (n ≥ 12 in a Langevin framework, or a true hard-disc Monte Carlo simulation)
that would prevent agent overlap and enable genuine crystallization.

---

## 4.23 Three-Dimensional Extension: Flocking Generalizes and v_eq Is Dimensionality-Independent (Finding 41)

All preceding sections studied agents on a 2D periodic unit square. This section extends
the model to a 3D periodic unit cube [0, 1]^3 and tests whether the core behaviors — flock
formation, the analytical equilibrium-speed result, and the noise-coherence tradeoff — hold
in three dimensions.

Parameters were scaled to maintain a similar neighborhood density as the 2D default. The
2D default (r_f = 0.10, N = 350) gives an expected neighbor count of N * pi * r_f^2 = 11.0
agents. In 3D, r_f = 0.20 gives N * (4/3) * pi * r_f^3 = 11.7, matching the 2D count.
The repulsion radius was set to r0 = 0.02, giving a volume fraction of approximately 0.012,
comparable to the 2D area fraction of 0.028. All other parameters are unchanged: alpha = 1.0,
v0 = 1.0, mu = 10.0, dt = 0.01 (flocking3d.py, N = 350, 8 seeds).

The results show that flocking forms cleanly in 3D and that the analytical result transfers
exactly:

| ramp | Phi (3D) | Mean speed |
|------|----------|------------|
| 0.0  | 1.0000   | 1.1000     |
| 0.5  | 0.9995   | 1.1000     |
| 1.0  | 0.9982   | 1.1001     |
| 2.0  | 0.9931   | 1.1006     |
| 5.0  | 0.9595   | 1.1035     |
| 7.0  | 0.9220   | 1.1068     |
| 10.0 | 0.8409   | 1.1138     |

**Equilibrium speed.** At ramp = 0, the measured mean speed is 1.1000, exactly matching the
prediction v_eq = v0 + alpha/mu = 1.0 + 1.0/10.0 = 1.100. The analytical derivation uses
only the force balance along the heading direction of an aligned flock:
alpha + mu * (v0 - v_eq) = 0, giving v_eq = v0 + alpha/mu.
This argument is dimensionality-independent, so the result holds in 3D for the same reason
it holds in 2D. The 3D measurement confirms this: the formula is a universal property of the
model's force structure, not an artifact of the 2D geometry.

**Noise-coherence tradeoff in 3D.** Phi degrades monotonically from 1.000 at ramp = 0 to
0.841 at ramp = 10. The seed-to-seed standard deviation is very small (std = 0.0022 at
ramp = 10), confirming consistent behavior across initializations. The 3D flock is somewhat
less noise-resistant than its 2D counterpart: at ramp = 10, 2D Phi is approximately 0.97-0.99
(from the original phase sweeps), while 3D Phi = 0.84. This difference is expected from the
noise geometry: in 3D, random kicks affect all three velocity components, so the total
perturbation magnitude scales as ramp * sqrt(3) per step, compared to ramp * sqrt(2) in 2D
(each component uniform on [-ramp, ramp] with variance ramp^2/3). With more degrees of
freedom available for noise to decorrelate agent headings, the alignment force maintains less
complete coherence at the same ramp value.

The crossover from high-Phi to low-Phi in 3D appears to lie at ramp >> 10 (the order
parameter remains at 0.84 even at ramp = 10, far from the plateau-to-collapse region). An
extended noise sweep to ramp ~ 20-30 is the natural next step to characterize the full
transition region.

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
means non-interacting walkers. Section 4.19 goes further by sweeping the repulsion
exponent from n = 1.5 to n = 12 — approaching the hard-core limit — and finds identical
behavior at every exponent. This rules out potential softness as the cause of the
crossover. The root cause is instead the non-equilibrium driving: uniform random kicks
without viscous dissipation do not satisfy the fluctuation-dissipation theorem, so the
system cannot equilibrate into a crystal phase. The smooth crossover is a universal
property of the model class, not a potential-specific artifact, and cannot be removed
without replacing the driving mechanism itself.

The long-time encirclement result is a caution against interpreting short-simulation
steady states as equilibria. The 4000-step snapshots used in Sections 4.5-4.7 reported
a "steady" disrupted flock at Phi ~ 0.72. The 30000-step runs in Section 4.15 reveal
that this value is a time-average hiding large oscillations (temporal std = 0.21 per
seed). The long-time attractor of the encircled flock is not a fixed fragmented state
but a persistent merge/split cycle. This has methodological implications: order-parameter
measurements from short runs near an encirclement configuration may give misleading
precision.

The encirclement gap experiment (Section 4.15) challenges intuitions about escape.
Removing one predator from a symmetric ring is MORE damaging to flock coherence than
the full ring, because the asymmetry creates a perpetually shifting CoM chase rather
than a balanced stable pattern. This is counterintuitive from a naive perspective where
gaps should always help. The result reflects a general feature of these agent models:
agents have no global spatial awareness. They respond only to local forces, so a gap
in the predator ring is invisible to agents far from the gap.

The outbreak persistence result (Section 4.16) highlights an asymmetry in recovery
timescales. Kinematic damage from encirclement reverses rapidly because it has no memory
beyond the agents' current positions and velocities; once the force pattern changes, the
trajectory changes. Epidemic damage has memory in the agent's internal state (panicked vs.
calm) and reverses on epidemic timescales set by the ratio of infection and recovery rates,
which are independent of kinematic parameters. This is a qualitative distinction with
potential biological relevance: a predator event that coincides with a near-threshold
social contagion (collective alarm behavior, epidemic disease) leaves a lasting mark on
the collective that outlives the predation event itself.

The adaptive encirclement result (Section 4.17) provides a constructive validation of
the R_enc/Rg scaling (Finding 31): if the universal optimum at R_enc/Rg ~ 0.5 is real,
a predator that tracks live Rg should maintain that optimum even as the flock fluctuates
during the merge/split cycle, and it does. The 34% reduction in high-coherence dwell time
(frac_above_0.85 from 0.56 to 0.37) is the direct signature of this — adaptive predators
suppress the recovery excursions that fixed predators permit. The modest mean-Phi effect
(8%) confirms that the fixed strategy was already close to optimal, with the fixed
R_enc/Rg sitting at 0.485 on average. Adaptation gains most at the margin, during the
brief consolidation phases where the flock is most recoverable.

The predator-strategy findings place this work in a broader literature of simulated predation
on flocking models. Demsar and Lebar Bajec (2014) compared attack tactics (target the center,
target the nearest, target isolated individuals) in a fuzzy individual-based model and found
that social flocking protects against predators targeting isolated prey. Importantly, they do
not test a coordinated multi-angle encirclement strategy, which is the key contribution of
Sections 4.5-4.7 here. Inada and Kawachi (2002) identified four escape-pattern categories
in a two-dimensional fish-school model, including "Split and Reunion" — the pattern that
Sections 4.6 and 4.7 quantify and mechanistically explain: encirclement causes the split, and
the reunion timescale is ~10 time units after predator removal. Bartashevich et al. (2024)
studied the "fountain effect" in sardines evading marlin using both agent-based models and
empirical observations, finding that prey optimize individual escape angles relative to the
predator's attack direction. The fountain effect is a single-predator pattern; our encirclement
result extends it to the multi-predator case where angular pressure from all sides prevents
any single escape direction from being optimal, leading to directional fragmentation rather
than a coherent fountain.

The most closely related existing work is Levis et al. (2020), who studied bidirectional
coupling between Vicsek-like flocking and SIS epidemic dynamics, finding that endogenous
clustering (infected agents altering their motion rules) reduces the epidemic threshold
in a similar way to the encirclement-induced compression found here. The critical
difference is that Levis et al.'s mechanism is internal: the epidemic state controls
flocking, so compression and contagion are inseparable. In the present model, the
compression is imposed externally by predators, making it possible to study the
removal experiment (Section 4.16) — what happens when the compressor is switched off
while the epidemic is still ongoing. That timescale asymmetry, and the practical
implication that external pressure can seed a long-lived epidemic state without
sustaining it, does not appear to be addressed in the existing literature.

The vaccination results (Sections 4.18 and 4.20) establish a broader principle beyond
the specific null results. In heterogeneous networks (Barabasi-Albert scale-free,
small-world), targeted vaccination of high-degree nodes reduces the epidemic threshold
substantially (Pastor-Satorras and Vespignani, 2002; Cohen et al., 2003). Spatial
vaccination strategies for geographic human populations have also been studied (Bhatt
et al., 2022; Zhou et al., 2021), showing advantages when infection clusters spatially.
The flock fails to benefit from either approach: degree-targeting is defeated by bounded
degree heterogeneity (CV = 0.68) and kinematic reorganization that restores hub positions;
spatial targeting is defeated by kinematic mixing that scrambles spatial distributions
before the epidemic runs. Both null results trace to the same property of the flock —
the constant spatial reorganization driven by the alignment force — which prevents any
static structural feature of the contact network from remaining stable long enough to be
exploited. For kinematic collectives, the epidemiological literature's targeting strategies
are inapplicable in their standard form: only a dynamic vaccination strategy applied
in real time during the epidemic, tracking current agent positions, could exploit the
spatial-clustering mechanism that inflates the threshold.

---

## 6. Conclusions

This study produced seventeen main results (selecting the most general across 41 findings):

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

7. **Encirclement is size-invariant when scaled to flock geometry:** The encirclement
   disruption floor depends on R_enc/Rg, not on absolute R_enc or absolute N. The
   optimal disruption at R_enc/Rg ~ 0.5 is universal across tested flock sizes. The
   apparent N-dependence reported in earlier experiments was an R_enc/Rg mismatch.

8. **Long-time encirclement is intermittent:** Over 300 time units, the encircled flock
   does not settle into a steady fragmented state. Phi oscillates between ~0.4 and ~0.95
   as sub-flocks repeatedly merge and re-split, driven by a self-reinforcing cycle
   between coherence and multi-directional predator pressure.

9. **Internal stressors require contagion to matter:** Statically labeled panicked
   agents do not propagate disruption — the alignment force keeps calm neighbors
   coherent even at 20% panic fraction. Once a contact-mediated SIS contagion mechanism
   is added, the epidemic threshold appears at beta/gamma ~ 1. Below threshold the flock
   contains the outbreak; above it an endemic panicked state suppresses coherence
   proportionally to beta/gamma. The flock is not a well-mixed population at the
   contagion length scale: the spatial herd-immunity threshold (~0.46) is more than
   twice the mean-field prediction (~0.20), driven by clustering of panicked sub-groups.

10. **Encirclement shifts the epidemic threshold by ~4%:** Compressing the flock raises
    local contact count ~3x and lowers the effective epidemic threshold from beta_c =
    1.93 to 1.85. The shift is real but modest: compression creates redundant contacts
    within already-panicked sub-clusters rather than fresh ones. Encirclement is a
    near-critical amplifier, not a general one.

11. **Epidemic damage outlasts kinematic damage by an order of magnitude:** When
    encirclement drives a sub-threshold contagion into a transient endemic state, removing
    the predators reverses the kinematic fragmentation (~10 time units, as in pure
    encirclement) but leaves the SIS epidemic intact. After 50 time units without predators,
    panic fraction remains at 0.41 (barely below the encirclement-elevated peak of 0.45)
    and Phi is only 0.27. The residual epidemic suppresses alignment for hundreds of time
    units. Kinematic damage is reversible; epidemic damage outlasts the event that caused it.

12. **Adaptive encirclement outperforms fixed radius:** Predators that continuously track
    live flock Rg and set R_enc = 0.5 * Rg maintain the universal optimum across the
    merge/split cycle (Finding 32), reducing mean coherence from Phi = 0.778 to 0.713 and
    cutting the fraction of time spent in a highly coherent state (Phi > 0.85) from 56% to
    37% compared to a fixed-radius strategy. The effect size is modest (8% in mean Phi)
    because the fixed radius was already near-optimal at initialization, but the adaptive
    strategy eliminates the off-optimum excursions that arise from flock geometry
    fluctuations during intermittent dynamics.

13. **All vaccination targeting strategies fail in a kinematic flock:** Neither
    degree-targeted vaccination (immunizing highest-contact-degree agents first, Finding
    36) nor spatially-targeted vaccination (farthest-point maxmin sampling to maximize
    spatial coverage, Finding 37) outperforms random vaccination. Both strategies require
    p_immune ~0.46, identical to random. The mechanism is the flock's kinematic
    reconfigurability: kinematic reorganization restores hub positions after high-degree
    agents are immunized (defeating degree-targeting), and kinematic mixing during the
    warmup phase scrambles the spatial distribution of immune agents before the epidemic
    begins (defeating spatial targeting). No static agent-selection rule can maintain its
    structural advantage in a continuously-mixing flocking collective; the 2x herd-immunity
    threshold inflation is a systemic property that cannot be efficiently exploited.

14. **The smooth crossover is a consequence of non-equilibrium driving, not repulsion
    softness:** Sweeping the repulsion exponent from n = 1.5 (current) to n = 12 (near
    hard-core) produces virtually identical finite-size scaling behavior in all cases —
    chi_peak at the top of the noise sweep, N-independent KE/N curves, no diverging
    susceptibility. In equilibrium statistical mechanics, 2D hard discs at the tested
    compactness (C = 0.40) do exhibit a phase transition (KTHNY melting). The absence of
    any transition at n = 12 demonstrates that the model's non-thermal driving — uniform
    random kicks without viscous dissipation — prevents the Boltzmann equilibration
    required for cooperative melting.

15. **Langevin dynamics thermalize correctly but KE/N cannot detect structural melting:**
    A Langevin thermostat (viscous damping + FDT-satisfying thermal noise) recovers
    equilibrium thermalization — KE/N = kT to within 1% (equipartition) — confirming the
    non-equilibrium diagnosis of Finding 38. However, the susceptibility chi = N*Var(KE/N)
    still peaks at the top of the kT sweep and shows no N-dependent shift, because KTHNY
    melting is a positional-order transition invisible to kinetic energy fluctuations.

17. **Three-dimensional extension confirms universality of v_eq = v0 + alpha/mu:**
    Extending the model to a periodic 3D unit cube with neighbor-count-matched parameters
    (r_f = 0.20 for ~12 expected neighbors) reproduces coherent flocking across the tested
    noise range. The equilibrium speed v_eq = v0 + alpha/mu = 1.100 holds exactly (measured:
    1.1000 at ramp = 0), confirming the analytical result is dimensionality-independent —
    a consequence of the 1D force balance along the heading direction. The 3D flock is
    slightly less noise-resistant than 2D at the same ramp (Phi = 0.84 vs ~0.98 at ramp = 10)
    because 3D velocity perturbations have one additional degree of freedom. Flocking forms
    cleanly in 3D and the core analytical result transfers exactly.

16. **Hexatic order parameter confirms soft repulsion cannot crystallize:**
    Measuring the hexatic order parameter |psi_6| directly (the correct diagnostic for KTHNY
    melting) reveals that n = 1.5 soft repulsion is incapable of forming a hexagonal solid at
    any accessible temperature. |psi_6| ≈ 0.4 across the entire kT range (0.001 to 5.0) for
    both compactness values (C = 0.60 and C = 0.70), with no solid-phase value near 1.0 and no
    fluid-phase collapse to 0. chi_psi6 peaks at the bottom of the kT sweep with no N-dependent
    growth. The mechanism is that the smooth n = 1.5 contact-avoidance potential allows agents to
    overlap at any finite kT, preventing the rigid hexagonal lattice required for KTHNY melting.
    Demonstrating the KTHNY transition in this model family requires a near-hard-core Langevin
    simulation (n ≥ 12) or a true hard-disc Monte Carlo framework. This closes the phase-
    transition thread: the correct observable has been identified; a harder potential is needed.

The consistent thread across all results is that collective alignment is both the source
of the flock's robustness and the mechanism by which stressors interact. It maintains
coherence under noise and naive predation; it transmits spatial clustering that amplifies
contagion; it drives the spatial reorganization that defeats all vaccination targeting
strategies; and it enables the reunion that makes kinematic damage reversible. The most
effective disruption strategies are those that operate at the flock's geometric scale
(encirclement at R_enc/Rg ~ 0.5) or that exploit a timescale the alignment force cannot
overcome (epidemic persistence after predator removal).

---

## References

Charbonneau, P. (2017). *Natural Complexity: A Modeling Handbook*. Princeton University Press.

Silverberg, J. L., Bierbaum, M., Sethna, J. P., and Cohen, I. (2013). Collective motion
of humans in mosh and circle pits at heavy metal concerts. *Physical Review Letters*,
110, 228701.

Levis, D., Diaz-Guilera, A., Pagonabarraga, I., and Starnini, M. (2020). Flocking-enhanced
social contagion. *Physical Review Research*, 2, 032056.

Pacher, K., Bierbach, D., Kurvers, R. H. J. M., and Krause, J. (2026). Strategic choices
of attack location allow predators to counter a collective prey defence. *Proceedings of
the Royal Society B*, 293, 20260566.

Inada, Y. and Kawachi, K. (2002). Order and flexibility in the motion of fish schools.
*Journal of Theoretical Biology*, 214, 371-387. (Split and Reunion escape patterns in
a two-dimensional fish-school model.)

Demsar, J. and Lebar Bajec, I. (2014). Simulated predator attacks on flocks: A comparison
of tactics. *Artificial Life*, 20(3), 343-359.

Bartashevich, P., Schellinck, J., Tully, T., and Romanczuk, P. (2024). Collective
anti-predator escape manoeuvres through optimal attack and avoidance strategies.
*Communications Biology*, 7, 1548.

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
| hybrid_sis.py | Sub-threshold SIS + encirclement (compression-amplification) |
| segregation.py | Active/passive segregation (mixed v0 populations) |
| segregation_alpha.py | Alpha-contrast segregation + local-purity diagnostic |
| large_N_encirclement.py | Encirclement at N=350, 700, 1000 |
| critical_shift.py | Beta sweep with/without encirclement; threshold shift measurement |
| herd_immunity.py | Immune sub-population sweep at supercritical SIS |
| renc_scaling.py | R_enc sweep at N=350 and N=1000; collapse on R_enc/Rg |
| long_encirclement.py | Long-time encirclement (30000 steps); merge/split dynamics |
| encirclement_gap.py | Incomplete encirclement; gap detection test |
| outbreak_removal.py | Encirclement+SIS then predator removal; epidemic persistence |
| adaptive_encirclement.py | Adaptive R_enc=0.5*live_Rg vs fixed R_enc; disruption comparison |
| targeted_immunity.py | Targeted (high-degree first) vs random vaccination; herd-immunity efficiency |
| spatial_vaccination.py | Spatial (farthest-point sampling) vs random vs degree-targeted vaccination |
| hard_repulsion.py | Finite-size scaling with harder repulsion exponents n=1.5,3,6,12 |
| langevin_repulsion.py | Langevin thermostat finite-size scaling; FDT diagnosis of crossover |
| langevin_hexatic.py | Langevin dynamics with hexatic order parameter; KTHNY structural melting test |
| flocking3d.py | 3D extension of the flocking model; v_eq validation and noise sweep |
| model.py | OOP foundation: Flock and Predator classes for new experiments |
