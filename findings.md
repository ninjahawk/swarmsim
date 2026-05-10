# Findings — PHY 351 Flocking Research
Nathan Langley | Started 2026-05-08

---

## Model Summary
Charbonneau Chapter 10. N agents on a periodic 2D unit square [0,1]^2.
Each agent subject to 4 forces per timestep:
- Repulsion: pushes agents apart within range 2*r0
- Flocking: aligns velocity with neighbors within range rf
- Self-propulsion: drives agent toward target speed v0
- Random: uniform noise in [-ramp, ramp]

Forward Euler integration at dt=0.01. Periodic boundaries via ghost agent buffer zone.
Default parameters: N=350, r0=0.005, eps=0.1, rf=0.1, alpha=1.0, v0=1.0, mu=10.0, ramp=0.5

Order parameter: Phi = |mean(v_hat)|, ranges 0 (random) to 1 (perfect alignment)

---

## Finding 1: Equilibrium speed is v_eq = v0 + alpha/mu
**What:** Agents don't cruise at v0 -- they cruise at v_eq = v0 + alpha/mu.
**Why:** When the flock aligns, the flocking force always pushes agents forward
at magnitude alpha. Self-propulsion equilibrates at a higher speed to compensate.
**Evidence:** With alpha=1, mu=10, v0=1 -> measured mean speed 1.098 vs predicted 1.100 (diff=0.002).
Verified across alpha = 0, 0.5, 1.0, 2.0 -- all match v0 + alpha/mu to within 0.002.
**Implication:** v0 and alpha cannot be set independently without affecting cruise speed.
To target a specific cruise speed v_cruise, set v0 = v_cruise - alpha/mu.

---

## Finding 2: Solid-to-fluid phase transition in repulsion-only system
**What:** With only repulsion and noise (alpha=0, v0=0), KE rises continuously with noise
amplitude eta. Transition appears continuous (not abrupt) across eta ~ 1-10.
**Evidence:** Sweep A in analysis.py, 5 seeds each with error bars.
eta=0.5: KE=0.00 | eta=3: KE=0.16 | eta=10: KE=1.74 | eta=30: KE=15.89
**Visual:** At low eta, agents form hexagonal crystal (close-packed). At high eta, fluid phase.
**Open question:** Is this a true phase transition or a crossover? Would need to look for
diverging susceptibility or correlation length to answer.

---

## Finding 3: Low threshold for flock formation
**What:** Flock forms (Phi > 0.5) at very small flocking amplitude alpha -- around alpha ~ 0.05-0.1.
**Evidence:** Sweep B in analysis.py, 5 seeds each.
alpha=0: Phi=0.088 | alpha=0.05: Phi=0.404 | alpha=0.10: Phi=0.607 | alpha=0.20: Phi=0.891
**Note:** Large variance at alpha=0.05-0.10 (std ~0.12-0.19) -- near the transition the outcome
is sensitive to initial conditions. Above alpha~0.2 the flock reliably forms.

---

## Finding 4: Full model robust to noise up to eta~10
**What:** With all forces active (default parameters), Phi stays near 1.0 up to eta~10,
then drops sharply, collapsing to Phi~0.4 at eta=20.
**Evidence:** Sweep C in analysis.py, 5 seeds each.
eta=1: Phi=0.999 | eta=5: Phi=0.976 | eta=10: Phi=0.906 | eta=20: Phi=0.417
**Contrast:** Repulsion-only system (Finding 2) doesn't form a flock at all.
The flocking force makes the system much more resistant to noise disruption.

---

## Finding 5: Flocking maintains coherence under predator pressure
**What:** Under predator pressure, flocking prey maintain Phi~1.0 throughout the simulation.
Non-flocking prey (alpha=0) scatter to Phi~0.1 almost immediately.
**Evidence:** Exp 2 in predator.py, 10 seeds each.
- Flocking: steady-state Phi = 0.998, mean predator-nearest distance = 0.112
- Non-flocking: steady-state Phi = 0.096, mean distance = 0.127
**Interpretation:** The flock absorbs the predator's pressure without breaking apart.
Non-flocking agents are marginally farther from the predator individually, but they have
no collective structure. Flocking sacrifices some individual distance for group cohesion.

---

## Finding 6: Flock coherence robust to predator aggression
**What:** Prey flock Phi stays near 0.95-0.98 across all tested predator aggression levels
(alpha_pred = 0 to 15, effective speeds 0.05 to 1.55). Flock does not break apart.
**Evidence:** Exp 3 in predator.py, 8 seeds each.
**Evidence:** Predator-flock distance drops from 0.24 (passive predator) to ~0.10
(alpha_pred >= 1) and then saturates -- the flock successfully holds the predator
at a minimum distance regardless of how fast the predator is.
**Interpretation:** There is an evasion floor. The collective repulsion response is enough
to maintain a fixed buffer distance even against a very aggressive predator.

---

## Finding 7: Dilution effect -- larger flocks expose smaller fractions
**What:** The fraction of the flock within predator threat range decreases with flock size.
**Evidence:** Exp 4 in predator.py, 5 seeds each.
N=10: fraction~0.49 | N=25: ~0.21 | N=50: ~0.19 | N=100: ~0.11 | N=200: ~0.14
**Interpretation:** Basic geometric dilution -- predator occupies a fixed area, larger
flock has more agents outside that area. Consistent with the biological "safety in numbers"
hypothesis. Note: N=200 slightly worse than N=100 (0.14 vs 0.11), possibly noise.

---

---

## Finding 8: Solid-to-fluid transition is a crossover, not a phase transition
**What:** Finite-size scaling across N=25,50,100,200 shows KE/N is essentially
independent of N. Susceptibility chi = N*var(KE/N) rises monotonically, no peak.
**Evidence:** phase_transition.py. All four N values give nearly identical KE/N curves.
**Interpretation:** High compactness (C~0.78) traps agents -- each agent oscillates
like an independent harmonic oscillator at its lattice site. A true phase transition
would require chi to diverge at finite eta and scale as N^(gamma/nu). Neither observed.

---

## Finding 9: Flock elongates with predator pressure and with stronger alpha
**What:** Aspect ratio AR increases with predator presence and flocking amplitude alpha.
**Evidence:** geometry.py
- No predator: AR=2.61 | With predator: AR=2.76
- alpha=0.2: AR=2.09 | alpha=1.0: AR=2.94 | alpha=2.0: AR=7.27
**Interpretation:** Strong velocity alignment forces agents into a tight stream.
Under predator pressure the flock thins and elongates, consistent with the book's
prediction of arched/thinning flocks.

---

## Finding 10: Multiple predators elongate the flock without breaking coherence
**What:** With 1-4 predators, Phi stays near 0.975-0.991 throughout. AR increases
from 2.83 (1 predator) to 7.91 (3 predators). Min predator-prey distance actually
increases slightly from 0.093 to 0.106.
**Evidence:** multi_predator.py, 8 seeds each.
**Interpretation:** More predators stretch and thin the flock but don't destroy it.
Counterintuitively, evasion distance increases with more predators -- pressure from
multiple directions may force the flock into a harder-to-surround configuration.

---

## Finding 11: Evasion distance increases because predators co-localize at prey CoM
**What:** With multiple predators, each one independently targets the flock center of mass.
Because all predators follow the same rule, they converge to the same location
(measured pred-pred distance ~0.001, essentially zero). This means more predators
pile up at the same point rather than surrounding the flock.
**Evidence:** evasion_analysis.py, 8 seeds each.
- n_pred=1: min_pred_prey_dist=0.094
- n_pred=2: pred-pred dist=0.001, min_pred_prey_dist=0.099
- n_pred=3: pred-pred dist=0.001, min_pred_prey_dist=0.105
- n_pred=4: pred-pred dist=0.001, min_pred_prey_dist=0.096
- Flock orientation vs predator centroid: ~43-46 deg (consistent with random, ~45 deg expected)
**Mechanism:** Co-localized predators exert combined repulsion from the same point.
Multiple predators at the same location produce a stronger net repulsion on nearby prey
than a single predator, pushing the flock farther away. The flock does NOT orient
deliberately (45-degree angle = no systematic orientation strategy).
**Implication:** The "chase CoM" predator strategy is self-undermining when used by
multiple predators. They inadvertently cooperate by concentrating force at one point,
and the flock benefits. Distributed predators approaching from different angles would be
more effective -- but that requires coordination the model does not give them.

---

## Finding 12: Crossover persists across compactness -- no phase transition in this model
**What:** When compactness is held fixed across N by scaling r0 = sqrt(C/(pi*N)),
both dense (C=0.78) and dilute (C=0.10) regimes give essentially identical results:
KE/N curves overlap for all N, susceptibility chi increases monotonically with eta,
and the susceptibility peaks only at the top of the sweep (eta=30) -- not at a finite
critical point. The crossover is not a property of the dense regime alone; it persists
at all tested compactness values.
**Evidence:** compactness_phase.py, 8 seeds per (N, eta) point.
- C=0.78: chi_max at eta=30 for all N; KE/N identical across N=25-200
- C=0.10: same behavior -- chi_max at eta=30, KE/N N-independent
- KE/N values nearly identical between C=0.78 and C=0.10
**Interpretation:** The absence of a critical point is not because agents are caged.
At C=0.10 (dilute), agents barely interact (repulsion radius too small relative to
inter-agent spacing), so they behave essentially as independent random walkers. KE/N
is then set entirely by the noise amplitude, independent of N. Both extremes (too
dense = caged, too dilute = non-interacting) produce N-independent KE/N and monotone
susceptibility. A genuine critical point would likely require intermediate compactness
where agents can form a solid and also rearrange cooperatively -- but even C=0.10 is
too dilute for a solid phase. The repulsion-only model may simply not exhibit a true
phase transition in any easily accessible parameter regime.

---

## Open Questions / Next Directions
1. What happens with coordinated predators that flock toward each other to herd prey?
2. Does the low-compactness phase transition show proper scaling collapse?
3. What is the critical exponent for the low-compactness transition?
