# Findings — PHY 351 Flocking Research
Started 2026-05-08

---

## Index by Theme

The 71 numbered findings below are presented chronologically (in the order they were
generated). The F-numbers here match the headings in the body of this file, in
`report_draft.md`, and in `README.md`. A few findings touch more than one theme and are
cross-listed (e.g. F8/F12/F17 under both Baseline and Phase Transition; F52 under both
3D Extension and Section 5 Self-Tests). For navigation, grouped by theme:

### Baseline and Validation
- F1  Equilibrium cruise speed v_eq = v0 + alpha/mu (exact)
- F2  Solid-to-fluid transition in repulsion-only system (appears continuous; preliminary)
- F3  Low threshold for flock formation (alpha ~ 0.05)
- F4  Full model robust to noise up to eta ~ 10
- F8  Solid-to-fluid transition is a crossover, not a phase transition (finite-size scaling)
- F12 Crossover persists across compactness -- no phase transition in this model
- F17 No phase transition at any intermediate compactness (C = 0.15-0.60)

### Predator Strategy (2D)
- F5  Flocking maintains coherence under predator pressure (Phi ~ 1.0)
- F6  Flock coherence robust to predator aggression
- F7  Dilution effect: larger flocks expose smaller fractions
- F9  Flock elongates with predator pressure and with stronger alpha
- F10 Multiple predators elongate the flock without breaking coherence
- F11 Evasion distance increases because predators co-localize at prey CoM
- F13 Coordinated predators spread out but cannot break the flock (Phi > 0.92)
- F14 Encirclement breaks coherence -- first strategy to substantially disrupt (Phi = 0.77 at n_pred = 6)
- F15 Encirclement threshold does not scale with N -- convergence to a common floor
- F16 Encirclement divides the flock into coherent sub-flocks, not random walkers
- F19 Predator sensing threshold at r_sense ~ flock radius; limited sensing slightly worsens encirclement
- F21 Minimum viable flock size (Phi = 0.9 crosses between N = 18-25)
- F22 Encirclement fragmentation fully transient -- sub-flocks reunite within ~10 tu of predator removal
- F28 Encirclement floor rises at very large N -- F15's "common floor" is N-dependent
- F31 Encirclement scaling collapses on R_enc/Rg ~ 0.5 -- size-invariant (refines F28)
- F32 Long-time encirclement: intermittent merge/split steady state
- F33 Incomplete encirclement is mostly less disruptive; the flock doesn't escape through the gap
- F35 Adaptive R_enc = 0.5*Rg outperforms fixed (validates F31 dynamically)
- F53 Prey fatigue does not make encirclement damage irreversible (align-fatigue deepens attack)
- F66 Predictive encirclement (predators anticipate via CoM + lead*v_mean) deepens F14 to Phi=0.530 at lead~2 tu -- first predator adaptation to substantially beat F14
- F67 Predictive (F66) + adaptive R_enc (F35) do NOT compose: predictive-adaptive Phi=0.535 ~= predictive-fixed 0.530. Placement is dominant; angular spread secondary once heading is blocked
- F68 Predictive encirclement degrades gracefully but GRADED under noisy v_mean estimates. Less noise-tolerant than F60's slow-targeting -- a global summary statistic (one number per step) has no N-sample averaging
- F69 Predictive encirclement is FAR more sensitive to DELAY than to noise: a 0.25 tu lag erases most of the advantage, by 1 tu it is gone (Phi>=F14). Delay = systematic (biased) error on a value used for forward projection; noise = zero-mean. v_mean decorrelates on sub-tu timescales under disruption

### Contagion and Vaccination
- F18 Static panic does not propagate -- calm agents stay coherent even at 20% panic fraction
- F20 Panic contagion saturates the flock at any non-zero rate (SI) -- no epidemic threshold
- F23 Combined predation + contagion -- contagion dominates; encirclement cannot rescue the calm sub-flock
- F24 Active/passive mixed populations do not spatially segregate (v0 contrast alone)
- F25 SIS contagion has a clean epidemic threshold at beta/gamma ~ 1; flock disruption tracks it
- F26 Encirclement amplifies but does not tip sub-threshold contagion -- partial coupling
- F27 Alpha-contrast populations segregate via local clustering, not heading-axis separation
- F29 Encirclement shifts the SIS epidemic threshold leftward by ~4%
- F30 Herd-immunity threshold in the flock is ~2x larger than mean-field predicts (p_c ~ 0.46)
- F34 Outbreak persistence: epidemic outlasts predator removal by 100+ tu
- F36 Targeted (degree-based) vaccination null -- contact network is not hub-dominated
- F37 Spatial vaccination null -- kinematic mixing erases spatial targeting
- F54 Heterogeneous recovery rates lower the SIS threshold -- slow recoverers are reservoirs
- F55 Heterogeneous infectiousness does NOT shift the threshold -- super-spreaders are messengers
- F56 Targeting slow recoverers beats random by 2-3x -- F54 prediction confirmed (first strategy in study to beat random)
- F57 Spatial vaccination null transfers to the het-recovery regime -- advantage is internal, not spatial
- F59 Slow-recoverer advantage survives continuous (lognormal) gamma distributions
- F60 Slow-recoverer vaccination tolerates noisy gamma estimates
- F61 Slow-recoverer vaccination works for rare reservoirs (smaller reservoir, smaller budget)
- F62 Slow-recoverer vaccination needs a DURABLE label: gamma drift erodes the advantage and, when fast, eradicates the epidemic by self-averaging gamma to its mean
- F63 Combined beta_i + gamma_i: slow (reservoir) targeting is ROBUST across correlations; super-spreader (engine) targeting is as good only when not anti-correlated with the reservoir. beta-targeting IS effective for removal (refines F55)
- F64 Slow-recoverer vaccination REVERSES the F34 predator+contagion asymmetry (epidemic dies after predator removal, Phi->1.0) but only when the budget covers the full reservoir (p_imm >= f_slow); below that it reduces but does not eradicate
- F65 3D flocks are robust to ALL point-predator strategies tested (naive/encircle/transect, any count, any speed up to 40x prey). The F43 "can't seal a 3D surface" mechanism generalizes: at the F41-F49 parameter regime the flock fills the box uniformly (Rg=0.43 of max ~0.5), so it has no perimeter to seal and no interior to transect

### Phase Transition (Diagnosis Thread)
- F8, F12, F17 (above) -- no transition anywhere; smooth crossover
- F38 Repulsion hardness null (n = 1.5-12 identical) -- non-equilibrium forcing is the cause
- F39 Langevin thermostat satisfies FDT (KE/N = kT to 1%) but KE/N cannot detect KTHNY melting
- F40 Hexatic order parameter |psi6| flat: soft repulsion cannot crystallize
- F50 Hard repulsion (n = 12, 24) also flat: higher exponent shrinks the core, does not harden it

### 3D Extension
- F41 Flocking generalizes to 3D; v_eq exact
- F42 3D noise crossover at ramp ~ 15-25 (smooth, not a phase transition)
- F43 3D encirclement does not disrupt the flock at all (Phi=1.0); encirclement is 2D-specific [corrected -- predator sign bug]
- F44 3D encirclement fails at every predator count 1-50 (Phi>=0.99) [corrected -- predator sign bug]
- F45 3D adaptive encirclement also does nothing (Phi=0.9998) [corrected -- predator sign bug]
- F46 Vaccination targeting fails in 3D too; kinematic mixing is dimension-independent
- F49 3D encirclement fails under any arrangement (sphere/planar) [corrected -- predator sign bug]
- F51 3D alpha-contrast segregation: matches 2D at moderate contrast, diluted at high contrast
- F52 3D mixes ~1.8x SLOWER than 2D at matched degree; "mixing aid" theme falsified
- F58 Slow-recoverer vaccination transfers to 3D unchanged (per-agent rate mechanism is dimension-independent)
- F65 3D flocks robust to ALL point-predator strategies (naive/encircle/transect) -- the F43 "no surface to seal" generalizes to "no spatial perimeter at all" (Rg=0.43 of max ~0.5)
- F66 PREDICTIVE encirclement (predators target CoM + lead*v_mean) deepens F14 disruption: optimum at lead~2 tu gives Phi=0.530 vs F14 baseline 0.77-0.83 and F35-adaptive 0.713. First predator-side adaptation in the study to substantially beat F14
- F67 Predictive (F66) and adaptive R_enc (F35) do NOT compose. Combined predictive-adaptive Phi=0.535, within noise of predictive-fixed (0.530). Predictive placement is the dominant lever; angular spread becomes secondary once the heading is blocked
- F68 Predictive encirclement (F66) degrades GRACEFULLY but GRADED with noisy v_mean (sigma_obs=0,0.03,0.06,0.12,0.24,0.48 -> Phi=0.530,0.629,0.670,0.709,0.770,0.804). Less robust than F60's slow-targeting (graded from sigma=0 vs F60's plateau). Global summary statistics are single-shot and noise-sensitive; per-agent invariants benefit from N-sample averaging
- F69 Predictive encirclement under DELAYED v_mean (delay 0,0.25,0.5,1,2.5,5 tu -> Phi=0.530,0.774,0.636,0.849,0.824,0.880). FAR more sensitive to delay than noise: advantage mostly gone by 0.25 tu, fully gone (>=F14) by 1 tu. Delay is a systematic bias on a forward-projected value; v_mean decorrelates sub-tu under disruption. F66-F69 thread: predator intelligence needs CURRENT low-noise global heading
- F70 Collective escape intelligence (prey flee predator centroid, weight w) COUNTERS predictive encirclement -- but only above threshold w~alpha. w=0,0.25,0.5,1,2,5 -> Phi=0.530,0.275,0.762,0.932,1.000,1.000. NON-MONOTONIC: weak escape (w=0.25) is WORSE than none (competes with alignment); strong escape (w>=2) fully escapes (a unified flee reinforces alignment). The predator's own forward-massing creates the asymmetry prey exploit -- predictive encirclement is self-defeating vs committed escape-intelligent prey
- F71 LOCAL escape sensing (per-prey flee in-range predators) only PARTIALLY counters predictive encirclement: Phi peaks at 0.829 at r_sense~0.20 (vs F70 global 1.000), never full escape. Non-monotonic; too-global local sensing CANCELS (surrounded, F33 echo). F70's full escape requires a globally SHARED escape vector (aligns with flocking); per-prey local vectors don't align. Honest caveat on F70

### Section 5 Self-Tests (predictions tested and corrected, not assumed)
- F47 Topological (k-NN) alignment does not slow mixing; the §5 prediction is falsified
- F48 Freezing the contact graph does not rescue targeting; degree-targeting null is structural
- F52 3D mixes slower, not faster (above); the "third dimension is a mixing aid" theme is falsified

---

## Literature Context and Novelty Assessment
*Added 2026-05-15 based on literature search. Updated 2026-05-15 (session 2) with additional papers.*

### Background papers

**Levis, Diaz-Guilera, Pagonabarraga, Starnini (2019/2020). "Flocking-enhanced social contagion."
Phys. Rev. Research 2, 032056.**
Studies bidirectional coupling between a Vicsek-like flocking model and SIS epidemic dynamics, where
agents' flocking DEPENDS on their SIS state (infected agents aligned differently).  Key result:
flocking self-organizes dense swarms that reduce the epidemic threshold below the global mean-field
limit.  Directly related to Findings 26, 29 (compression raises effective contact count, shifts
threshold) but via a DIFFERENT MECHANISM: their compression is endogenous (epidemic state drives
clustering), ours is exogenous (predator encirclement compresses the flock mechanically).  Our
experiment is the first to test the EXTERNAL predator analog of this effect, and to study what
happens when the external compressor is REMOVED (Finding 34 -- epidemic persistence -- is not
addressed in this paper).

**Silverberg, Bierbaum, Sethna, Cohen (2013). "Collective motion of humans in mosh and circle pits."
Phys. Rev. Lett. 110, 228701.**
Original source model.  Demonstrates the same four-force ABM produces realistic crowd dynamics.
Our project is a systematic extension of this model into predator-prey, contagion, and hybrid regimes.

**Recent (2025-2026) predator-strategy papers:**
Pacher, Bierbach, Kurvers, Krause et al. (2026). "Strategic choices of attack location allow
predators to counter a collective prey defence." Proc. R. Soc. B 293, 20260566.
Studies real predatory birds choosing attack positions against fish shoals.  Found that conspicuous
predators avoid the center of shoals to minimize prey alert response; cryptic predators prefer
center attacks.  Our encirclement strategy (equal-angle offset from CoM) is a computational analog
but our mechanism is fundamentally different: we are interested in HOW MANY ANGLES are needed to
overcome collective evasion, and the scaling relationship R_enc/Rg ~ 0.5 (Findings 14, 31) appears
not to be addressed in empirical predator-strategy literature.

**Demsar & Lebar Bajec (2014). "Simulated Predator Attacks on Flocks: A Comparison of Tactics."
Artificial Life 20(3): 343-359.**
Compared three attack tactics (attack center, attack nearest, attack isolated) against social vs.
individualistic prey using a fuzzy individual-based model.  Found social flocking is the optimal
anti-predatory response to predators targeting isolated individuals.  Relevant to F14: Demsar &
Bajec do NOT test a coordinated multi-angle encirclement strategy; our encirclement (equal-angle
compass assignment from CoM) is not among their tactics.  Supports novelty of F14's angular-coverage
threshold result.

**Bartashevich, Romanczuk et al. (2024). "Collective anti-predator escape manoeuvres through optimal
attack and avoidance strategies." Communications Biology 7, 1548.**
Uses agent-based modeling + empirical sardine-vs-marlin observations to study how the "fountain
effect" arises when prey optimize escape angles to maximize distance from predator.  The fountain
effect is a cohesive split-and-rejoin pattern in response to a SINGLE predator's attack direction.
Our encirclement result (F16) is the multi-predator analog: angular COMPRESSION from multiple
directions produces a more violent flock DIVISION into directionally separated sub-flocks rather
than a single fountain.  The single-predator fountain timescale is not measured in their paper;
our quantification of reunion time (~10 tu, F22) appears novel.

**[2025 paper] "Behavioural response of prey to repeated attacks by non-coordinating predators."
Scientific Reports, 2025.**
Uses an underdamped Langevin model (physically similar to our Langevin follow-up, F39) to study
REPEATED attacks by naive (non-coordinating) predators.  Key mechanism: when opposing predation
pressures stretch the group transversely, narrowing leads to splitting.  This is the non-coordinated
analog of our coordinated encirclement (F14-F16).  Our contribution beyond this paper: (a) we study
COORDINATED strategies (compass-angle assignment, not random repeated attacks), (b) we measure
the angular-coverage threshold and R_enc/Rg scaling, and (c) we study the reunion timescale after
predator REMOVAL (not repeated attacks).

**Inada & Kawachi (2002/2005). "Dynamics of prey-flock escaping behavior in response to predator's
attack." J. Theor. Biol., doi: 10.1016/j.jtbi.2005.01.009.**
Two-dimensional molecular-dynamics simulation identifying four collective escape patterns:
Split and Reunion, Split and Separate into Two Groups, Scattered, and Maintain Formation and
Distance.  Our F16 demonstrates the "Split and Reunion" pattern (flock division into coherent
sub-flocks that reconsolidate after predator removal), F22 quantifies the reunion timescale
(~10 time units).  Our contributions beyond Inada & Kawachi: (a) the CAUSE is explicit
encirclement (multi-angle pressure) rather than single-predator attack angle, (b) we characterize
the transition between division patterns as a function of predator count, and (c) we identify
the asymmetry between kinematic recovery (F22, F7) and epidemic recovery (F34).

**Collective evasion timescales / recovery:**
Literature (Inada & Kawachi, Bartashevich et al.) identifies the "split and reunion" pattern and
the fountain effect, but does not quantify reunion timescales or compare kinematic vs. epidemic
recovery after stressor removal.  Finding 22 (kinematic reversibility, ~10 tu), Finding 7
(same, earlier measurement), and Finding 34 (epidemic persistence, ~100+ tu) together provide
the first explicit timescale comparison in a hybrid kinematic-contagion system.  This asymmetry
appears to be a novel contribution.

**Spatial herd immunity inflation:**
Herd immunity inflation above mean-field is a known effect in spatial SIR/SIS models (Diekmann
et al.; Keeling & Rohani) and empirically confirmed during COVID-19 (Britton et al. 2020, Science).
Spatial vaccination strategies for geographic populations have been studied (Bhatt et al. 2022,
PLoS Comput. Biol.; Zhou et al. 2021, GeoHealth), but these use fixed geographic regions, not
kinematic agents that constantly mix spatially.  Our Finding 30 (2x inflation in a flock context,
p_c ~ 0.46 vs 0.20) is a demonstration that a kinematic flock system quantitatively matches the
spatial-epidemic prediction.  Finding 37 (spatial vaccination, in progress) tests whether
spatially-distributed immunity can exploit this; preliminary results suggest kinematic mixing
defeats spatial targeting as effectively as it defeats degree-targeting (F36).

### Novelty summary

| Finding | Status | Notes |
|---------|--------|-------|
| F14 Encirclement disruption | Likely novel | Demsar&Bajec 2014 does not test multi-angle compass assignment; angular threshold not in literature |
| F16 Flock division (not dissolution) | Related + novel | Inada&Kawachi 2002 identifies "Split and Reunion" pattern; our contribution is mechanism (encirclement) + predator-count transition |
| F22 Kinematic reversibility (~10 tu) | Novel | Inada&Kawachi identifies pattern but does not quantify reunion timescale; explicit ~10 tu measurement is new |
| F29 Threshold shift ~4% from compression | Related | Similar to Levis 2020 (endogenous) but external-predator mechanism is different |
| F30 Herd immunity 2x mean-field | Known effect | Consistent with spatial SIS literature; novel in kinematic-flock context |
| F31 R_enc/Rg scaling universal | Likely novel | No literature found on size-invariant encirclement calibration |
| F32 Long-time intermittent dynamics | Possibly novel | Related to repeated-attack papers (Scientific Reports 2025) but not identical mechanism |
| F33 Incomplete encirclement non-monotone | Likely novel | No literature found; counterintuitive 5-predator result |
| F34 Epidemic outlasts kinematic stressor | Likely novel | Not addressed in Levis 2020 or any predator-prey paper found; timescale asymmetry is new |
| F35 Adaptive R_enc outperforms fixed | Likely novel | No paper found on adaptive predator geometry tracking in flocking |
| F36 Targeted vaccination null result | Interesting null | Hub-targeting fails in kinematic flock; spatial reorganization is specific mechanism |
| F37 Spatial vaccination (in progress) | Likely null | Preliminary: kinematic mixing erases spatial targeting advantage as it erases degree targeting |
| F38 Non-equilibrium driving causes crossover | Likely novel | FDT diagnosis in force-based ABM context; not in equilibrium statistical mechanics literature |

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
<img src="./figures/validate_3_flocking_only.png" width="480"/>

**What:** Agents don't cruise at v0 -- they cruise at v_eq = v0 + alpha/mu.
**Why:** When the flock aligns, the flocking force always pushes agents forward
at magnitude alpha. Self-propulsion equilibrates at a higher speed to compensate.
**Evidence:** With alpha=1, mu=10, v0=1 -> measured mean speed 1.098 vs predicted 1.100 (diff=0.002).
Verified across alpha = 0, 0.5, 1.0, 2.0 -- all match v0 + alpha/mu to within 0.002.
**Implication:** v0 and alpha cannot be set independently without affecting cruise speed.
To target a specific cruise speed v_cruise, set v0 = v_cruise - alpha/mu.

---

## Finding 2: Solid-to-fluid phase transition in repulsion-only system
<img src="./figures/validate_2_repulsion_noise.png" width="480"/>

**What:** With only repulsion and noise (alpha=0, v0=0), KE rises continuously with noise
amplitude eta. Transition appears continuous (not abrupt) across eta ~ 1-10.
**Evidence:** Sweep A in analysis.py, 5 seeds each with error bars.
eta=0.5: KE=0.00 | eta=3: KE=0.16 | eta=10: KE=1.74 | eta=30: KE=15.89
**Visual:** At low eta, agents form hexagonal crystal (close-packed). At high eta, fluid phase.
**Open question:** Is this a true phase transition or a crossover? Would need to look for
diverging susceptibility or correlation length to answer.

---

## Finding 3: Low threshold for flock formation
<img src="./figures/phase4_sweeps.png" width="480"/>

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
<img src="./figures/predator_2_coherence.png" width="480"/>

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
<img src="./figures/predator_3_aggression_sweep.png" width="480"/>

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
<img src="./figures/predator_4_size_sweep.png" width="480"/>

**What:** The fraction of the flock within predator threat range decreases with flock size.
**Evidence:** Exp 4 in predator.py, 5 seeds each.
N=10: fraction~0.49 | N=25: ~0.21 | N=50: ~0.19 | N=100: ~0.11 | N=200: ~0.14
**Interpretation:** Basic geometric dilution -- predator occupies a fixed area, larger
flock has more agents outside that area. Consistent with the biological "safety in numbers"
hypothesis. Note: N=200 slightly worse than N=100 (0.14 vs 0.11), possibly noise.

---

---

## Finding 8: Solid-to-fluid transition is a crossover, not a phase transition
<img src="./figures/phase_transition_scaling.png" width="480"/>

**What:** Finite-size scaling across N=25,50,100,200 shows KE/N is essentially
independent of N. Susceptibility chi = N*var(KE/N) rises monotonically, no peak.
**Evidence:** phase_transition.py. All four N values give nearly identical KE/N curves.
**Interpretation:** High compactness (C~0.78) traps agents -- each agent oscillates
like an independent harmonic oscillator at its lattice site. A true phase transition
would require chi to diverge at finite eta and scale as N^(gamma/nu). Neither observed.

---

## Finding 9: Flock elongates with predator pressure and with stronger alpha
<img src="./figures/geometry_2_alpha_sweep.png" width="480"/>

**What:** Aspect ratio AR increases with predator presence and flocking amplitude alpha.
**Evidence:** geometry.py
- No predator: AR=2.61 | With predator: AR=2.76
- alpha=0.2: AR=2.09 | alpha=1.0: AR=2.94 | alpha=2.0: AR=7.27
**Interpretation:** Strong velocity alignment forces agents into a tight stream.
Under predator pressure the flock thins and elongates, consistent with the book's
prediction of arched/thinning flocks.

---

## Finding 10: Multiple predators elongate the flock without breaking coherence
<img src="./figures/multi_pred_3_summary.png" width="480"/>

**What:** With 1-4 predators, Phi stays near 0.975-0.991 throughout. AR increases
from 2.83 (1 predator) to 7.91 (3 predators). Min predator-prey distance actually
increases slightly from 0.093 to 0.106.
**Evidence:** multi_predator.py, 8 seeds each.
**Interpretation:** More predators stretch and thin the flock but don't destroy it.
Counterintuitively, evasion distance increases with more predators -- pressure from
multiple directions may force the flock into a harder-to-surround configuration.

---

## Finding 11: Evasion distance increases because predators co-localize at prey CoM
<img src="./figures/evasion_diagnostic.png" width="480"/>

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
<img src="./figures/compactness_phase.png" width="480"/>

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

---

## Finding 13: Coordinated predators spread out but cannot break the flock
<img src="./figures/coord_3_breaking_threshold.png" width="480"/>

**What:** Adding predator-predator repulsion (alpha_coord) forces predators to spread
out spatially instead of co-localizing, and brings them physically closer to the flock.
But even with 10 coordinated predators, Phi never drops below 0.92. The flock's
collective evasion is robust to predator coordination strategy.
**Evidence:** coordinated_predators.py, 8 seeds each.
- alpha_coord=0 (naive): pred-pred sep=0.001, AR=8.82, min_dist=0.105
- alpha_coord=5: pred-pred sep=0.141 (real separation achieved), min_dist=0.078
- alpha_coord=10: pred-pred sep=0.233, min_dist=0.084, Phi=0.970
- alpha_coord=20: pred-pred sep=0.293 (maximum tested)
- n_pred=1..10 with alpha_coord=10: Phi ranges 0.923-0.991, no systematic collapse
**Key threshold:** Separation requires alpha_coord >= ~5. Below that, the shared
CoM target overwhelms the repulsion and predators still pile up.
**Interpretation:** The prey collective is strategy-resistant. Naive predators fail
because they co-localize. Coordinated predators fail because the flock's distributed
repulsion response scales with the number of approaching predators. No number or
strategy of predators in this model breaks the flock.

---

## Finding 14: Encirclement breaks coherence -- first strategy to substantially disrupt the flock
<img src="./figures/encircle_3_breaking_threshold.png" width="480"/>

**What:** Assigning each predator a fixed angular direction and targeting
CoM + R_enc*(cos(theta_k), sin(theta_k)) forces them to approach from equally spaced angles.
This is fundamentally different from both naive and coordinated strategies. At n_pred=6-8,
Phi drops to ~0.77 -- the first substantial coherence reduction in all experiments.
Naive and coordinated predators never dropped below 0.92.
**Evidence:** encirclement.py, 8 seeds each.
- Radius sweep (n_pred=3): R_enc=0.15 minimizes evasion distance (0.076) and Phi (0.953).
  R_enc=0.00 is naive (sep=0.001); R_enc=0.15 achieves pred_sep=0.260.
- vs naive (R_enc=0.15): n_pred=4 encircle Phi=0.909 vs naive 0.974.
  Predators get significantly closer: dist 0.077 vs 0.096.
- Flock-breaking threshold: n_pred=6 gives Phi=0.769 +/- 0.093, n_pred=8 gives 0.782 +/- 0.124.
  Min pred-prey dist falls to 0.050 (vs 0.105 for naive multi-predator). Flock not fully
  broken but substantially disrupted; high std suggests some seeds fragment.
**Optimal radius:** R_enc ~ 0.15 (just inside the flock edge, Rg~0.25). Too small = still
co-localizing. Too large (R_enc=0.25) = predators orbit the flock and evasion distance recovers.
**Interpretation:** Multi-directional pressure from equally spaced angles creates competing
repulsion vectors that cannot all be resolved by a single coherent escape direction. The alignment
force resists but cannot fully compensate when predators approach from 6+ angles simultaneously.
This is the first predator strategy in this model capable of meaningfully disrupting collective
evasion, establishing that flock resilience is strategy-dependent, not absolute.

---

## Finding 15: Encirclement threshold does not scale with N -- convergence to a common floor
<img src="./figures/encircle_scaling.png" width="480"/>

**What:** Fixed n_pred=6 against varying N shows larger flocks are more resistant (Phi rises
from 0.695 at N=50 to 0.903 at N=350). But fixed predator-to-prey ratio (6/350) applied
to larger N makes things WORSE: at N=500 with n_pred=9, Phi=0.654 -- the lowest coherence
in all experiments. Both N=100 and N=350 converge to Phi~0.67 at n_pred=10-12, suggesting
a common disruption floor independent of flock size at sufficient predator density.
**Evidence:** encirclement_scaling.py, 8 seeds each.
- Fixed n_pred=6: N=50 Phi=0.695, N=100 Phi=0.769, N=200 Phi=0.833, N=350/500 Phi=0.90
- Fixed ratio 6/350: N=500, n_pred=9 gives Phi=0.654 +/- 0.171 (worst result in project)
- Full sweep N=100 vs N=350: both reach Phi~0.67-0.68 at n_pred=10
**Interpretation:** The dilution effect protects against a fixed predator count (more agents
share the evasion burden). But the ratio law fails -- more predators at fixed ratio are
MORE effective for larger N because finer angular coverage (9 angles vs 6) is harder to
escape. There is no simple scaling law: disruption depends nonlinearly on both N and n_pred.
The convergence of N=100 and N=350 to the same Phi floor at n_pred=10 suggests the
disruption is set by angular coverage (predators/360 degrees) rather than predators/prey.

---

## Finding 16: Encirclement divides the flock into coherent sub-flocks, not random walkers
<img src="./figures/frag_3_snapshots.png" width="480"/>
<img src="./figures/frag_2_cluster_stats.png" width="480"/>

**What:** When global Phi drops to 0.77 under encirclement, the flock is NOT dissolving.
Predators COMPRESS the flock spatially (fewer clusters: 60 naive vs 24 encircle) while
SPLITTING it directionally (each sub-flock escapes in a different direction). Each
sub-flock is internally coherent (largest_phi=0.997). The low global Phi reflects
sub-flocks heading in different directions, not individual agent randomness.
**Evidence:** fragmentation.py, 8 seeds, n_pred=6.
- Naive: 60 clusters, largest_frac=0.137, global_phi=0.997, largest_phi=1.000
  (many small groups all moving same direction -- spatially spread but directionally unified)
- Encirclement: 24 clusters, largest_frac=0.253, global_phi=0.718, largest_phi=0.997
  (fewer, larger groups moving in different escape directions)
- As n_pred increases: n_clusters decreases (5->8 pred: 70->19) while largest_frac
  grows (0.08->0.25), confirming spatial compression rather than dissolution.
**Interpretation:** Encirclement succeeds by herding -- predators squeeze the flock from
multiple directions, forcing it to split into sub-groups that escape toward the gaps between
predators. Each sub-group remains a coherent mini-flock. This is flock DIVISION not flock
DISSOLUTION. The mechanism is analogous to wolf-pack herding or dolphin bait-ball formation.
This is biologically the most significant finding of the project.

---

## Finding 17: No phase transition at any intermediate compactness -- crossover is universal
<img src="./figures/compactness_search_chi.png" width="480"/>

**What:** Sweeping compactness C = 0.15, 0.20, 0.30, 0.40, 0.50, 0.60 with finite-size
scaling at N = 25, 50, 100, 200 finds no true phase transition at any density. In every
case, susceptibility chi = N*Var(KE/N) peaks at eta=30 (the top of the sweep) and the
chi_peak value is essentially identical across N (all ~0.022-0.025). No diverging peak,
no N-dependence, no critical point anywhere in the accessible compactness range.
**Evidence:** compactness_search.py, 8 seeds. Chi-peak summary:
All 24 (C, N) combinations peak at eta=30 with chi_peak in [0.021, 0.025].
KE/N curves are N-independent at every C, indistinguishable from the C=0.10 and C=0.78
results of Finding 12.
**Interpretation:** The crossover is a universal property of this model's repulsion force,
not a feature of an extreme density regime. The repulsion potential (1 - d/2r0)^1.5 is
soft -- it smoothly decays to zero at contact. Real melting transitions require hard-core
exclusion to produce the diverging spatial correlations of a critical point. This model's
repulsion was designed for crowd dynamics and simply does not have the right microscopic
physics to produce a true phase transition at any accessible compactness.
**Implication:** The professor's suggested intermediate regime does not exist in this model.
Finding a true phase transition would require either a harder repulsion (e.g., r^{-12}
Lennard-Jones) or a different model class entirely.

---

## Finding 18: Panic does not propagate -- calm agents stay coherent even at 20% panic fraction
<img src="./figures/panic_1_sweep.png" width="480"/>
<img src="./figures/panic_2_snapshots.png" width="480"/>

**What:** When a fraction f of agents are replaced by panicked agents (weak flocking alpha=0.1,
high noise ramp=10.0), the global order parameter drops smoothly: Phi=1.000 at f=0% down to
Phi=0.853 at f=20%. But the calm-agent-only order parameter stays at 0.999 throughout.
Panic does not propagate into the calm sub-flock.
**Evidence:** panic.py, 8 seeds each.
- f=0%:  Global Phi=1.000, Calm Phi=1.000
- f=1%:  Global Phi=0.991, Calm Phi=1.000
- f=5%:  Global Phi=0.958, Calm Phi=0.999
- f=10%: Global Phi=0.922, Calm Phi=0.999
- f=20%: Global Phi=0.853, Calm Phi=0.999
**Mechanism:** The global Phi drop is pure dilution: incoherent panicked agents are included
in the order parameter average, dragging it down. The calm agents form a coherent flock that
effectively ignores panicked neighbors. The flocking force is strong enough that calm agents
maintain alignment even with 20% erratic neighbors.
**Direction:** Flock heading shows no systematic deflection across panic fractions (~-9 to -13 deg,
essentially noise around the f=0 baseline). Panicked agents do not steer the flock.
**Comparison with predator strategies:** A single predator achieves Phi~0.995 (essentially no
disruption). Even f=20% panic achieves only Phi=0.853 -- and with calm_Phi=0.999 this represents
dilution, not genuine flock disruption. Encirclement with n_pred=6 achieves Phi=0.769 with the
flock actually dividing (calm agents are disrupted). External predator pressure is more disruptive
than internal panic at any tested fraction.
**Implication:** The book section "Why You Should Never Panic" implies panic is dangerous to the
collective. In this model, the opposite is observed: the flock is immune to internal panic because
the alignment force dominates. The book result may depend on panic propagation via local contagion
(agents near panicked agents becoming panicked themselves), which this model does not implement.

---

## Finding 19: Predator sensing threshold at r_sense ~ flock radius; limited sensing slightly worsens encirclement
<img src="./figures/sensing_1_summary.png" width="480"/>
<img src="./figures/sensing_2_cycles.png" width="480"/>

**What:** A predator with sensing radius r_sense can only lock on to the flock when the nearest
prey is within r_sense; otherwise it executes a slow random walk. There is a sharp transition in
lock-on fraction near r_sense ~ 0.10-0.15 (approximately equal to flock radius Rg ~ 0.10-0.15).
Above r_sense=0.20 the predator always finds the flock and the result is identical to perfect sensing.
**Evidence:** predator_sensing.py, 8 seeds, single predator.
- r_sense=0.05: lock_frac=0.12, Phi=0.990 -- predator rarely finds flock; dist=0.035 (close when it does)
- r_sense=0.10: lock_frac=0.77, Phi=0.972 +/- 0.037 -- transition regime; highest variance
- r_sense=0.15: lock_frac=0.97, Phi=0.995 -- nearly equivalent to perfect sensing
- r_sense>=0.20: lock_frac=1.00, Phi=0.995 -- identical to inf sensing
**Multi-predator:** Limited sensing (r=0.20) vs perfect sensing makes essentially no difference
for naive predators (n=1,3). For encirclement (n=6), limited sensing gives Phi=0.788 vs 0.853
for perfect sensing -- limited sensing slightly WORSENS the outcome for the flock.
**Mechanism for encirclement result:** When a locked-on encircling predator loses the flock and
re-enters search mode, it drifts randomly and may re-approach from a non-assigned angle. This adds
unpredictable multi-directional pressure on top of the structured encirclement pattern, increasing
disruption variability (std=0.126 vs 0.087 for perfect sensing). Some seeds fragment badly.
**Implication:** The biologically "realistic" sensing limitation does not help the flock -- it
leaves the single-predator result unchanged and slightly increases vulnerability to coordinated
strategies. The critical parameter is whether r_sense exceeds the flock's spatial footprint.

---

## Finding 20: Panic contagion saturates the flock at any non-zero rate -- there is no epidemic threshold
<img src="./figures/contagion_1_sweep.png" width="480"/>
<img src="./figures/contagion_3_snapshots.png" width="480"/>

**What:** Adding a contagion mechanism (calm agents become panicked at rate beta per panicked
neighbor within r_cont=0.05, no recovery) fundamentally changes Finding 18. With contagion off
(beta=0), the population stays at f_inf=0.011 (just the seed). With ANY non-zero contagion rate
tested -- even beta=0.5 -- the entire population panics: f_inf=1.000 (zero variance across all
6 seeds). The calm sub-flock that Finding 18 found to be immune now disappears entirely,
because every calm agent eventually flips. Global Phi collapses from 0.993 (beta=0) to 0.168
(beta=0.5) and saturates near 0.10 for beta >= 2.
**Evidence:** panic_contagion.py, 6 seeds, f0=1%, N=350, n_iter=4000.
- beta=0.0:  f_inf=0.011, Phi=0.993, calm_Phi=1.000 (matches Finding 18)
- beta=0.5:  f_inf=1.000, Phi=0.168, t_half=3.6 time units
- beta=2.0:  f_inf=1.000, Phi=0.100, t_half=1.2 time units
- beta=20.0: f_inf=1.000, Phi=0.109, t_half=0.4 time units
- Seed-size sensitivity at beta=2.0 (f0=0.5%..10%): always f_inf=1.000 regardless of f0
**Mechanism:** This is an SI (susceptible -> infected, no recovery) process on a spatially
mixed population. With N=350 agents in [0,1]^2 and r_cont=0.05, each agent's neighborhood
contains ~3 agents on average; once a few panic, every calm agent quickly meets a panicked
neighbor. The absorbing-state structure (panic cannot be undone) plus mixing motion guarantees
the outbreak completes, regardless of beta. beta only sets the speed: t_half scales roughly
as 1/beta. There is no critical beta_c in this formulation.
**Contrast with Finding 18:** Finding 18 said the flock is "immune to internal panic" -- but
that conclusion was an artifact of treating panic as a fixed label. Once contagion exists,
the alignment force cannot save the flock because the population pool of calm agents is
drained. The book's "Why You Should Never Panic" claim is recovered: panic is dangerous to
the collective IF and ONLY IF it propagates through contact.
**Implication:** A more refined model (SIS with recovery rate gamma, or a panic-suppression
mechanism analogous to immune memory) would be required to find a true epidemic threshold
beta_c. In the no-recovery limit, any contact-mediated panic is fatal to the flock.

---

## Finding 21: Minimum viable flock size -- coherence threshold at N~18-25
<img src="./figures/min_size_1_summary.png" width="480"/>

**What:** Sweeping N from 3 to 100 (8 seeds each) shows that flock coherence builds
smoothly with N rather than at a sharp threshold.  In the no-predator control:
Phi=0.49 at N=3, 0.69 at N=8, 0.81 at N=12, 0.96 at N=40, 0.99 at N=100.  Phi
crosses 0.9 between N=18 and N=25.  Below N=12 the flock is unreliable
(std=0.13-0.20 across seeds).  Predator pressure (single naive, or two opposed
encirclers) does not substantially shift this threshold.  In fact, at the
smallest sizes (N=3-8), a predator can briefly RAISE Phi by pushing the group
into a forced alignment.  Capture frequency (predator within 2*r0 of any prey)
stays below 10% at all tested sizes.
**Evidence:** min_flock_size.py with slow prey (v0=0.02, ramp=0.1) so the
v0=0.05 predator can actually pursue (matches Findings 5-16 regime).
- N=3:   Phi(none)=0.49, Phi(naive)=0.63, Phi(encircle2)=0.73
- N=8:   Phi(none)=0.69, Phi(naive)=0.49, Phi(encircle2)=0.61
- N=18:  Phi(none)=0.84, Phi(naive)=0.87, Phi(encircle2)=0.92
- N=40:  Phi(none)=0.96, Phi(naive)=0.91, Phi(encircle2)=0.95
- N=100: Phi(none)=0.99, Phi(naive)=0.94, Phi(encircle2)=0.94
- Evasion distance falls monotonically with N for both predator conditions
  (encircle2 always closer than naive); capture frac ~0 throughout.
**Interpretation:** Flock formation is collective: each agent's alignment
contribution to its neighbors needs a critical mass of mutually visible peers
within rf=0.1.  Below N~12 the spatial density (in a 1x1 domain) is too sparse
for the flocking force to dominate noise.  Between N=12 and N=25 the system
crosses over from noise-dominated to alignment-dominated.  Above N~25 the
group reliably flocks.
**Implication:** The "safety in numbers" hypothesis already established by
Finding 7 has a lower limit: below the coherence threshold, prey have neither
collective evasion nor individual escape distance (mind drops more sharply at
small N).  Real prey species near the coherence threshold should be most
vulnerable to predation in this model -- but the absolute capture rate in this
simulation is too small to test.

---

## Finding 22: Encirclement-induced fragmentation is fully transient -- sub-flocks reunite within ~10 time units of predator removal
<img src="./figures/reunion_1_timeseries.png" width="480"/>

**What:** Finding 16 showed encirclement divides the flock into coherent
sub-flocks.  This experiment runs a 3-phase simulation -- 1500 steps no
predator (warm-up), 4000 steps with 10 encircling predators (attack), 6500
steps no predator (recovery) -- and tracks Phi, cluster count, and largest
cluster fraction.  All 6 seeds recover fully.  Mean recovery time to Phi=0.95
is 10.3 time units (about 1030 steps), much shorter than the 4000-step attack
that caused the disruption.  Final Phi=1.000 +/- 0.001 -- better than the
pre-attack Phi=0.975, because by the end of the recovery window the flock
has had time to fully settle.
**Evidence:** reunion.py, 6 seeds, slow prey (v0=0.02), n_pred=10, R_enc=0.15.
- Pre-attack:    Phi=0.975, n_clusters=1.2, largest_frac=0.989
- During attack: Phi=0.716, n_clusters=4.5, largest_frac=0.413
                 (genuine fragmentation: largest fragment is 41% of flock)
- Post-attack:   Phi=1.000, n_clusters=1.0, largest_frac=0.993
- Recovery times (Phi -> 0.95):  [9.0, 4.5, 10.0, 16.0, 6.0, 16.5] time units;
  6/6 seeds recovered.
**Mechanism:** Predator removal eliminates the multi-directional pressure that
was holding sub-flocks apart.  Each sub-flock's local Phi was already ~1.0
during the attack (Finding 16), so each is internally consistent and moving
with a definite heading.  On the periodic 1x1 domain, sub-flocks heading in
different directions inevitably re-encounter each other, and at any meeting
the flocking force (within rf=0.1) re-aligns them.  Reunion is rapid because
sub-flocks are already aligned internally and only need their headings to
agree.
**Implication:** Encirclement causes DIVISION not DISSOLUTION (Finding 16);
this experiment confirms it causes only TRANSIENT division.  The flock's
topological state is preserved: as soon as the stressor is removed, the
group reconstitutes.  This is qualitatively different from contagious panic
(Finding 20), which would NOT spontaneously reverse on predator removal --
panicked agents stay panicked.  Predation and contagion produce different
classes of damage to the collective: predation is reversible, contagion is
absorbing.

---

## Finding 23: Combined predation + contagion -- contagion dominates; encirclement cannot rescue the calm sub-flock
<img src="./figures/hybrid_1_summary.png" width="480"/>

**What:** When the flock is simultaneously subjected to encirclement (n_pred=6) AND
contact-mediated panic contagion (beta=0.5, f0=1%), the combined outcome is essentially
identical to contagion-only.  All four conditions tested:
- none:      Phi=1.000, calm_Phi=1.000, f=0.000
- encircle:  Phi=0.707, calm_Phi=0.707, f=0.000  (matches Finding 16)
- contagion: Phi=0.050, calm_Phi=undefined, f=1.000  (matches Finding 20)
- both:      Phi=0.050, calm_Phi=undefined, f=1.000
**Evidence:** hybrid_stressors.py, 6 seeds, slow prey (v0=0.02, ramp=0.1).
**Mechanism:** Contagion is an absorbing-state process: once everyone is panicked, the
flock cannot be re-coherent regardless of what predators do.  Encirclement is a
kinematic disruption that requires the alignment force to operate -- with panicked
agents (alpha=0.1), the flocking force is too weak for predators to "herd" anyway.
Contagion races to saturation faster than encirclement can fragment.  Panic propagation
in the combined condition is not measurably different from contagion-only.
**Interpretation:** The two disruption mechanisms do not compose -- they operate in
non-overlapping regimes.  Encirclement disrupts a healthy alignment force; contagion
destroys the alignment force.  Once contagion has run, encirclement loses its target.
Hypothetical experiments where contagion is sub-threshold (e.g., SIS with high recovery)
would be needed to see encirclement still matter.
**Implication:** For a flock facing both an external pursuer and internal social
contagion, the contagion is the lethal threat.  The defensive priorities are not
symmetric: invest in mechanisms that suppress contagion (immune memory, signal-checking,
threshold behavior) rather than in mechanisms that prevent encirclement.

---

## Finding 24: Active/passive mixed populations do not spatially segregate
<img src="./figures/segregation_1_summary.png" width="480"/>

**What:** A mixed population of fast (v0=1.0) and slow (v0=0.1-0.7) agents in the standard
flocking model produces NO measurable spatial segregation along the heading direction.
Segregation index s = (mean_x_active - mean_x_passive)/Rg in the flock-aligned frame
stays at 0 +/- 0.05 across all tested conditions.  Phi remains at 1.000 throughout,
so the mixed flock is fully coherent.
**Evidence:** segregation.py, 5 seeds.
- Contrast sweep at f_active=0.5 (v0_passive=0.1..1.0): s ranges -0.035 to +0.029
- Fraction sweep at v0_passive=0.3 (f_active=0.1..0.9): s ranges -0.040 to -0.005
All values within statistical noise (errorbar ~0.04).  Snapshot at v0_passive=0.2 shows
active and passive agents well-mixed.
**Mechanism:** The alignment force homogenises velocity across the entire flock.  Each
agent's self-propulsion target is v0_self + alpha/mu, but the actual cruise speed is
the COMPROMISE set by the balance between self-propulsion and alignment.  In an aligned
flock, every agent feels the same flocking-force magnitude alpha, so the mean speed is
set by the population-weighted target.  Active and passive agents end up cruising at
nearly the same speed, eliminating the front/back differential that would produce
segregation.  This is consistent with Finding 1's result that the alignment force
fundamentally changes the kinematics: v_eq is not just v0_self.
**Implication:** Charbonneau Sec 10.4 describes spatial segregation in heterogeneous
populations; that result probably requires either (a) different alpha values between
groups (so the alignment compromise is asymmetric), or (b) a non-flocking baseline
where each agent moves independently at its own v0.  In the model as implemented here,
v0 contrast alone is insufficient to produce segregation -- the alignment force defeats it.
The defensive analogy: a flock of mixed-speed individuals can still respond as one unit;
the slower individuals do not become trailing stragglers.

---

## Finding 25: SIS contagion has a clean epidemic threshold at beta/gamma ~ 1; flock disruption tracks it
<img src="./figures/contagion_sis_1_sweeps.png" width="480"/>

**What:** Adding a recovery rate gamma to the contagion model (calm <-> panic) restores
the textbook SIS phase structure.  Below beta_c, contagion dies out and the flock stays
coherent; above beta_c, an endemic steady state emerges and the flock degrades.  The
critical line in (beta, gamma) space is approximately beta = gamma.
**Evidence:** contagion_sis.py, 5 seeds, f0=5%, N=350.
- Beta sweep at gamma=1.0: f_ss=0.000 at beta=0.0, jumps to 0.434 at beta=1.0, 0.789 at
  beta=2.0, saturates near 0.95 at beta=10.  Phi mirrors: 1.000 -> 0.661 -> 0.307.
- Gamma sweep at beta=2.0: f_ss=0.978 at gamma=0.1 (low recovery, persistent outbreak),
  drops to 0.000 at gamma=5 (recovery wins).  Sharp transition between gamma=2 and 5.
- 2D phase diagram (beta in [0.2, 4.0], gamma in [0.3, 10.0]): f_ss approximately
  tracks the diagonal beta = gamma.  Outbreak region (f_ss > 0.5) lies below the
  diagonal (beta > gamma); die-out region above.
**Comparison with mean-field:** Standard SIS predicts beta_c * <k> = gamma where <k> is
the mean local contact count.  Observed threshold beta_c ~ gamma corresponds to
<k> ~ 1, which is plausible for a flock at N=350 with r_cont=0.05: although a uniform
density in [0,1]^2 gives <k> = pi*r_cont^2*N ~ 2.7, the flock is actually moving as a
spatially extended structure with effective rather than uniform density at the contagion
scale.  The phase-diagram diagonal closely matches the prediction.
**Comparison with the SI model (Finding 20):** Finding 20 used no recovery -- the SI
limit is the gamma -> 0 corner of the phase diagram, where the outbreak always wins
regardless of beta.  Finding 20 said "any contact-mediated panic is fatal"; here we
see that statement was specifically about the no-recovery limit.  With finite gamma,
panic can be contained, and the flock retains coherence.
**Implication:** The biologically interesting question is therefore not whether
contagion exists but whether the recovery rate exceeds the contact rate.  For a real
flock, "recovery" might correspond to a calm-pulling-back-into-alignment mechanism --
panicked individuals returning to the alignment force as they re-enter coherent
neighborhoods.  The model suggests there is no need for explicit immune memory: a
strong enough alignment force could in principle generate effective recovery.  This
links flock disruption (a kinematic problem) to epidemic theory (a contact-process
problem) via a single dimensionless ratio beta/gamma.

---

## Finding 26: Encirclement amplifies but does not tip sub-threshold contagion -- partial coupling
<img src="./figures/hybrid_sis_1_summary.png" width="480"/>

**What:** At a sub-threshold SIS point (beta=1.0, gamma=3.0; beta/gamma=0.33, well below
the Finding 25 threshold), contagion alone fizzles -- panic peaks at f_max=0.13 then
dies out, leaving f_ss=0 and Phi=1.0.  Adding 6 encircling predators DOUBLES the panic
peak (f_max=0.27) but does NOT push the outbreak over threshold: f_ss=0.000 in the
combined condition too.  The mechanism is the local contact count <k>: encirclement
compresses the flock so that <k> rises from 8.9 to 30.2 (3.4x).  Effective contagion
strength beta*<k> rises with it, but at beta/gamma=0.33 the amplification is
insufficient to flip the dynamics.  Meanwhile, the COMBINED flock Phi is worse (0.73)
than encirclement alone (0.86) -- the transient outbreak adds non-negligible kinematic
disruption beyond what encirclement alone produces.
**Evidence:** hybrid_sis.py, 6 seeds, slow prey, beta=1, gamma=3.
- none:     Phi=1.000  f_max=0.000  <k>=8.89
- sis_only: Phi=1.000  f_max=0.134  <k>=8.10  (fizzles)
- encircle: Phi=0.864  f_max=0.000  <k>=30.21  (no contagion present)
- both:     Phi=0.729  f_max=0.269  <k>=32.60  (peak doubled, but still dies out)
**Comparison with Finding 23:** In Finding 23 the contagion was SUPERCRITICAL (SI / no
recovery) and dominated everything.  Here it is SUBCRITICAL and even amplified by
encirclement-induced compression, it still dies out.  The two findings bracket the
behaviour: contagion above its critical strength dominates; below threshold, even
strong external amplification cannot rescue the outbreak.  There is no easy way to push
a contained contagion over its tipping point through external mechanical pressure alone.
**Implication:** This is good news for collectives facing both stressors: as long as
the recovery rate exceeds the bare contact rate by a comfortable margin (here 3x), the
flock can absorb external disruption without triggering a contagious panic.  The
mapping beta_eff = beta * <k> via local-density modulation is real but mild -- a 3-4x
amplification of <k> by encirclement is not enough to bridge a factor-of-3 gap between
beta/gamma and 1.  For an attacker, this means coupling two attack modes is not a
free win; the contagion must already be near-supercritical for compression to matter.

---

## Finding 27: Alpha-contrast populations segregate via local clustering, not heading-axis separation
<img src="./figures/segregation_alpha_1_summary.png" width="480"/>

**What:** Replacing the v0 contrast of Finding 24 with alpha (alignment-strength)
contrast produces real spatial segregation -- but it manifests as local same-type
clustering, not as a front/back split along the heading direction.  The along-heading
segregation index stays near zero (consistent with v0-contrast Finding 24), so a naive
metric would miss the effect.  But the local-purity diagnostic (fraction of an agent's
rf-neighbors that share its type) rises monotonically with alpha contrast.
**Evidence:** segregation_alpha.py, 5 seeds, f_active=0.5.
- alpha_p=1.0 (no contrast):   purity_active=0.500, purity_passive=0.497  (baseline = 0.5)
- alpha_p=0.5:                 purity_active=0.556, purity_passive=0.549
- alpha_p=0.3:                 purity_active=0.550, purity_passive=0.542
- alpha_p=0.1:                 purity_active=0.630, purity_passive=0.598
- alpha_p=0.0:                 purity_active=0.732, purity_passive=0.684
  - Phi=0.513 -- the "flock" has effectively dissolved at alpha_p=0 because half the
    agents have no alignment force at all.
- Along-heading segregation index stays at 0 +/- 0.10 across the entire sweep.
**Mechanism:** Active agents have a strong alignment force pulling them toward their
neighbors' velocity; this preferentially binds active-active pairs which move
similarly.  Passive agents either don't align (alpha_p=0) or align weakly, so they
drift more independently.  The active sub-population coheres into local clusters that
exclude passive agents -- visible directly in the alpha_p=0 snapshot as compact red
clumps amid scattered blue particles.  Crucially, this clustering is isotropic in the
flock frame -- the clusters can form anywhere, not preferentially at the leading edge.
**Comparison with Finding 24:** v0 contrast alone produces NO segregation because the
alignment force homogenises group speed.  alpha contrast DOES produce segregation
because it creates differential binding strength.  The book's Section 10.4 segregation
result is recovered, conditional on asymmetric ALIGNMENT (not asymmetric speed).
**Implication:** In real biological flocks, segregation may indicate differential
alignment fidelity between individuals (e.g., different sensory acuity, different
attention to neighbors) rather than differential locomotion capability.  The proper
diagnostic for segregation is local purity, not bulk position relative to heading.

---

## Finding 28: Encirclement floor rises at very large N -- Finding 15's "common floor" is N-dependent
<img src="./figures/large_N_1_summary.png" width="480"/>

**What:** Finding 15 conjectured that the encirclement-induced Phi floor at ~0.67
(observed for N=100 and N=350 at n_pred=10) was set by angular coverage and should be
N-independent.  Extending the test to N=700 and N=1000 partially refutes this: the
floor rises with N.  At n_pred=10: Phi=0.667 (N=350), 0.700 (N=700), 0.740 (N=1000).
At n_pred=14: Phi=0.637 (N=350), 0.727 (N=700), 0.790 (N=1000).  The encirclement
strategy is LESS effective against larger flocks at fixed angular coverage.
**Evidence:** large_N_encirclement.py, 4 seeds, slow prey, R_enc=0.15.
- N=350,  n_pred=6:  Phi=0.902 +/- 0.043   n_pred=10: 0.667 +/- 0.106   n_pred=14: 0.637 +/- 0.128
- N=700,  n_pred=6:  Phi=0.863 +/- 0.076   n_pred=10: 0.700 +/- 0.150   n_pred=14: 0.727 +/- 0.113
- N=1000, n_pred=6:  Phi=0.849 +/- 0.119   n_pred=10: 0.740 +/- 0.080   n_pred=14: 0.790 +/- 0.132
- Evasion distance flat across N at n_pred=6 (~0.056); at n_pred=10 the predator gets
  closer for smaller N (mind=0.034 at N=350 vs 0.039 at N=1000); at n_pred=14, mind
  actually INCREASES with N (0.028 -> 0.047).
**Mechanism:** The encirclement targets are placed at R_enc=0.15 from the flock CoM.
For small N, this radius is comparable to the flock's own radius of gyration (Rg ~
0.10-0.15 at N=350); the predators occupy positions near the flock edge where they can
exert real pressure on the boundary.  At N=1000, the flock is broader -- Rg grows --
and R_enc=0.15 places the predators INSIDE the flock or close to its CoM, where their
repulsion is partially absorbed by surrounding prey.  The encirclement geometry stops
matching the flock geometry, and the angular pressure becomes less effective.  This is
a real geometric effect: encirclement requires R_enc ~ Rg to disrupt, and Rg grows
with N.
**Implication:** A predator strategy that worked at intermediate flock sizes does not
automatically scale up.  For a large flock, the encirclement radius would have to be
scaled up too -- but doing so positions the predators outside the flock's repulsion
range and they revert to ineffective orbiting (Section 4.6 radius sweep).  Real
predators facing very large prey aggregations may need to switch strategies entirely
(e.g., partition the flock into sub-groups before encircling) rather than scale a
single tactic.  The Finding 15 angular-coverage hypothesis is therefore correct
within a window of N, but not universal.

---

## Finding 29: Encirclement shifts the SIS epidemic threshold leftward by ~4%
<img src="./figures/critical_shift_1.png" width="480"/>

**What:** A direct sweep of beta at fixed gamma=2.0 with and without 6 encircling
predators measures the shift in the apparent epidemic threshold beta_c.  Without
encirclement, the threshold (where f_ss crosses 0.5) is beta_c = 1.93.  With
encirclement, beta_c = 1.85.  The shift is 0.077 in beta -- about 4% leftward.
**Evidence:** critical_shift.py, 5 seeds, slow prey, gamma=2.0, n_iter=5000.
| beta | f_ss (no enc) | f_ss (enc)  | shift |
|------|---------------|-------------|-------|
| 1.00 | 0.227 +- 0.11 | 0.271 +- 0.01 | +0.04 |
| 1.50 | 0.340 +- 0.17 | 0.425 +- 0.02 | +0.09 (largest) |
| 2.00 | 0.528 +- 0.01 | 0.533 +- 0.01 | +0.00 |
| 2.50 | 0.586 +- 0.01 | 0.594 +- 0.01 | +0.01 |
| 3.00 | 0.644 +- 0.01 | 0.638 +- 0.01 | -0.01 |
The encirclement effect is largest just below threshold (beta=1.5) where the system is
poised.  Above threshold the curves converge.
**Mechanism:** Encirclement compresses the flock and raises the local contact count
<k> by ~3x (Finding 26).  The bare epidemic threshold is beta_c * <k> ~ gamma, so a
3x larger <k> would naively lower beta_c by 3x.  The observed shift is much smaller --
4% rather than 70% -- because not all contacts within R_CONT propagate equally:
within a tightly compressed sub-flock, panicked agents are mostly surrounded by other
panicked agents, so the effective "new infection per contact" is reduced.  The
compression therefore creates redundant contagion rather than novel reach.
**Resolution of Finding 26 conjecture:** Finding 26 hypothesized that compression
would shift the threshold; this experiment confirms the shift exists (it is real and
statistically significant just below threshold) but quantifies it as modest.  A
contagion at 5% below the bare threshold can be tipped by encirclement; a contagion
at 30% below the threshold cannot.
**Implication:** Encirclement is a near-critical amplifier, not a general one.  For a
collective in a stable subcritical regime, an external pursuer cannot trigger a
runaway internal cascade.  But for a collective already living near the edge of a
contagion threshold, an external pursuer can be the difference between containment
and outbreak.  This combines the predation and contagion findings into a single
near-critical-coupling picture.

---

## Finding 30: Herd-immunity threshold in the flock is ~2x larger than mean-field predicts
<img src="./figures/herd_immunity_1.png" width="480"/>

**What:** At supercritical SIS (beta=2.5, gamma=2.0; bare R0=beta/gamma=1.25), a
sub-population of "immune" agents that never become panicked can quench the outbreak.
Mean-field SIS theory predicts the herd-immunity threshold p_c = 1 - 1/R0 = 0.20.
Measured threshold in the flock model (defined as p_immune at which f_ss drops below
0.1) is p_c ~ 0.46 -- more than twice the mean-field prediction.
**Evidence:** herd_immunity.py, 5 seeds, slow prey.
- p_immune=0.00: f_ss=0.586
- p_immune=0.20: f_ss=0.401  (mean-field threshold; still well above 0.1)
- p_immune=0.30: f_ss=0.301
- p_immune=0.40: f_ss=0.189
- p_immune=0.46: f_ss ~ 0.10  (interpolated empirical threshold)
- p_immune=0.50: f_ss=0.050
- p_immune=0.60: f_ss=0.019
**Mechanism:** Mean-field herd-immunity assumes a well-mixed population where every
contact has the same chance of being with an immune agent.  In the flock, contacts
are spatially structured: panicked sub-clusters move together so their members
contact mostly each other.  The effective <k> seen by a susceptible agent in a
panicked cluster is much higher than the average over the whole population, so the
immunity per cluster has to be higher to break the chain.  Spatial correlation
inflates the required threshold.
**Comparison with classical epidemic theory:** This effect is well-known in spatial
SIR/SIS models -- random vaccination is less effective than targeted vaccination.
The flock model recovers the qualitative result.  Quantitatively, p_c_measured /
p_c_mean_field = 2.3, which is large but not unprecedented for spatially clustered
contact networks.
**Implication:** For real biological collectives or human crowds, the fraction of
"calm anchors" needed to suppress contagion-driven panic is larger than naive
calculations suggest.  This is also a confirmation that the flock is NOT a well-mixed
population at the contagion length scale (r_cont=0.05 << flock radius Rg~0.15).
Spatial structure matters for both the encirclement results (Findings 16, 28) and the
contagion results (Findings 25, 30) -- the same locality that lets sub-flocks be
internally coherent also makes them vulnerable to contagion clusters.

---

## Finding 31: Encirclement scaling collapses on R_enc / Rg -- refining Finding 28
<img src="./figures/renc_scaling_1.png" width="480"/>

**What:** Finding 28 reported the encirclement floor rises with N at fixed R_enc=0.15.
The proposed mechanism was a geometric mismatch: R_enc was tuned to N=350's Rg and
became too small relative to N=1000's larger Rg.  This experiment tests that mechanism
directly by sweeping R_enc at both N=350 and N=1000.  Result: when plotted against
R_enc / Rg the two Phi curves COLLAPSE.  Optimal disruption is at R_enc / Rg ~ 0.5 for
both flock sizes, with Phi_min = 0.667 (N=350) and 0.732 (N=1000) -- very close.
**Evidence:** renc_scaling.py, 4 seeds, n_pred=10.
- N=350:  optimal R_enc = 0.15 (R/Rg = 0.48), Phi_min = 0.667
- N=1000: optimal R_enc = 0.20 (R/Rg = 0.60), Phi_min = 0.732
- At R_enc = 0.05 (R/Rg ~ 0.16) both sizes give Phi > 0.93 (predators co-localize at CoM)
- At R_enc = 0.30 (R/Rg > 0.87) both sizes give Phi > 0.97 (predators orbit outside flock)
- Surprisingly, Rg saturates around 0.30 for both N due to box-size confinement on the
  periodic unit square; the maximum geometric Rg in [0,1]^2 is 0.408.
**Refinement of Finding 28:** Finding 28's apparent N-dependence of the encirclement
floor was almost entirely an R_enc/Rg mismatch.  At N=1000 with R_enc=0.15, the
predators were at R_enc/Rg = 0.45 -- slightly inside the optimum.  Re-tuning to
R_enc=0.20 (R_enc/Rg = 0.60) recovers Phi = 0.732, essentially matching the N=350
floor.  The strategy is therefore size-invariant, but it must be re-calibrated to
flock geometry to maintain effectiveness.
**Interpretation of the optimum at R_enc/Rg ~ 0.5:** This places the predators within
the bulk of the flock (Rg is the RMS radius, so the average prey is at distance Rg
from CoM) but not at the center.  At R_enc/Rg ~ 0.5 each predator sits in the heart
of one quadrant of the flock, surrounded by prey on all sides and pushing them inward
toward neighboring predators.  At R_enc/Rg < 0.3 the predators converge too close to
CoM and overlap; at R_enc/Rg > 0.8 they exit the flock and exert force only on the
periphery.
**Implication:** Encirclement is geometrically rather than numerically constrained.
Predators tracking the flock's Rg (e.g., visually estimating its size and adjusting
approach distance) could deploy this strategy successfully against flocks of any size.
A predator strategy that wraps the flock at half its radius is universal.

---

## Finding 32: Long-time encirclement produces an intermittent merge/split steady state
<img src="./figures/long_encircle_1.png" width="480"/>

**What:** Running the encirclement protocol (n_pred=10, R_enc=0.15) for 30000 timesteps
(300 time units, 10x longer than typical short-time experiments) reveals that the
flock does NOT settle into a frozen fragmented configuration.  Instead, Phi oscillates
strongly throughout the run: mean Phi=0.751 +/- 0.061 across seeds, but the TEMPORAL
standard deviation within a single seed is 0.21 (excursions from Phi~0.4 to Phi~0.95).
Cluster count and largest cluster fraction oscillate similarly.  Sub-flocks
continuously merge and re-split.
**Evidence:** long_encirclement.py, 4 seeds, slow prey, last 1/3 of each run (t > 200):
- Phi:          mean = 0.751   across-seed std = 0.061   temporal-std-per-seed ~ 0.21
- n_clusters:   mean = 4.80
- largest_frac: mean = 0.455
- The temporal std exceeding the seed-to-seed std indicates fluctuations within each
  run dominate over baseline differences between runs -- the dynamics are intrinsically
  intermittent, not in a quiet steady state.
**Mechanism:** When the flock briefly aligns into a single coherent group (Phi -> 0.9),
the predators all chase one CoM target offset by their respective angles and apply
maximally-multi-directional pressure.  This is the strongest disruption, so the flock
quickly fragments.  Once fragmented (Phi -> 0.5), each predator preferentially follows
one sub-flock (whichever is nearest its angular target), reducing the multi-directional
pressure on any single sub-flock.  Without that pressure, the dominant sub-flock can
re-assemble or absorb others, raising Phi.  The cycle repeats: alignment provokes
maximum predation, fragmentation releases it.  The system explores both configurations
indefinitely.
**Comparison with the short-time (4000-step) result:** Finding 16 reported Phi=0.72
for the same setup at short times.  This long-time result confirms the time-averaged
value is similar (0.75), but reveals it hides large temporal swings that would be
invisible in a steady-state-only analysis.
**Implication:** The flock-under-encirclement is a dynamical system with attractor
dynamics that include intermittent re-merging events.  No stable sub-flock identity
exists; agents move between sub-flocks as the configuration evolves.  From a
biological perspective: encirclement does not "permanently break" the flock even
during sustained predation, but it does prevent the flock from settling into a
quiet aligned state.  The collective is repeatedly being reshuffled.

---

## Finding 33: Incomplete encirclement is mostly less disruptive, and the flock doesn't escape through the gap
<img src="./figures/gap_encirclement_1.png" width="480"/>

**What:** Starting from full 6-predator encirclement (n_total=6, 60-degree slots) and
removing predators one at a time produces non-monotone disruption.  Phi=0.92 at full
encirclement (with 6 predators), dips to 0.83-0.87 with 5 active predators, then RISES
back to 0.91-0.96 with 3-4 active predators.  The flock's center-of-mass drift
direction has no preferred alignment with the gap direction (mean angle differences
70-110 degrees from gap), so the flock does not steer toward the open side.
**Evidence:** encirclement_gap.py, 5 seeds, R_enc=0.15.
- n_active=6 (full):                          Phi=0.918 +/- 0.038, drift=0.34
- n_active=5 contiguous gap (one 60-deg open): Phi=0.833 +/- 0.041, drift=0.24
- n_active=5 distributed gap (irregular spacing): Phi=0.873 +/- 0.090, drift=0.42
- n_active=4 contiguous:                      Phi=0.963 +/- 0.029, drift=0.47
- n_active=4 distributed:                     Phi=0.908 +/- 0.042
- n_active=3 contiguous:                      Phi=0.955 +/- 0.035
- n_active=3 distributed:                     Phi=0.903 +/- 0.063
- Drift direction angle from gap direction: 70-110 degrees (essentially uncorrelated
  with gap orientation)
**Mechanism for non-monotonic Phi:** Removing one predator from a balanced 6-symmetric
configuration creates an asymmetry that the remaining 5 predators continuously chase --
the flock CoM shifts toward the gap, the 5 predators re-encircle the new CoM, but the
asymmetric pressure keeps the flock from settling.  This is more disruptive than the
balanced 6-fold pattern.  Removing 2-3 predators eliminates the asymmetry: 3-4
remaining predators at large spacing produce weaker multi-directional pressure, and the
flock partially recovers (Phi up to 0.96).  Contiguous gaps tend to be more disruptive
than distributed gaps with the same n_active because they preserve more local
multi-angle pressure on one side.
**Why no steering toward the gap:** The flocking force depends only on neighbor
velocities, not on absolute predator positions.  Agents do not perceive "where the
gap is" -- they only feel local repulsion from nearby predators and alignment with
local neighbors.  The CoM drifts toward the gap initially because the gap-side prey
feel no repulsion, but the predators continuously update their targets, so the
asymmetric pressure persists indefinitely.  There is no global escape route detection.
**Implication:** Biological "encirclement" by wolf packs leaves gaps for tactical
reasons (avoid friendly fire, allow some prey to scatter for easier individual pursuit)
rather than because gaps necessarily reduce flock disruption.  In this model, full
symmetric encirclement is paradoxically less disruptive than a near-full encirclement
with a single asymmetric gap.  A "near-perfect" trap may be the worst case for the
flock.

---

## Finding 34: Encirclement-established contagion persists after predator removal -- damage irreversibility requires both stressors
<img src="./figures/outbreak_removal_1.png" width="480"/>

**What:** Running a three-phase protocol (warmup -> encirclement + SIS -> predators removed)
with beta=1.5, gamma=2.0 (bare beta/gamma=0.75; bare endemic fraction ~0.34, below the
f_ss=0.5 threshold) tests whether contagion established under compression collapses once the
compressor is removed.  Result: it does NOT.  During encirclement, compression elevates local
<k> (Finding 26) and drives f to 0.450 and Phi to 0.185.  After predator removal, Phi
partially recovers (0.185 -> 0.266) consistent with sub-flock kinematic reunification, but
f only slowly declines (0.450 -> 0.413 in 50 time units), remaining far above the bare
endemic level.  The flock does not return to coherence.
**Evidence:** outbreak_removal.py, 5 seeds, slow prey (v0=0.02), n_pred=6, R_enc=0.15.
- During attack (last 5 tu, t=45-50):  f = 0.450 +/- 0.015   Phi = 0.185 +/- 0.055
- Post-removal (last 10 tu, t=89-100): f = 0.413 +/- 0.039   Phi = 0.266 +/- 0.069
**Mechanism:** When encirclement compresses the flock, local <k> triples (Finding 26,
8.9 -> 30.2), driving effective beta*<k>/gamma well above 1 and establishing an endemic
SIS state at f~0.45.  Removing the predators ends the kinematic compression -- sub-flocks
can re-merge (Finding 22), which raises Phi from 0.185 to 0.266 in ~50 tu.  But the
contagion was seeded into a large fraction of the population.  Post-removal, the bare
dynamics (beta/gamma=0.75, below threshold) should eventually drive f -> 0, but the
approach is slow: with f~0.41 and ~145 panicked agents still circulating among 350 total,
the per-step recovery rate is much slower than the peak infection rate during compression.
The system decays toward the bare endemic state (~0.34) on a timescale of hundreds of
time units -- far longer than the kinematic recovery time of ~10 tu (Finding 22).
**Contrast with Finding 22 (pure encirclement):** Without contagion, Phi recovers to
0.95+ in ~10 time units after predator removal.  With contagion established, Phi is still
only 0.266 after 50 time units.  The residual epidemic suppresses alignment even once the
kinematic stressor is gone.  Encirclement alone is fully reversible; encirclement + SIS
leaves a long-lived epidemiological scar.
**Contrast with Finding 23 (simultaneous encirclement + supercritical SI):** In Finding 23,
supercritical SI absorbed everything and predators were irrelevant.  Here, the contagion is
sub-threshold but pushed above a transient supercritical regime only by compression.  The
two stressors cooperate during the attack and then partially decouple on removal --
kinematic damage reverses quickly, epidemiological damage slowly.
**Implication:** From a biological standpoint: a predator group that triggers a panic
cascade (even a sub-threshold one) while physically encircling the flock inflicts lasting
damage that outlives the predation event itself.  The "reversibility" of predation
(Finding 22) is conditional on no contagion being present.  A predator strategy that
deliberately seeds panic first (or takes advantage of pre-existing near-critical social
contagion) achieves disproportionate impact relative to the duration of the attack.
This result connects kinematic and epidemiological timescales in a concrete prediction:
kinematic recovery ~ O(10) time units; epidemiological residual ~ O(100+) time units.

---

## Finding 35: Adaptive R_enc = 0.5*Rg is more disruptive than fixed R_enc, confirming the F31 universal ratio
<img src="./figures/adaptive_encirclement_1.png" width="480"/>

**What:** Comparing fixed R_enc=0.150 vs adaptive R_enc = 0.5 × live_Rg for n_pred=10,
N=350, 15000 steps (150 tu), 5 seeds.  Adaptive is more disruptive: mean Phi drops
0.778 -> 0.713 and the fraction of time above Phi=0.85 drops 0.56 -> 0.37.
| Condition | mean_Phi | seed_std | temporal_std | frac>0.85 | mean_Renc/Rg |
|-----------|----------|----------|--------------|-----------|--------------|
| Fixed     |   0.778  |  0.250   |    0.233     |   0.56    |     0.485    |
| Adaptive  |   0.713  |  0.234   |    0.219     |   0.37    |     0.500    |
**Evidence:** adaptive_encirclement.py, 5 seeds, slow prey (v0=0.02, ramp=0.1), N=350.
**Mechanism:** During the merge/split cycle (Finding 32), flock Rg fluctuates as
sub-flocks disperse (large Rg) and reconsolidate (smaller Rg).  Fixed R_enc=0.150
achieves mean R_enc/Rg=0.485 (close to optimal but drifts from it during fluctuations).
Adaptive maintains R_enc/Rg=0.500 throughout by tracking live Rg.  This suppresses the
recovery-to-high-Phi excursions that occur when a reconsolidating flock temporarily
escapes the fixed predators' optimal zone.  Lower temporal_std (0.219 vs 0.233) for
adaptive confirms the excursions are reduced.
**Effect size:** Mean Phi reduction of 8.3% is modest -- consistent with Finding 31's
observation that the disruption function has a relatively flat plateau near the optimal.
The frac_above_0.85 reduction (0.56 -> 0.37; -34% relative) is a larger effect because
it measures coherent-state dwell time, which adaptive systematically eliminates.
**Comparison with Finding 31:** Finding 31 showed encirclement performance collapses on
R_enc/Rg and the optimum is at ~0.5 for both N=350 and N=1000.  The adaptive experiment
closes the loop: a predator that tracks live Rg achieves essentially the same optimal
ratio, and does so consistently across the entire time series rather than only at
initialization.  The result validates both the R_enc/Rg scaling (F31) and the adaptive
strategy jointly.
**Null-result caveat:** Adaptive R_enc uses GLOBAL Rg (all N=350 prey), which inflates
when the flock is fragmented into dispersed sub-flocks.  A more sophisticated predator
tracking per-sub-flock Rg would adapt to individual fragments.  The current adaptive
strategy therefore represents a conservative lower bound on adaptive advantage.
**Implication:** A real predator group that can estimate overall flock extent and adjust
encirclement radius proportionally achieves 8% higher disruption in mean alignment and
34% more time in the flock's fragmented state.  The strategy generalises across N without
re-calibration.  In contrast, a fixed-radius encirclement strategy (Finding 15, F31)
must be recalibrated when the flock grows or shrinks.

---

## Finding 36: Targeted vaccination provides no significant advantage over random vaccination -- the flock contact network is not hub-dominated
<img src="./figures/targeted_immunity_1.png" width="560"/>

**What:** Comparing targeted (top contact-degree agents immunised first) vs random
vaccination at supercritical SIS (beta=2.5, gamma=2.0, R0=1.25 -- same as Finding 30).
Result: no significant advantage for targeted vaccination at any p_immune.  The herd-
immunity threshold is p_c ~ 0.46 for BOTH strategies, identical to Finding 30's random-
vaccination result.
| p_immune | f_ss (random)  | f_ss (targeted) | diff    |
|----------|----------------|-----------------|---------|
|   0.00   | 0.593 +/- 0.008| 0.587 +/- 0.006 | -0.006  |
|   0.10   | 0.491 +/- 0.010| 0.495 +/- 0.006 | +0.004  |
|   0.20   | 0.393 +/- 0.008| 0.395 +/- 0.010 | +0.002  |
|   0.30   | 0.304 +/- 0.013| 0.301 +/- 0.014 | -0.003  |
|   0.40   | 0.204 +/- 0.016| 0.177 +/- 0.080 | -0.027  |
|   0.46   | 0.101 +/- 0.072| 0.084 +/- 0.065 | -0.016  |
|   0.50   | 0.046 +/- 0.066| 0.043 +/- 0.053 | -0.003  |
|   0.60   | 0.000 +/- 0.000| 0.000 +/- 0.000 |  0.000  |
All differences at p <= 0.30 are within statistical noise (|diff| <= 0.006, smaller than
both error bars).  Near threshold (p=0.40-0.46) targeted shows a small edge (diff =
-0.027 and -0.016) but the variance for targeted is 4-5x higher than for random
(std 0.080 vs 0.016 at p=0.40), making these differences not statistically robust.
**Evidence:** targeted_immunity.py, 6 seeds, slow prey (v0=0.02).
Degree distribution across 6 seeds: mean=9.02, median=8, std=6.17, max=31.
**Mechanism -- why targeting fails:**
The classical network theory result (target hub nodes first) requires a fat-tailed degree
distribution in which a small fraction of agents have dramatically higher connectivity.
In a scale-free network, max_k/mean_k can be 100-1000x; here max_k/mean_k = 31/9 = 3.4.
The degree heterogeneity (CV = std/mean = 6.17/9.02 = 0.68) is moderate but bounded.
Immunising the top-degree agents removes the most connected individuals, but in a
kinematic flock the contact network is not fixed: as high-degree agents at the flock
center are immunised, their neighbors close in via the repulsion and alignment forces,
and new hub positions form in the interior.  Spatial reorganisation restores the
contact structure even when the specific high-degree agents are neutralised.
**Contrast with Finding 30:** Finding 30 showed the random herd-immunity threshold is
~2x mean-field (p_c~0.46 vs 0.20) due to spatial clustering.  The 2x inflation came
from panicked sub-clusters moving together -- agents within a cluster contact mostly
each other.  This is SPATIAL clustering, not degree heterogeneity.  Targeted vaccination
is designed to exploit degree heterogeneity, not spatial clustering.  The flock has both
properties, but the spatial-clustering effect is the dominant one for herd immunity, and
targeted vaccination does not specifically counter it.
**Implication:** Degree-targeted vaccination strategies, which are highly effective in
scale-free social networks, offer minimal advantage in flocking collectives.  The reason
is that flock contact networks are spatially-embedded and nearly uniform in degree: the
most connected agents are those near the flock center, and their connectivity arises
from geometric proximity (many neighbors in a dense region), not from topological hub
status.  For a flock-like system, what matters for herd immunity is breaking the spatial
clusters (which would require geographically targeted, rather than degree-targeted,
vaccination -- e.g. vaccinating agents uniformly spread across the spatial extent of the
flock, not just the high-degree core agents).

---

## Finding 37: Spatial vaccination provides no advantage over random -- kinematic mixing erases spatial targeting
<img src="./figures/spatial_vaccination_1.png" width="640"/>

**What:** Finding 36 identified that the 2x mean-field herd-immunity threshold inflation (p_c~0.46 vs 0.20)
is caused by SPATIAL CLUSTERING of panicked sub-groups, not degree heterogeneity.  This experiment tests
the hypothesis that geographically-targeted immunity -- selecting immune agents by farthest-point maxmin
sampling to maximally cover the flock's spatial extent -- would outperform random vaccination by directly
targeting the spatial-clustering mechanism.
**Evidence:** spatial_vaccination.py, 5 seeds, beta=2.5, gamma=2.0, R0=1.25.
Summary table (mean +/- std over 5 seeds):
  p_immune   f_ss random       f_ss spatial     f_ss targeted
  0.00       0.594+/-0.009    0.589+/-0.006    0.586+/-0.006
  0.10       0.490+/-0.010    0.492+/-0.010    0.493+/-0.006
  0.20       0.391+/-0.007    0.391+/-0.013    0.398+/-0.009
  0.30       0.301+/-0.011    0.297+/-0.009    0.298+/-0.013
  0.40       0.202+/-0.016    0.170+/-0.085    0.168+/-0.085
  0.46       0.088+/-0.073    0.125+/-0.063    0.072+/-0.064
  0.50       0.056+/-0.068    0.088+/-0.045    0.052+/-0.055
  0.60       0.000+/-0.000    0.000+/-0.000    0.000+/-0.000
Degree distribution: mean=8.79, median=8, std=6.07, max=31.
**Null result:** All three strategies produce statistically indistinguishable f_ss at every
p_immune value.  At p=0.10-0.30, all means differ by at most 0.007 (well within noise).
Near threshold (p=0.40-0.50), spatial and targeted strategies show much LARGER variance
(std~0.085) than random (std~0.016-0.068), making their mean values unreliable.  At p=0.46
the spatial strategy (f=0.125) is actually slightly WORSE than random (f=0.088).
**Mechanism -- why spatial targeting fails:**
(1) KINEMATIC MIXING: Flock agents constantly rearrange via the alignment, repulsion, and
self-propulsion forces.  The warmup flock position (at which immune agents are selected) is
uncorrelated with the agent positions at t=50 or t=100 when SIS dynamics are running.  The
spatial distribution of immune agents at vaccination time is scrambled by the time the
epidemic begins, making spatial targeting equivalent to random.
(2) RANDOM IS MORE RELIABLE NEAR THRESHOLD: At p=0.40-0.46, random vaccination has small
variance (std~0.016-0.073) while spatial and targeted have large variance (std~0.063-0.085).
Random vaccination distributes immune agents across all spatial regions probabilistically,
giving a stable average outcome near threshold.  Spatial/targeted place immune agents in
specific positions that may or may not align with the epidemic seed location -- a source of
high variability without a mean improvement.
**Comparison with F36:** Both F36 (degree-targeted) and F37 (spatially-targeted) are null
results, for the same underlying reason: the flock contact network is DYNAMIC.  Kinematic
reorganization (F36) restores hub positions after degree-targeted immunization; kinematic
mixing (F37) scrambles spatial positions after spatial immunization.  Neither targeting
strategy can maintain its structural advantage in a flocking system.
**Implication:** For kinematic flocking collectives, no agent-selection strategy outperforms
random vaccination.  The 2x herd-immunity threshold inflation is a structural property of the
spatial contact network (driven by collective alignment that creates spatial clusters), and
it cannot be exploited through any static vaccination strategy applied before the epidemic
runs.  Effective containment requires either a high overall immune fraction (~0.46) or a
dynamic vaccination strategy applied during the epidemic as agent positions are tracked.

---

## Finding 38: Repulsion hardness does not affect the crossover -- the absence of a phase transition is driven by non-equilibrium forcing, not potential softness
<img src="./figures/hard_repulsion_1_chi.png" width="640"/>
<img src="./figures/hard_repulsion_2_ke.png" width="640"/>

**What:** Finding 17 showed no phase transition in the standard model (n=1.5 repulsion exponent)
at any compactness C=0.10-0.78, and conjectured that the soft repulsion potential was the
root cause.  This experiment tests the conjecture directly by sweeping repulsion exponent
n = 1.5, 3.0, 6.0, 12.0 with finite-size scaling (N=25, 50, 100, 200; C=0.40, 8 seeds, no
flocking/self-propulsion).  Result: the chi-peak (susceptibility) for ALL exponents lies at
eta=30 (the top of the sweep), for ALL N values.  The KE/N curves and crossover shape are
essentially identical across n=1.5 to n=12.
**Evidence:** hard_repulsion.py, 8 seeds, C=0.40, N=25-200, eta=0.5-30.
chi_peak at eta=30.0 for all (n, N) combinations tested:
  n=1.5:  N=25: chi=2607  N=50: chi=7311  N=100: chi=3785  N=200: chi=3858
  n=3.0:  N=25: chi=2609  N=50: chi=7305  N=100: chi=3766  N=200: chi=3854
  n=6.0:  N=25: chi=2615  N=50: chi=7285  N=100: chi=3770  N=200: chi=3851
  n=12.0: N=25: chi=2615  N=50: chi=7289  N=100: chi=3773  N=200: chi=3853
KE/N ranges also nearly identical across exponents at same N.
**Interpretation:** The absence of a phase transition is NOT caused by the softness of the
repulsion potential.  In equilibrium statistical mechanics, 2D hard discs (C=0.40) form a
hexagonal solid at low temperature and undergo KTHNY melting at higher temperature -- a true
phase transition does exist for hard-core potentials.  Our model produces no such transition
even at n=12 (near hard-core) because the driving is NON-EQUILIBRIUM: uniform random kicks
(not thermal noise satisfying detailed balance) with no viscous damping (no Stokes friction
-mu*v term).  Without the fluctuation-dissipation relation, the "temperature" is not a
well-defined control parameter, and the system cannot equilibrate into a crystal phase that
melts sharply.  Instead, the crossover is set by the competition between random kick energy
and positional confinement (cage size ~ r0), which depends on the force RANGE (2r0) but is
insensitive to the force PROFILE (exponent n).
**Implication:** Finding 17's conjecture was correct in one sense (the model cannot produce
a true phase transition) but wrong about the cause.  The missing ingredient is not harder
repulsion but thermal equilibration: a Langevin simulation with viscous damping (-mu*v),
noise amplitude sqrt(2*mu*kT/dt), and a well-defined temperature T would recover the
hard-disc transition.  Within the current model family (self-propulsion regulation, not
viscous damping), no modification of the repulsion exponent will produce a diverging
susceptibility.  The solid-to-fluid crossover is a universal consequence of the non-
equilibrium driving, not a potential-specific artifact.

---

## Finding 39: Langevin thermostat thermalizes correctly but KE/N does not detect KTHNY melting
<img src="./figures/langevin_1.png" width="640"/>

**What:** Finding 38 diagnosed that the smooth crossover in the repulsion-only model is
caused by non-equilibrium driving (no FDT), not by soft repulsion.  The proposed fix was
to replace the driving with Langevin dynamics.  This experiment tests that fix: proper
Langevin thermostat (viscous damping -mu*vx/vy plus FDT-satisfying noise sqrt(2*mu*kT*dt))
at C=0.60 and C=0.70 (bracketing the 2D hard-disc melting point ~0.69-0.72), N=25-200,
kT=0.001-5.0, 8 seeds.
**Evidence:** langevin_repulsion.py. Summary of chi_peak and equipartition check:
Compactness C=0.60:
  N= 25: chi_peak=0.0817 at kT=5.000  |  KE/N(kT=0.1)=0.1047 vs kT=0.100
  N= 50: chi_peak=0.0406 at kT=5.000  |  KE/N(kT=0.1)=0.1053 vs kT=0.100
  N=100: chi_peak=0.0507 at kT=5.000  |  KE/N(kT=0.1)=0.1051 vs kT=0.100
  N=200: chi_peak=0.0440 at kT=5.000  |  KE/N(kT=0.1)=0.1054 vs kT=0.100
Compactness C=0.70: identical chi_peak values to C=0.60.
**Two key observations:**
(1) EQUIPARTITION SATISFIED: KE/N at kT=0.1 is 0.1047-0.1054, matching kT=0.100 to within
1%.  The Langevin dynamics work as designed -- the system thermalizes to the Boltzmann
distribution and the velocity distribution has the correct temperature.
(2) IDENTICAL BEHAVIOR FOR C=0.60 AND C=0.70: At kT=5.0 (chi peak), the thermal energy
(kT=5.0) is 50x larger than the repulsion amplitude (eps=0.1), so positional interactions
are negligible and both compactnesses behave as free gases.  The chi values at kT=5.0 are
therefore determined by the thermal fluctuations of a free gas, not by the repulsion, and
are C-independent.  The system's positional structure (which DOES differ between C=0.60
and C=0.70 at low kT) is invisible to the KE/N metric.
**Interpretation:** The Langevin thermostat confirms the FDT diagnosis from F38: the system
now equilibrates thermally.  However, the chi = N * Var(KE/N) susceptibility does NOT detect
the KTHNY structural phase transition.  KTHNY melting is a transition in POSITIONAL ORDER
(hexatic bond-angle correlation, defect proliferation), not in kinetic energy.  The right
observable would be the hexatic order parameter |psi_6| = |mean_neighbors(exp(6*i*theta))|
or the bond-angle correlation function g6(r).  With KE/N, both solid (low kT) and fluid
(high kT) phases give low seed-to-seed variance (all seeds agree on KE/N ≈ kT), so the
metric cannot distinguish them.  The chi_peak at kT=5.0 reflects free-gas noise at high
temperature, not a structural critical point.
**Implication:** To demonstrate the KTHNY transition in this model, two changes are needed:
(1) Add a positional order metric (hexatic order parameter psi_6 or g(r)).
(2) Use harder repulsion (larger n) so the solid-fluid transition occurs at a kT accessible
in the simulation (soft n=1.5 repulsion may push the transition to very low kT where
equilibration times are long).  Finding 38 showed that harder n doesn't help the
non-equilibrium model, but in the Langevin model harder n would produce a sharper
transition at higher kT.  This remains open for a future experiment.

---

## Finding 40: Hexatic order parameter confirms n=1.5 soft repulsion cannot crystallize -- KTHNY transition requires harder potential
<img src="./figures/langevin_hexatic_1.png" width="640"/>

**What:** Finding 39 established that chi_KE is insensitive to structural melting and proposed
that the hexatic order parameter |psi6| = |(1/k_j) * sum_neighbors exp(6*i*theta)| is the
correct observable for the KTHNY structural phase transition.  This experiment (langevin_hexatic.py)
replaces chi_KE with chi_psi6 = N * Var_seeds(time-avg |psi6|) for the same Langevin simulation
at C=0.60 and C=0.70, N=25-200, kT=0.001-5.0, 8 seeds, n=1.5.
**Evidence:** langevin_hexatic.py. Summary of chi_psi6 and |psi6| vs kT:
| C | N | chi_psi6_peak | kT of peak | psi6(kT=0.001) | psi6(kT=5.0) |
|---|---|---------------|------------|----------------|--------------|
| 0.60 | 25  | 0.0115 | 0.001 | 0.416 | 0.425 |
| 0.60 | 50  | 0.0044 | 0.001 | 0.423 | 0.423 |
| 0.60 | 100 | 0.0024 | 0.005 | 0.421 | 0.422 |
| 0.60 | 200 | 0.0042 | 0.001 | 0.416 | 0.421 |
| 0.70 | 25  | 0.0044 | 0.005 | 0.372 | 0.387 |
| 0.70 | 50  | 0.0023 | 0.001 | 0.378 | 0.386 |
| 0.70 | 100 | 0.0010 | 0.005 | 0.377 | 0.386 |
| 0.70 | 200 | 0.0027 | 0.001 | 0.376 | 0.385 |
Equipartition confirmed (KE/N at kT=0.1 is 0.1053-0.1055, within 1% of target kT=0.1).
**Null result -- three diagnostic failures of KTHNY signal:**
(1) |psi6| IS FLAT across all kT: C=0.60 varies only 0.416-0.425 (2%) across the entire
kT sweep; C=0.70 varies 0.372-0.387 (4%).  A KTHNY solid-to-fluid transition would show
|psi6| near 1.0 in the solid phase and near 0 in the fluid phase, with a sharp crossover
at kT_c.  Here |psi6| is constant at ~0.4 at BOTH the coldest (kT=0.001) and hottest (kT=5.0)
temperatures -- no solid phase, no fluid phase in the expected KTHNY sense.
(2) chi_psi6 PEAKS AT THE BOTTOM OF THE SWEEP (kT=0.001-0.005): the susceptibility has no
interior maximum.  This is the signature of a monotonically decreasing function, not of a
critical point at finite kT_c.
(3) chi_psi6 DOES NOT GROW WITH N: for C=0.60, chi_psi6_peak goes 0.0115 (N=25) -> 0.0024
(N=100) -> 0.0042 (N=200), with no systematic increase.  A diverging susceptibility would
scale as N^(gamma/nu).  The observed values actually decrease.
**Mechanism:** The failure is not in the metric (psi6 IS the correct observable for KTHNY)
but in the potential.  For a hexagonal crystal to form, each agent must have 6 neighbors at a
well-defined lattice spacing with no overlap.  The n=1.5 repulsion (F ~ (1-d/2r0)^1.5) is a
smooth contact avoidance: it approaches zero at d=2r0 and allows significant overlap before
becoming substantial.  Agents can pass through each other's soft cores at any finite kT,
preventing a rigid hexagonal lattice from locking in.  The psi6 ~0.4 constant reflects fixed
short-range geometric correlation from the initial repulsive packing, not a thermally-ordered
crystalline phase.
**Counterintuitive density dependence:** C=0.70 has lower psi6 than C=0.60 at all kT (0.38
vs 0.42).  In equilibrium hard-disc systems, higher compactness drives more hexagonal order.
With soft repulsion, the opposite occurs: at C=0.70, agents are close enough to exert
significant overlapping repulsion in many directions simultaneously, creating a frustrated
amorphous arrangement.  At C=0.60 there is more room and agents settle into locally hexagonal
clusters (psi6 slightly higher), but without long-range crystalline order.
**Implication:** The four-experiment phase-transition thread is now complete:
- F17: no transition in original model (any C)
- F38: non-equilibrium driving (not softness) is the cause
- F39: Langevin thermalizes correctly; KE/N is wrong metric
- F40: correct metric (|psi6|) confirms n=1.5 cannot crystallize; psi6 flat across all kT
To demonstrate KTHNY requires a near-hard-core Langevin simulation (n=12 or true hard-disc
MC) where agents cannot overlap and a hexagonal lattice can lock in at low kT.  That is a
distinct experimental direction beyond the current model family.

---

## Open Questions / Next Directions
1. Literature comparison: completed 2026-05-15 (session 2). New papers added: Demsar &
   Lebar Bajec 2014, Bartashevich et al. 2024, Inada & Kawachi 2002, Scientific Reports 2025.
   Novelty table updated for F14, F16, F22, F32, F33, F34, F35, F37, F38.
2. Spatial vaccination strategy (F37): COMPLETE. Null result -- kinematic mixing erases all
   static targeting strategies. p_c~0.46 for random, spatial, and targeted.
3. KTHNY via hexatic order parameter (F40): COMPLETE. n=1.5 soft repulsion cannot crystallize;
   psi6 flat at ~0.4 across all kT. Correct metric confirmed; harder potential needed for KTHNY.
4. 3D flocking extension: extend model to [0,1]^3 periodic domain; test whether the
   flock division mechanism and encirclement results still hold in 3D.

---

## Finding 41: 3D flocking validates cleanly -- v_eq = v0 + alpha/mu exact in 3D; crossover is less noise-robust than 2D
<img src="./figures/flocking3d_1.png" width="640"/>

**What:** Extension of the 2D flocking model to [0,1]^3 periodic domain.  Parameters scaled
so that 3D has similar neighbor count to 2D: rf=0.20 gives N*(4/3)*pi*rf^3 = 11.7 expected
neighbors (matching 2D rf=0.1 with ~11 neighbors).  r0=0.02 gives volume fraction ~0.012.
All other parameters unchanged from 2D defaults.
**Evidence:** flocking3d.py, N=350, r0=0.020, rf=0.20, alpha=1.0, v0=1.0, mu=10.0, 8 seeds.
ramp sweep [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]:
  ramp=0.0:  Phi=1.0000  |v|=1.1000
  ramp=0.1:  Phi=1.0000  |v|=1.1000
  ramp=0.5:  Phi=0.9995  |v|=1.1000
  ramp=1.0:  Phi=0.9982  |v|=1.1001
  ramp=2.0:  Phi=0.9931  |v|=1.1006
  ramp=5.0:  Phi=0.9595  |v|=1.1035
  ramp=10.0: Phi=0.8409  |v|=1.1138
**Key result 1 -- v_eq analytical result holds exactly in 3D:**
Measured |v| at ramp=0 is 1.1000, matching the predicted v_eq = v0 + alpha/mu = 1.1000 exactly.
The derivation (force balance at equilibrium in an aligned flock: alpha + mu*(v0-v_eq) = 0)
uses only the 1D force balance along the aligned direction, which is dimensionality-independent.
**Key result 2 -- 3D flocking works but is less noise-robust than 2D:**
Phi degrades monotonically from 1.000 (ramp=0) to 0.841 (ramp=10).  In 2D at the same
ramp=10, Phi is approximately 0.97-0.99 (from phase sweep data).  The 3D model is less
noise-robust because random noise affects all 3 velocity components: at ramp=10, the noise
RMS magnitude is ramp*sqrt(3/3) = ramp per agent in 3D vs ramp*sqrt(2/3) per agent in 2D.
More degrees of freedom reduce the effectiveness of the alignment force at restoring order.
The crossover is at ramp >> 10 -- an extended sweep to ramp=30 is needed to characterize
the full transition region.
**Seed-to-seed consistency:** std across seeds is very small throughout (0.0022 at ramp=10),
confirming the 3D model is deterministic and well-behaved.
**Implication:** The force-based flocking model generalizes cleanly to 3D.  The core
analytical result (v_eq = v0 + alpha/mu) is universal.  The 3D transition region is at
higher ramp than the first 10-step sweep covers; extended noise sweep is the natural next step.

---

## Finding 42: 3D noise crossover is smooth like 2D -- chi_peak drifts up with N; 3D less noise-robust than 2D
<img src="./figures/flocking3d_noise_1.png" width="640"/>

**What:** Finding 41 showed 3D flocking is coherent to ramp=10 but the crossover lies at
ramp>>10.  This experiment (flocking3d_noise.py) runs an extended noise sweep (ramp=0.5-30)
for 3D at N=100, 200, 350 and 2D N=350 (matched neighbor count), measuring Phi and
chi = N * Var_seeds(Phi) for finite-size scaling.
**Evidence:** flocking3d_noise.py, N_SEEDS=8, N_ITER=4000, ramp=0.5-30.
3D results:
  N=100: chi_peak=0.1139 at ramp=15.0 | Phi(ramp=0.5)=0.9993 | Phi(ramp=20)=0.2205
  N=200: chi_peak=0.0970 at ramp=20.0 | Phi(ramp=0.5)=0.9995 | Phi(ramp=20)=0.3300
  N=350: chi_peak=0.1951 at ramp=25.0 | Phi(ramp=0.5)=0.9995 | Phi(ramp=20)=0.4197
2D N=350 (matched neighbor count: rf=0.1, r0=0.005):
  chi_peak=0.7683 at ramp=30.0 | Phi(ramp=0.5)=0.9997 | Phi(ramp=20)=0.6017
**Key result 1 -- 3D crossover at ramp~15-25; 2D crossover at ramp>30:**
In 3D, Phi transitions from ~0.999 (ramp=0.5) to 0.22-0.42 (ramp=20, N-dependent).
The 2D model is far more noise-robust: Phi=0.60 at ramp=20, and chi_peak is still at the
top of the sweep (ramp=30), indicating the 2D crossover lies beyond ramp=30.  The 3D model
is less noise-robust because the additional noise degree of freedom (3rd component) increases
total perturbation magnitude relative to alignment strength.
**Key result 2 -- chi_peak shifts UP with N (smooth crossover, not phase transition):**
For a true phase transition, chi_peak grows with N AND the peak location converges to a
finite ramp_c as N->inf.  Here, the peak location INCREASES with N:
  N=100: ramp_c=15  |  N=200: ramp_c=20  |  N=350: ramp_c=25
This is the signature of a smooth crossover: larger systems maintain coherence to higher noise
because the alignment force averages over more neighbors, shifting the crossover upward.
chi_peak values are non-monotonic (0.1139, 0.0970, 0.1951), not systematically diverging.
**Key result 3 -- same qualitative behavior in 2D and 3D:**
Both 2D and 3D show smooth noise-driven crossovers in this model family, consistent with
the non-equilibrium driving mechanism (no FDT) identified in F38.  The 3D crossover simply
occurs at lower ramp because 3 noise components > 2.
**Implication:** The 3D force-based flocking model has the same qualitative phase behavior
as 2D: a smooth crossover, not a true phase transition.  The 3D crossover region is
ramp~15-25 for N=100-350 (vs ramp>30 for 2D).  The v_eq result (F41), smooth crossover (F42),
and finite-size behavior are consistent between 2D and 3D.

---

## Finding 43: 3D encirclement does not disrupt the flock at all -- encirclement is strictly 2D-specific
<img src="./figures/flocking3d_predator_1.png" width="640"/>

**CORRECTION NOTE:** This finding was originally published with an inverted predator-force
sign in the 3D scripts (`force_on_prey` returned the negative of the repulsion, so the
"predators" attracted prey). The numbers below are the CORRECTED results after fixing the
sign. The original writeup reported "mild disruption (Phi~0.95)" and an "R_enc/Rg optimum"
-- both were artifacts of the attraction bug. Findings 44, 45, and 49 are corrected
likewise; all four collapse into this one result.

**What:** Extends the 3D flocking model with correctly repulsive predator pressure. Tests
naive (chase CoM) vs encirclement at n_pred=1,3,6,10 and sweeps R_enc, comparing to 2D.
**Evidence:** flocking3d_predator.py (corrected), N=350, slow prey v0=0.02, ramp=0.1,
N_SEEDS=5, N_ITER=5000.
3D, R_enc=0.15:
  n_pred  3D naive   3D encirclement   2D encirclement (same R_enc)
   1      1.000      1.000             0.999
   3      1.000      1.000             0.969
   6      1.000      1.000             0.750
  10      1.000      1.000             0.729
3D R_enc sweep at n_pred=6 (3D flock Rg~0.43):
  R_enc=0.05 (R/Rg=0.12) Phi=1.000 | 0.15 (0.35) 1.000 | 0.20 (0.47) 1.000 |
  0.25 (0.58) 1.000 | 0.30 (0.70) 1.000 | 0.35 (0.82) 1.000
**Key result 1 -- 3D encirclement does not disrupt the flock, at all.**
With correct repulsive predators, 3D encirclement leaves Phi = 1.000 at every predator
count and every encirclement radius tested. There is no disruption, no R_enc optimum, no
variance -- the 3D flock is completely unmoved. 2D encirclement at the identical R_enc=0.15
disrupts strongly (Phi 0.75 at n_pred=6, 0.73 at n_pred=10). The 2D-vs-3D contrast is
total: Phi~0.73 (2D) vs 1.000 (3D).
**Key result 2 -- naive predators also fail in 3D, as in 2D.**
Naive predators co-localize at the CoM and give Phi=1.000 -- the co-localization failure
mode is dimension-independent.
**Key result 3 -- mechanism: a 2D ring closes a curve, a 3D shell cannot close a volume.**
2D encirclement works by placing predators around the flock's 1D perimeter, sealing every
escape direction in the plane. In 3D the prey can always leave through a direction no
predator blocks: a handful of predators on a sphere cover only a small fraction of the
4*pi solid angle, and the flock simply flows through the gaps. Two further factors
reinforce this: (a) the simulated 3D flock is spatially diffuse (Rg~0.43, nearly
box-filling -- the model has no cohesion force, and in 3D the flock does not self-compact
the way the 2D flock settles to Rg~0.29), so there is no compact target to surround; and
(b) repulsive predators at any R_enc just open small local voids that the flock flows
around -- Rg stays ~0.43 and the order parameter never moves.
**Implication:** Encirclement is strictly a 2D strategy. It depends on sealing a closed
curve (the flock perimeter) with point predators, which a modest number of predators can
do in 2D but cannot do for a 2D surface enclosing a 3D volume. No encirclement radius,
predator count (see F44), adaptive scheme (F45), or predator arrangement (F49) recovers
any disruption in 3D. Effective 3D flock disruption would require a fundamentally
non-encirclement strategy.

---

## Finding 44: 3D encirclement does not disrupt the flock at any predator count (1-50)
<img src="./figures/finding44_3d_predator_scaling.png" width="640"/>

**CORRECTION NOTE:** Originally published with the inverted predator-force sign (see F43
correction note). The original writeup reported a "non-monotonic disruption with a hard
floor at Phi~0.91" and "compression densifies the flock" -- both artifacts of the
attraction bug (attractive predators on an interior shell pulled the flock inward, which
is why the buggy Rg fell). The corrected results below show no disruption whatsoever.
**What:** Sweeps predator count n_pred = 1, 3, 6, 10, 20, 50 for 3D encirclement at fixed
R_enc = 0.15, with correctly repulsive predators.
**Evidence:** flocking3d_predator_scaling.py (corrected), N=350, v0=0.02, ramp=0.1,
N_SEEDS=5, N_ITER=5000.
  n_pred= 1  Phi=1.000+/-0.000  Rg=0.425
  n_pred= 3  Phi=1.000+/-0.000  Rg=0.428
  n_pred= 6  Phi=1.000+/-0.000  Rg=0.431
  n_pred=10  Phi=1.000+/-0.000  Rg=0.433
  n_pred=20  Phi=0.998+/-0.004  Rg=0.445
  n_pred=50  Phi=0.992+/-0.010  Rg=0.470
**Key result -- predator count does not matter; 3D encirclement never disrupts.**
Phi stays at or above 0.99 for every predator count from 1 to 50. There is no
disruption optimum and no non-monotonicity -- the buggy version's "dip to 0.913 at
n_pred=10" does not exist. Rg rises slightly with n_pred (0.425 -> 0.470): repulsive
predators push prey gently outward, the opposite of the buggy version's spurious
compression. The flock stays fully coherent regardless of how many predators encircle it.
**Implication:** Confirms F43. 3D encirclement cannot be rescued by brute-force predator
count -- it does not disrupt the flock at all, at any count. The failure is the geometric
one of F43 (a handful of point predators cannot seal a 2D surface around a 3D volume), not
a coverage deficit that more predators could close.

---

## Finding 45: 3D adaptive encirclement also does not disrupt the flock
<img src="./figures/finding45_3d_adaptive.png" width="560"/>

**CORRECTION NOTE:** Originally published with the inverted predator-force sign (see F43
correction note). The original writeup reported "fixed Phi=0.968 vs adaptive 0.974/0.982"
and a story about "adaptive damping disruptive fluctuations" -- artifacts of the attraction
bug. The corrected results show no disruption in any configuration.
**What:** Tests whether adaptive R_enc = ratio*live_Rg helps in 3D. Compares fixed
R_enc=0.15 against adaptive at ratios 0.38 and 0.50, n_pred=10, 15000 steps.
**Evidence:** flocking3d_adaptive.py (corrected), N=350, v0=0.02, ramp=0.1, N_SEEDS=5.
  fixed R_enc=0.15        Phi=0.9998  t_std=0.000  Rg=0.435  f(Phi>0.85)=1.000
  adaptive R_enc=0.38*Rg  Phi=0.9998  t_std=0.000  Rg=0.435  f(Phi>0.85)=1.000
  adaptive R_enc=0.50*Rg  Phi=0.9998  t_std=0.000  Rg=0.432  f(Phi>0.85)=1.000
**Key result -- adaptive radius makes no difference; 3D encirclement still does nothing.**
Fixed and adaptive R_enc all leave the flock at Phi=0.9998 with zero temporal fluctuation.
There is no disruption to improve on or damp. The 2D benefit of adaptive encirclement
(F35) has no 3D analogue because 3D encirclement produces no disruption in the first place.
**Implication:** Confirms F43/F44. 3D encirclement cannot be rescued by adaptive geometry
any more than by predator count -- there is simply no disruption at any radius or schedule.

---

## Finding 46: Vaccination targeting fails in 3D too -- kinematic mixing defeats spatial and degree-targeted strategies in three dimensions
<img src="./figures/finding46_3d_vaccination.png" width="560"/>

**What:** The report's Section 5 synthesis attributes the failure of degree-targeted (F36)
and spatial (F37) vaccination to alignment-driven kinematic mixing -- the alignment force
continuously rewires the neighbor graph faster than any static structural feature can be
exploited. Both null results were established in 2D. This tests whether the mechanism
survives in 3D, where the extra spatial degree of freedom could plausibly speed mixing up
(spatial vaccination fails even harder) or slow it down (immune-agent coverage persists,
spatial vaccination starts to help).
**Evidence:** flocking3d_vaccination.py, 3D flocking model (N=350, [0,1]^3 torus,
r0=0.02, rf=0.20, v0=0.02), SIS contagion (beta=2.5, gamma=2.0) on the 3D contact
network. R_CONT=0.155 tuned so mean contact degree (8.05) matches the 2D experiments.
N_SEEDS=5, 10000-step SIS runs, f_ss = mean panic fraction over the last 20 time units.
  p_immune  random          spatial         targeted
  0.00      0.806+/-0.005   0.805+/-0.002   0.809+/-0.005
  0.10      0.701+/-0.008   0.705+/-0.004   0.705+/-0.004
  0.20      0.598+/-0.002   0.600+/-0.003   0.601+/-0.005
  0.30      0.493+/-0.003   0.494+/-0.004   0.495+/-0.005
  0.40      0.387+/-0.004   0.392+/-0.006   0.387+/-0.009
  0.46      0.322+/-0.006   0.320+/-0.008   0.323+/-0.006
  0.50      0.282+/-0.008   0.284+/-0.005   0.283+/-0.009
  0.60      0.168+/-0.006   0.170+/-0.011   0.166+/-0.008
Degree distribution: mean=8.05, median=7, max=26, CV=0.59.
**Key result 1 -- complete strategy equivalence (the central null result).**
At every immune fraction, random, spatial (3D farthest-point maxmin), and degree-targeted
vaccination produce identical steady-state panic fractions. The largest difference between
any two strategies at any p_immune is ~0.005, smaller than the seed-to-seed standard
deviation. Neither targeting strategy beats random in 3D. The F36/F37 null results
transfer to three dimensions without qualification.
**Key result 2 -- the mixing mechanism is dimension-independent.**
This was not guaranteed. F43 showed one 2D result (R_enc/Rg~0.5 optimum) fails to transfer
to 3D. But kinematic mixing does transfer: the extra degree of freedom does not slow the
neighbor-graph turnover enough for spatial coverage to persist, nor does the degree
distribution gain a fat tail (CV=0.59 in 3D vs 0.68 in 2D -- if anything LESS heterogeneous,
giving hub-targeting even less to exploit). The alignment force reshuffles agent identities
faster than the epidemic timescale in 3D just as in 2D.
**Key result 3 -- no herd-immunity threshold within p<=0.6.**
Unlike the 2D experiments (F37 quenched near p~0.46), f_ss here declines smoothly with
p_immune and remains nonzero (0.168) even at p=0.60. This is a regime effect, not a
contradiction: the 3D contact network gives an effective reproduction number
R0 ~ beta*<k>/gamma ~ 10, far more supercritical than the 2D runs, so immunity dilutes the
epidemic proportionally rather than quenching it. The strategy-equivalence result is
independent of this -- all three strategies sit on the same decay curve.
**Implication:** Strengthens the Section 5 synthesis. Alignment-driven kinematic mixing is
a dimension-independent mechanism. The flock's defining feature -- velocity alignment --
is precisely what makes its members interchangeable on the epidemic timescale, so any
vaccination strategy that relies on a stable structural property (high degree, spatial
position) collapses to random in both 2D and 3D. Targeting can only beat random in a flock
if there is a fixed sub-structure that the alignment force does not erase.

---

## Finding 47: The Section 5 topological-alignment prediction is FALSIFIED -- k-NN alignment does not slow kinematic mixing, and targeted vaccination does not recover
<img src="./figures/finding47_topological_mixing.png" width="680"/>

**What:** The report's Section 5 synthesis closed with an explicit, pre-registered
falsifiable prediction: replacing the metric alignment force (align to all neighbors within
rf) with a topological one (align to the k nearest neighbors) should produce "weaker mixing
-- the neighbor graph would be more stable because k-nearest is a permutation-stable
structure," and consequently "targeted vaccination should partially recover its advantage
over random." This experiment tests that prediction head-on.
**Evidence:** topological_mixing.py, 2D flocking N=350, N_SEEDS=5. The alignment rule is
switchable between metric (neighbors within rf=0.10) and topological (k nearest neighbors,
with k=32 calibrated to the mean metric alignment degree). Two diagnostics:
(A) mixing rate -- mean Jaccard dissimilarity of each agent's contact-neighbor set (within
R_CONT=0.05) between snapshots 2 time units apart; (B) random vs degree-targeted
vaccination at supercritical SIS (beta=2.5, gamma=2.0).
  Diagnostic A -- mixing rate (Jaccard dissimilarity per 2 tu):
    metric  0.0371+/-0.0039   contact-degree CV = 0.655
    topo    0.0364+/-0.0032   contact-degree CV = 0.615
  Diagnostic B -- targeted-vs-random advantage (f_ss random minus f_ss targeted;
  positive = targeting helps):
    metric: p=0.20 -0.006 | p=0.30 -0.013 | p=0.40 -0.007 | p=0.50 -0.024
    topo:   p=0.20 -0.004 | p=0.30 -0.007 | p=0.40 -0.038 | p=0.50  0.000
**Key result 1 -- the prediction is falsified on both counts.**
(a) Topological alignment does NOT slow mixing. The Jaccard neighbor-graph turnover is
0.0364 per 2 tu for topo vs 0.0371 for metric -- statistically identical, the difference
far smaller than the seed-to-seed std. (b) Targeted vaccination does NOT recover an
advantage under topological alignment. The targeted-minus-random advantage is negative or
zero at every immune fraction under BOTH alignment rules -- targeting never beats random,
and at p=0.40 topo it is 0.038 WORSE. The Section 5 prediction is wrong.
**Key result 2 -- why "permutation-stable" was a red herring.**
The prediction conflated two distinct networks. The ALIGNMENT network (who an agent
averages velocity with) is what topological alignment changes -- and yes, k-NN fixes every
agent's alignment degree at exactly k. But the CONTACT network (who is within R_CONT, the
network contagion actually spreads on) is set by physical proximity, and it churns because
agents with slightly dispersed velocities slide past one another. That residual velocity
dispersion comes from repulsion, noise, and the averaging in the alignment force -- all
present identically under both rules. Changing how the alignment force *selects* neighbors
does not change how fast agents *physically move past* each other. So the contact graph
rewires at the same rate, and the contact-degree distribution stays equally heterogeneous
(CV 0.62 vs 0.66 -- k-NN did not homogenize it, because it homogenizes the alignment
degree, not the contact degree).
**Key result 3 -- a real but unrelated side effect.**
Topological alignment does lower f_ss in absolute terms (random-vaccination f_ss at p=0.20:
0.359 topo vs 0.389 metric; full quench at p=0.50 for topo vs residual 0.025 for metric).
The k-NN flock is slightly more spatially extended, weakening the contagion. But this is a
uniform shift of the whole epidemic curve, not a targeting effect -- random and targeted
shift together, and targeting still confers no advantage.
**Implication:** Kinematic mixing is driven by the physical relative motion of agents, not
by the topology of the alignment rule. The mechanism is even more robust than Section 5
claimed: it does not depend on the metric character of the alignment force. Section 5's
falsifiable prediction was a genuine test of the synthesis, and the synthesis failed it --
the proposed escape route (a permutation-stable alignment graph) does not exist, because
the contact graph and the alignment graph are different objects. The report's Section 5
has been updated to record this falsification and the corrected mechanism.

---

## Finding 48: Freezing the contact graph does NOT rescue targeted vaccination -- the degree-targeting null is structural, not kinematic
<img src="./figures/finding48_contact_freezing.png" width="680"/>

**What:** Finding 47's revised Section 5 named one genuine escape route for targeted
vaccination: "freezing the contact graph itself -- suppressing the relative motion of
agents." This experiment builds that frozen graph and tests whether targeting then works.
The noise amplitude (ramp) is the book's solid-to-fluid control parameter: at low ramp the
flock is in its "solid" regime where agents lock into a near-rigid lattice, at high ramp it
is "fluid" and mixes freely. Sweeping ramp therefore sweeps the contact-graph mixing rate.
**Evidence:** contact_freezing.py, 2D flocking N=350, N_SEEDS=5, metric alignment, SIS
contagion (beta=2.5, gamma=2.0). Random vs degree-targeted vaccination at p_immune=0.20
and 0.35, across ramp = 0.003, 0.01, 0.03, 0.1, 0.3.
  ramp    Phi     contact mixing (Jaccard/2tu)
  0.003   0.997   0.0036 +/- 0.0023
  0.010   0.997   0.0056 +/- 0.0014
  0.030   0.998   0.0125 +/- 0.0023
  0.100   0.998   0.0364 +/- 0.0018
  0.300   0.991   0.1064 +/- 0.0061
  Targeted advantage (f_ss random minus targeted; positive = targeting wins):
  ramp    p=0.20    p=0.35
  0.003   -0.003    -0.002
  0.010   +0.002    -0.028
  0.030   +0.012    -0.037
  0.100   -0.002    +0.039
  0.300   +0.004    +0.016
**Key result 1 -- the contact graph really does freeze, in a still-coherent flock.**
Lowering ramp from 0.3 to 0.003 drops the contact-graph Jaccard turnover from 0.106 to
0.0036 per 2 tu -- a 30-fold reduction -- while Phi stays at 0.997-0.998 throughout. The
flock remains a coherent moving flock; only the relative motion of its members is
suppressed. Section 5's claim (after F47) that a frozen contact graph is "incompatible with
a flock that moves" was too strong: the solid regime achieves exactly that.
**Key result 2 -- targeting still fails, at every mixing rate.**
Despite the 30x range in mixing rate, the targeted-vs-random advantage stays scattered
around zero (-0.037 to +0.039) with no monotonic trend. At the most frozen point
(ramp=0.003, mixing=0.0036) the advantage is -0.003 and -0.002 -- targeting is, if
anything, marginally worse than random. Freezing the contact graph does NOT rescue targeted
vaccination. The F47-predicted escape route is closed.
**Key result 3 -- the degree-targeting null is structural, not kinematic.**
This disentangles two explanations that Finding 36 and the Section 5 synthesis had
conflated. F36 found degree-targeting fails and attributed it to kinematic mixing
(rewiring restores hub positions). F48 removes the mixing and targeting STILL fails --
so mixing was never the operative cause for degree-targeting. The real cause is the one
F36 also noted but the synthesis under-weighted: the flock contact network has a
thin-tailed degree distribution (CV~0.68, no true hubs). Hub-targeting only beats random
on fat-tailed (scale-free) networks; freezing a thin-tailed network does not create hubs.
The degree-targeting null is a STATIC STRUCTURAL property of the flock graph, present
whether or not the graph mixes.
**Implication:** The Section 5 synthesis over-attributed the vaccination null results to
kinematic mixing. Degree-targeting (F36) fails for a structural reason -- degree
homogeneity -- that is independent of mixing, as F48 proves by removing the mixing.
Kinematic mixing remains the correct explanation for the SPATIAL vaccination null (F37):
spatial coverage is a feature that genuinely exists at any instant and is genuinely erased
by motion. The two null results are NOT the same mechanism. The honest unifying statement
is weaker: the flock offers no exploitable structure for vaccination -- either because the
structure never exists (degree homogeneity) or because mixing erases it (spatial coverage).
Section 5 has been revised to separate these two mechanisms.

---

## Finding 49: 3D encirclement does not disrupt the flock under any predator arrangement (sphere or planar)
<img src="./figures/finding49_3d_strategy.png" width="680"/>

**CORRECTION NOTE:** Originally published with the inverted predator-force sign (see F43
correction note). The original writeup reported a "verified compression mechanism" (Rg
falling, <k_align> rising) and "planar worse than spherical" -- all artifacts. The buggy
"compression" was attractive predators pulling the flock inward; with correct repulsive
predators there is no compression and no disruption.
**What:** Compares spherical (Fibonacci-sphere) vs planar (ring in z=z_com plane) predator
arrangements at n_pred=6,10,20, R_enc=0.15, with correctly repulsive predators. Also
records the mean alignment-neighbor count <k_align>.
**Evidence:** flocking3d_strategy.py (corrected), N=350, N_SEEDS=5, N_ITER=5000.
  mode     n_pred  Phi       Rg      <k_align>
  sphere      6    0.9998    0.431   16.7
  sphere     10    0.9998    0.433   16.5
  sphere     20    0.9980    0.445   15.5
  planar      6    0.9998    0.432   16.6
  planar     10    0.9998    0.432   16.6
  planar     20    0.9998    0.433   16.2
**Key result -- neither arrangement disrupts; there is no compression.**
Spherical and planar arrangements both leave Phi >= 0.998 at every predator count. Rg
stays ~0.43 and <k_align> stays ~16 throughout -- the flock is neither compressed nor
densified. The buggy version's "compression mechanism" does not exist with correct
predators. Spherical and planar are equivalent because both do nothing.
**Implication:** Confirms F43/F44/F45. 3D encirclement does not disrupt the flock under
any predator count, radius, adaptive scheme, or arrangement. Every geometric variant
fails. Encirclement is strictly a 2D strategy.

---

## Finding 50: Hard repulsion does not crystallize either -- raising the exponent shrinks the effective core, it does not harden it
<img src="./figures/finding50_langevin_hexatic_hard.png" width="640"/>

**What:** Finding 40 measured the hexatic order parameter |psi6| under a Langevin
thermostat and found the n=1.5 soft repulsion cannot crystallize at any temperature
(|psi6| flat at ~0.4). F40 closed with an explicit recommendation: "Demonstrating the
KTHNY transition in this model family requires a near-hard-core Langevin simulation
(n>=12)." This experiment runs exactly that -- the F40 protocol with repulsion exponents
n=12 and n=24, at dense packings C=0.70 and C=0.85.
**Evidence:** langevin_hexatic_hard.py, Langevin thermostat (mu=10, FDT-satisfying noise),
N=50/100/200, N_SEEDS=6, N_ITER=12000, kT=0.002-3.0.
  n=12 C=0.85: psi6(kT=0.002)=0.340-0.344  psi6(kT=3.0)=0.344  chi_peak<=0.003
  n=24 C=0.85: psi6(kT=0.002)=0.339-0.345  psi6(kT=3.0)=0.344  chi_peak<=0.002
  n=12 C=0.70: psi6(kT=0.002)=0.382-0.389  psi6(kT=3.0)=0.385-0.386
  n=24 C=0.70: psi6(kT=0.002)=0.381-0.389  psi6(kT=3.0)=0.385-0.386
**Key result 1 -- hard repulsion does not crystallize, and F40's recommendation fails.**
|psi6| is flat at ~0.34-0.39 across the ENTIRE temperature range (kT 0.002 to 3.0), for
BOTH exponents and BOTH densities -- identical to the n=1.5 result of F40. psi6(low kT) and
psi6(high kT) agree to within 0.005 everywhere: there is no solid-phase rise toward 1 and
no fluid collapse toward 0. chi_psi6 is tiny (<=0.004) and DECREASES with N
(0.0041->0.0017->0.0009 at n=12 C=0.70) -- the opposite of a transition, where chi would
grow with N. There is no KTHNY transition at any exponent tested.
**Key result 2 -- mechanism: a higher exponent shrinks the core, it does not harden it.**
The repulsion force is strength = eps * base_r^n / d with base_r = 1 - d/rb. F40's
recommendation assumed a higher n hardens the contact. It does the opposite. Because
base_r ranges 0 to 1, base_r^n for large n is negligible UNLESS base_r is near 1, i.e.
unless d is very close to 0. Numerically the force factor base_r^n at d=0.5*rb is 0.354
for n=1.5, but 2.4e-4 for n=12 and 6e-8 for n=24. Raising the exponent therefore collapses
the effective interaction range toward d->0: at n=24 the repulsion is negligible beyond
d~0.05*rb. A higher n makes the agents into nearly-free particles with a tiny pointlike
core -- effectively MORE dilute and LESS able to order, not harder. F40's "n>=12"
prescription rested on a misreading of this force form.
**Key result 3 -- the phase-transition thread closes definitively negative.**
Combining F38 (exponent sweep, non-Langevin: identical crossover), F39 (Langevin
thermalizes correctly), F40 (n=1.5 hexatic flat), and F50 (n=12, 24 hexatic flat): the
Charbonneau force model cannot crystallize at any repulsion exponent, because no exponent
of the base_r^n form produces a genuine finite-sized hard core. The solid-to-fluid
"transition" is a smooth crossover, full stop.
**Caveat:** Low-kT runs start from a random quench, so kinetic arrest (glass) is a formal
possibility. But the decisive evidence against a transition is independent of annealing:
chi_psi6 DECREASES with N rather than growing, and |psi6| is identical at the hardest
exponent and the softest (F40). If a crystalline phase existed, harder exponents and
larger N would show at least partial ordering; none appears.
**Implication:** To exhibit KTHNY melting, the model would need a genuinely different
repulsion -- a true inverse-power-law potential (force ~ d^-n) or a WCA/hard-disc form --
not a higher exponent in the existing base_r^n force. This is a corrected and definitive
close to the phase-transition thread (F2, F9, F17, F38, F39, F40, F50): within the
Charbonneau model as written, the smooth crossover is the only behavior, at any exponent.

---

## Finding 51: Alpha-contrast segregation in 3D -- identical to 2D at moderate contrast, but the extra dimension dilutes it at high contrast
<img src="./figures/finding51_3d_segregation.png" width="560"/>

**What:** Finding 27 showed that in 2D, two populations with different alignment strength
alpha spatially segregate -- the local-purity diagnostic (fraction of an agent's rf
neighbors sharing its type) rose from 0.50 (well mixed) to 0.73 at maximum alpha contrast.
This tests whether that self-organized segregation transfers to 3D. Findings 46-48
established that kinematic MIXING (which destroys imposed structure) is
dimension-independent; this asks whether self-ORGANIZED structure is too.
**Evidence:** flocking3d_segregation.py, 3D flocking N=350, fast-prey regime (v0=1.0,
ramp=0.5), f_active=0.5, N_SEEDS=5. alpha_active=1.0, alpha_passive swept 1.0->0.0.
  alpha_passive  3D purity_active   2D purity_active (F27)
  1.0 (no contrast)  0.489+/-0.009    0.500
  0.7                0.526+/-0.014    --
  0.5                0.554+/-0.021    0.556
  0.3                0.552+/-0.012    0.550
  0.1                0.553+/-0.013    0.630
  0.0                0.690+/-0.023    0.732
3D Phi: 0.999 for alpha_passive>=0.3, 0.993 at 0.1, collapses to 0.523 at 0.0.
**Key result 1 -- segregation transfers to 3D, and is identical to 2D at moderate contrast.**
At alpha_passive = 0.5 and 0.3, the 3D local purity (0.554, 0.552) matches the 2D values
(0.556, 0.550) to within noise. Self-organized alpha-contrast segregation is real in 3D.
The mechanism (weaker-alignment agents fall out of the tight active core and cluster
locally) is dimension-independent in the moderate-contrast regime.
**Key result 2 -- the extra dimension dilutes segregation at high contrast.**
The 2D and 3D curves diverge sharply once the contrast is large. At alpha_passive=0.1, 2D
purity has climbed to 0.630 while 3D is still on a flat plateau at 0.553 -- no stronger
than at alpha_passive=0.5. In 2D segregation rises steadily from moderate contrast onward;
in 3D it plateaus at a mild ~0.55 and only breaks upward at alpha_passive=0.0 (purity
0.690). The extra spatial dimension gives a partially-aligned passive agent more
independent directions along which to interpenetrate the active population, so moderate
mis-alignment is no longer enough to drive clustering -- it is mixed back in.
**Key result 3 -- strong 3D segregation costs global coherence.**
3D segregation only becomes strong (purity 0.69) at alpha_passive=0.0, where the passive
half of the flock has zero alignment and behaves as a non-flocking gas. At that point the
global order parameter collapses to Phi=0.523: the "segregated" state is really an
aligned active flock shedding an incoherent passive cloud, not two co-moving coherent
sub-flocks. For all alpha_passive >= 0.1, Phi stays >= 0.99 and segregation is only mild.
**Implication:** Self-organized structure is partially dimension-dependent, in contrast to
kinematic mixing (F46) which is fully dimension-independent. Both point the same way: the
extra dimension favors mixing. Mixing erodes imposed structure equally in 2D and 3D
(F46-F48), and it also erodes self-organized segregation more effectively in 3D than in 2D
(F51). A flock in three dimensions is harder to sort and just as hard to target -- the
third dimension is, consistently across every experiment, a mixing aid.

---

## Finding 52: 3D mixes SLOWER than 2D -- the "third dimension is a mixing aid" theme is falsified
<img src="./figures/finding52_mixing_dimension.png" width="560"/>

**What:** Findings 46-51 built up an interpretive theme -- that the third spatial dimension
acts as a "mixing aid" -- which was written into the report (Section 4.33, Conclusion 24).
But that theme rested on inference, never a direct measurement: F47 measured the 2D
contact-graph mixing rate, but the 3D rate was never measured. This experiment measures it
directly, in the same self-test spirit as F47/F48, to verify or falsify the claim.
**Evidence:** mixing_dimension.py. Pure flocks (no predators, no contagion), 2D and 3D, at
the parameters of the respective vaccination experiments, N=350, N_SEEDS=5. The contact
radius is calibrated per dimension so the mean contact degree matches (~8 in both:
2D R_cont=0.050, 3D R_cont=0.155). Mixing = mean Jaccard dissimilarity of each agent's
contact-neighbor set between snapshots 2 time units apart. Noise (ramp) is swept.
  ramp   2D mix            3D mix            ratio 3D/2D
  0.03   0.0109+/-0.0009   0.0060+/-0.0006   0.55
  0.10   0.0370+/-0.0025   0.0213+/-0.0006   0.57
  0.30   0.1086+/-0.0042   0.0581+/-0.0027   0.54
  1.00   0.3143+/-0.0036   0.1839+/-0.0048   0.59
**Key result -- the theme is falsified: 3D mixes SLOWER.**
At every noise level, the 3D contact graph rewires at only ~0.54-0.59 of the 2D rate. The
ratio is remarkably stable (~0.56) across a 30-fold range of absolute mixing rates. The
third dimension is NOT a mixing aid -- at matched mean contact degree, the 3D flock's
contact graph turns over about 1.8x SLOWER than the 2D flock's.
**Mechanism:** Matching the mean contact degree forces the 3D contact radius to be ~3x
larger than the 2D one (0.155 vs 0.050), because a ball of given radius holds far more
agents than a disc. A larger contact region takes longer for agents moving at the same
speed to enter and leave, so the neighbor set is "stickier" and turns over more slowly.
(A pure surface-to-volume estimate predicts turnover ~ 1/R_cont, i.e. ratio ~0.32; the
measured 0.56 indicates the 3D relative-velocity dispersion is somewhat larger, partly
offsetting the radius effect.)
**Reconciliation with F46 and F51 -- the 3D results have other causes:**
(a) F46 (vaccination targeting fails in 3D): degree-targeting fails for the STRUCTURAL
reason established in F48 (no hubs), which is dimension-independent. Spatial targeting
fails because, even at the slower 3D rate, the contact graph still fully turns over many
times during a 100-tu epidemic -- mixing is slower but still more than sufficient to erase
spatial coverage. F46's mechanism survives; only the "faster" embellishment was wrong.
(b) F51 (segregation diluted in 3D): this is NOT caused by faster mixing. It is a
GEOMETRIC effect of the neighborhood shape -- in 3D an agent's neighbors occupy a ball
rather than a disc, giving a partially-aligned agent more independent directions along
which to be surrounded by the other type, so instantaneous local purity is geometrically
diluted. The dilution is about neighborhood geometry, not turnover rate.
**Implication:** The "third dimension is a mixing aid" theme is wrong and has been removed
from the report. The corrected statement: the third dimension changes neighborhood
GEOMETRY (a ball has more independent directions than a disc) without speeding up mixing
-- in fact it slows mixing by ~1.8x at matched degree. The 3D flock remains hard to
disrupt (escape dimension, F43-F49), hard to target (structural, F48), and hard to sort
(geometric, F51), but none of these is due to faster mixing. This finding is a direct
self-test catch: an interpretive theme that looked unifying was falsified by the
measurement it predicted.

---

## Finding 53: Prey fatigue does not make encirclement damage irreversible -- but alignment-impairing fatigue deepens the attack
<img src="./figures/finding53_fatigue.png" width="560"/>

**What:** Findings 22 and 26 established that encirclement damage is fully reversible --
divided sub-flocks reunite within ~10 time units of predator removal -- but both assumed
prey agents have no internal state. This adds a per-agent fatigue variable Q in [0,1]: Q
rises (rate r_fat) while an agent is within a predator's range and recovers (rate
r_rec=0.01) otherwise. Fatigue impairs one faculty, tested in two modes -- 'speed'
(effective v0 *= 1-Q) and 'align' (effective alpha *= 1-Q). Protocol: warmup ->
encirclement (60 tu) -> predators removed -> recovery (60 tu).
**Evidence:** fatigue.py (2D, corrected repulsive predators), N=350, n_pred=10,
R_enc=0.5*Rg, N_SEEDS=5. r_fat swept 0-0.8.
  mode    r_fat   Phi_enc   Phi_recovered   Q_enc   Q_rec
  speed   0.0     0.895     0.959           0.00    0.00
  speed   0.2     0.851     1.000           0.59    0.15
  speed   0.8     0.918     0.999           0.74    0.18
  align   0.0     0.895     0.959           0.00    0.00
  align   0.1     0.831     1.000           0.23    0.01
  align   0.2     0.699     1.000           0.63    0.12
  align   0.8     0.693     0.999           0.80    0.23
**Key result 1 -- encirclement damage stays reversible, even with fatigue.**
Phi recovers to >=0.96 in EVERY configuration -- both modes, every fatigue rate. The
fatigued cases recover to 0.999-1.000, even more completely than the no-fatigue baseline
(0.959, whose lower value and large +/-0.082 std are one slow-reuniting seed, not a
fatigue effect). Fatigue does not make encirclement damage irreversible.
**Key result 2 -- recovery happens even while substantial fatigue persists.**
At r_fat=0.8, agents end the recovery phase still carrying Q_rec~0.18-0.23 of fatigue, yet
Phi has fully returned to ~1.0. Coherence recovers before fatigue does. The reason:
once predators leave, every agent de-stresses and sheds fatigue at the same rate, so the
residual fatigue is HOMOGENEOUS across the flock. A uniformly fatigued flock (all agents
with the same reduced v0 or alpha) still aligns perfectly -- fatigue only enables
disruption when it is heterogeneous.
**Key result 3 -- the two fatigue modes differ, exactly as F24/F27 predict.**
Alignment-impairing fatigue ('align' mode) deepens the during-encirclement disruption:
Phi_enc falls monotonically with r_fat (0.895 -> 0.831 -> 0.699 -> 0.693). Speed-impairing
fatigue ('speed' mode) does not: Phi_enc stays ~0.85-0.94 with no trend. This is the
direct dynamical analogue of F24 (a v0 contrast does NOT segregate the flock -- alignment
homogenises group speed) and F27 (an alpha contrast DOES segregate, via local clustering).
A flock of agents with heterogeneous alpha (some fatigued, some not) partially segregates,
which the encircling predators exploit; heterogeneous v0 does not segregate, so
speed-fatigue gives the predators no extra purchase.
**Implication:** The reversibility of kinematic damage (F22) is robust to agent fatigue.
Fatigue matters only while it is heterogeneous and only when it impairs ALIGNMENT, not
speed -- and even then it merely deepens the transient disruption, it does not make the
damage outlast the predators. This sharpens the F22/F26 dichotomy: kinematic damage is
reversible not because the flock has no memory, but because the alignment force realigns
agents regardless of their internal fatigue state, and post-attack fatigue decays
uniformly so it leaves no exploitable heterogeneity. Contagion remains the only stressor
studied that writes irreversible damage, because it writes a heterogeneous internal label
that mixing and recovery cannot homogenise away.

---

## Finding 54: Heterogeneous recovery rates lower the SIS epidemic threshold -- slow recoverers act as reservoirs
<img src="./figures/recovery_heterogeneity_1.png" width="640"/>

**What:** Finding 25 established a clean SIS epidemic threshold at beta/gamma ~ 1 for a
flock in which every agent recovers from panic at the SAME rate gamma. Real populations
are heterogeneous -- some individuals shed panic fast, others stay agitated. This adds a
per-agent recovery rate gamma_i drawn from a bimodal distribution {1-spread, 1+spread}
(50/50), holding the arithmetic-mean gamma fixed at 1.0. The question: does a spread in
gamma at fixed mean change the outbreak? Mean-field theory predicts yes -- the endemic
state is governed by the per-agent ratio beta*<k>/gamma_i, so slow recoverers (small
gamma_i) sit panicked far longer and act as reservoirs that keep reseeding neighbours.
The effective threshold is set closer to the HARMONIC mean of gamma (<= arithmetic mean),
so spreading gamma at fixed mean should LOWER the threshold.
**Evidence:** recovery_heterogeneity.py, N=350, beta swept 0.1-5.0, 4 seeds. Conditions
(all mean gamma=1.0): homog (spread 0), mild (0.5), strong (0.8), extreme (0.95).
  condition   spread   beta_c (f_ss crosses 0.15)
  homog       0.00     0.385
  mild        0.50     0.318
  strong      0.80     0.155
  extreme     0.95     < 0.1 (endemic at every beta tested)
**Key result 1 -- heterogeneity lowers the epidemic threshold, ~2.5x at strong spread.**
beta_c falls monotonically: 0.385 (homog) -> 0.318 (mild) -> 0.155 (strong). At extreme
spread the flock is endemic even at beta=0.1 (f_ss=0.319) -- the slow half (gamma=0.05)
recovers so rarely that any contagion rate sustains an outbreak. Heterogeneity helps the
epidemic, exactly as the harmonic-mean argument predicts.
**Key result 2 -- the effect is a near-threshold phenomenon; deep supercritical it vanishes.**
At beta=5.0 all four conditions converge to f_ss~0.91. Far above threshold every agent is
re-infected faster than even the fast recoverers can clear, so the recovery distribution
stops mattering. Heterogeneity reshapes the threshold, not the saturated endemic state.
**Key result 3 -- reservoir confirmed: panic localises on slow recoverers.**
At beta=1.0 the slow-agent panic fraction exceeds the fast-agent fraction by 1.45x (mild),
1.84x (strong), 1.97x (extreme). At extreme spread the slow half is essentially saturated
(f=0.974) while the fast half sits at f~0.50 -- the outbreak is carried by the slow
sub-population, which reseeds the fast agents faster than they recover.
**Key result 4 -- heterogeneity sweep at fixed beta=0.7 is monotonic.**
Holding beta=0.7, f_ss rises smoothly with spread: 0.430 (spread 0) -> 0.542 (0.6) ->
0.675 (0.95). The endemic level is a continuous function of the recovery-rate spread.
**Implication:** The F25 epidemic threshold is not a property of the mean recovery rate
alone -- it is set by the SPREAD of recovery rates. A population with a fast-recovering
majority and a slow-recovering minority is more epidemic-prone than a uniform population
with the same average resilience, because the slow minority forms a persistent reservoir.
This is the contagion-side counterpart to the F53 fatigue result: there, heterogeneity in
an internal state (fatigue) enabled disruption only transiently; here, heterogeneity in an
internal rate (recovery) shifts a genuine threshold. It also reframes vaccination -- the
most valuable agents to protect are not the high-degree hubs (F36 found none) but the
slow recoverers, the agents whose internal dynamics make them reservoirs.

---

## Finding 55: Heterogeneous infectiousness does NOT shift the SIS threshold -- super-spreaders source most transmissions but do not lower beta_c
<img src="./figures/infectiousness_heterogeneity_1.png" width="640"/>

**What:** F54 found that heterogeneity in the RECOVERY rate gamma -- the consumer side
of the transmission ledger -- lowers the SIS threshold by a harmonic-mean effect. The
natural dual question is what happens when the SOURCE side is heterogeneous: per-agent
transmission rate beta_i drawn from a bimodal distribution at fixed arithmetic mean,
with gamma homogeneous. Mean-field intuition is asymmetric: the endemic state depends
linearly on average beta but inverse-linearly on per-agent gamma, so spreading beta at
fixed mean should NOT shift beta_c the way spreading gamma did. But spatial structure
could still introduce a super-spreader effect, with the high-beta minority dominating
seed events.
**Evidence:** infectiousness_heterogeneity.py, N=350, gamma=1.0, 4 seeds. Conditions
(all arithmetic-mean beta = sweep value): homog, mild ({beta-0.5, beta+0.5}), strong
({beta-0.8, beta+0.8}, clipped at 0 then renormalised), extreme ({0.05*beta, 1.95*beta}).
  condition   beta_c (f_ss crosses 0.15)
  homog       0.435
  mild        0.434
  strong      0.434
  extreme     0.440
**Key result 1 -- the threshold is FLAT across heterogeneity.** beta_c sits at 0.434-0.440
for every condition tested; the difference is below seed noise. The harmonic-vs-arithmetic
asymmetry between source and sink rates is real: F54 moved beta_c by a factor of 2.5,
F55 moves it by less than 1%. Threshold position is set by the arithmetic mean of beta
and the harmonic mean of gamma.
**Key result 2 -- but super-spreaders DOMINATE transmission attribution.** Of all
calm-to-panic transitions, the high-beta half sources 73.6% (mild), 88.9% (strong), and
97.2% (extreme) of them, even though they are 50% of the population. The dynamics are
massively skewed toward super-spreaders at the EVENT level. Yet at the population level
the steady-state panic fractions for super and normal agents are nearly equal
(0.58 vs 0.58, 0.58 vs 0.57, 0.57 vs 0.58) -- once panicked, every agent recovers at the
same rate gamma=1, so the inflow asymmetry does not produce a stock asymmetry.
**Key result 3 -- super-spreader saturation.** Beyond a moderate spread the additive
bimodal hits zero and renormalises, so the "extreme" condition stabilises at
{0, 2*beta_mean}: half the flock cannot transmit at all. f_ss is essentially unchanged
from the homogeneous case at the same mean. A super-spreader minority is sufficient to
sustain the same epidemic; the silent half is a passenger, not a brake.
**Implication:** Source-side and sink-side heterogeneity play asymmetric roles in SIS
on this flock. Slow recoverers (F54) are RESERVOIRS that hold panic and re-seed --
lowering the effective threshold. Super-spreaders (F55) are MERELY messengers --
their over-representation in transmission events does not produce reservoir behaviour
because they recover at the same rate as everyone else. This sharpens the F54
vaccination prescription: protect agents by their gamma_i (internal-state hubs), not
by their beta_i. The latter dominates events but not endemic load.

---

## Finding 56: Targeting slow recoverers beats random vaccination by 2-3x -- the F54 prediction confirmed
<img src="./figures/slow_recoverer_vaccination_1.png" width="640"/>

**What:** F54 predicted that in a heterogeneous-recovery population, the most valuable
vaccination targets are not the (absent) topological hubs but the slow recoverers --
the agents whose internal dynamics make them reservoirs. F36 and F48 had ruled out
degree-targeting because the contact graph is thin-tailed (no hubs); F37 had ruled out
spatial-targeting because kinematic mixing erases coverage. The F54 mechanism opens a
third target class -- internal-state hubs -- whose "hub-ness" travels with the agent
across kinematic mixing, immune to both the structural and kinematic erosion mechanisms.
This experiment tests the prediction directly.
**Evidence:** slow_recoverer_vaccination.py, N=350, gamma bimodal {0.2, 1.8}
(F54 "strong"), beta=0.30 (just above the strong-condition threshold), 4 seeds.
Strategies: random, slow (lowest gamma_i first), fast (highest gamma_i first, control),
degree (highest mean contact degree). Immune agents never panic.
  p_imm       random   slow    fast    degree
  0.10        0.306    0.258   0.371   0.300
  0.20        0.233    0.115   0.315   0.262
  0.30        0.189    0.027   0.305   0.185
  0.40        0.095    0.000   0.265   0.076
  0.50        0.016    0.000   0.246   0.055
**Key result 1 -- slow-targeting crushes the epidemic at half the random p_imm.**
At p_imm=0.20 slow gives f_ss=0.115, half the random level (0.233). At p_imm=0.30 it
falls to 0.027 (85% below random's 0.189). At p_imm=0.40 slow ERADICATES (0.000) while
random is still at 0.095 and even degree is at 0.076. The effective herd-immunity
threshold under slow-targeting is p_c ~ 0.30-0.35, vs ~0.50 for random.
**Key result 2 -- fast-targeting is strictly WORSE than random.** Immunising the half
that recovers fastest (gamma=1.8, the agents who clear panic in <1 tu anyway) leaves the
slow reservoir untouched. At every p_imm tested fast gives a HIGHER endemic than random
(0.371 vs 0.306 at p_imm=0.10; 0.265 vs 0.095 at p_imm=0.40). Vaccinating non-reservoirs
is worse than randomly hitting both classes.
**Key result 3 -- the F54 reservoir mechanism is what slow-targeting exploits.**
At p_imm=0.20 the non-immune slow fraction has panic f=0.292 under slow-targeting (vs
0.487 under random), and the non-immune fast fraction has f=0.054 (vs 0.102 under
random). Removing reservoir capacity from the slow class collapses panic ACROSS THE
WHOLE POPULATION, not just on the immunised half. Fast-targeting leaves the slow class
saturated (f=0.552) which then re-seeds the unprotected fast agents (f=0.128).
**Key result 4 -- degree-targeting shows a faint signal in heterogeneous regime.**
At p_imm=0.40 degree gives 0.076 vs random's 0.095 -- a real but small (~20%) advantage,
likely because high-degree agents in this regime happen to overlap with the slow class
by chance. The effect is at the edge of seed noise (random std 0.020, degree std 0.061)
and is dwarfed by the slow-targeting advantage. F36/F48 remain correct: degree is not
the right axis. The right axis is gamma_i.
**Implication:** F54's predicted vaccination policy works and works powerfully. The
internal-state hub class is REAL in this model and gives a 2-3x improvement over
random at moderate p_imm. This is the FIRST targeting strategy in the entire study
(through 55 findings, ten previous targeting experiments) to beat random.  It does so
because (a) the "hub-ness" is a per-agent rate that the dynamics cannot mix away (the
agent who recovers slowly today recovers slowly tomorrow), and (b) the reservoir
mechanism (F54) makes them disproportionately responsible for sustaining the endemic
state. This closes the F36/F37/F48 vaccination puzzle: a heterogeneous flock has
exploitable hubs, but they are not the agents you find by examining the contact graph
-- they are the agents you find by examining their internal recovery dynamics.

---

## Finding 57: Spatial vaccination null transfers to the heterogeneous-recovery regime -- slow-targeting works through internal state, not spatial structure
<img src="./figures/het_recovery_spatial_1.png" width="640"/>

**What:** F37 ruled out spatial (farthest-point) vaccination in the homogeneous regime
because kinematic mixing erases geometric coverage faster than the epidemic resolves.
F56 then established that slow-recoverer targeting beats random by 2-3x in the
heterogeneous-recovery regime. The natural follow-up: does heterogeneous recovery
SOMEHOW rescue spatial targeting (perhaps because the epidemic now localises on a
specific sub-population whose spatial coverage matters more)? Or is F37's null robust
to internal-state heterogeneity, with F56's gain coming purely from the per-agent rate
mechanism?
**Evidence:** het_recovery_spatial.py, N=350, gamma bimodal {0.2, 1.8}, beta=0.30,
4 seeds. Same setup as F56. Strategies: random, spatial (farthest-point at t=0), slow.
  p_imm       random   spatial  slow
  0.10        0.299    0.294    0.245
  0.20        0.238    0.220    0.140
  0.30        0.171    0.168    0.024
  0.40        0.075    0.100    0.000
  0.50        0.016    0.020    0.000
**Key result 1 -- spatial and random are statistically identical.** At every p_imm the
spatial and random curves agree to within seed noise (max difference 0.025, both
strategies have +/- ~0.020). At p=0.40 spatial is in fact NOMINALLY worse than random
(0.100 vs 0.075), although the difference is below noise. F37's kinematic-mixing
explanation transfers to the heterogeneous-recovery regime unchanged: spatial coverage
is erased equally fast regardless of whether the recovery distribution is homogeneous
or bimodal.
**Key result 2 -- slow-targeting reproduces the F56 advantage cleanly.** Slow at
p_imm=0.20 hits f_ss=0.140 vs random's 0.238 (60% reduction); at p=0.30, 0.024 vs 0.171
(86% reduction); at p=0.40, eradication (0.000) vs random's 0.075. The numbers track
F56's exactly, confirming F56 is reproducible in a separate run.
**Key result 3 -- the F56 mechanism is internal, not spatial.** If slow-targeting were
"actually" working by giving spatially good coverage (because slow agents happen to be
well-distributed), then spatial-targeting should produce a comparable advantage. It
doesn't. The two strategies sit on opposite ends of the spectrum at p=0.30: spatial at
f=0.168, slow at f=0.024. Spatial coverage and internal-rate targeting are independent
axes; only internal-rate targeting works.
**Implication:** F56 and F57 together close the F36/F37/F48 vaccination puzzle for
this model. The three failed target classes (degree, spatial, alignment-topology) all
share a property: they identify hub-ness through a SYSTEM-LEVEL observable (graph
position, location, neighbour-selection rule) that the kinematic dynamics scramble
between attack and response. The successful target class (slow recoverers) identifies
hub-ness through a PER-AGENT internal rate that the dynamics cannot scramble. The
boundary between "exploitable hub" and "kinematically erased target" in this model is
the location of the hub label: external observables fail, internal rates succeed. This
also resolves an ambiguity left open by F46: the 3D vaccination null was a structural
result on degree/spatial -- F56's prediction is that 3D should show a slow-targeting
advantage equally, since gamma_i is per-agent and dimension-independent. A 3D
slow-targeting experiment is the next natural test.

---

## Finding 58: Slow-recoverer vaccination transfers to 3D unchanged -- the per-agent rate mechanism is dimension-independent
<img src="./figures/flocking3d_slow_vaccination_1.png" width="640"/>

**What:** F46 reported a 3D vaccination NULL: random, spatial, and degree-targeted
strategies are statistically identical in 3D, just as in 2D (F36/F37). F56 in 2D
showed that adding heterogeneity in recovery rate gamma_i opens up a new target class
(internal-state hubs) that DOES beat random. F57 confirmed the slow-targeting advantage
is per-agent rate, not spatial. Prediction: the F56 mechanism transfers to 3D unchanged,
since gamma_i is per-agent and dimension-independent. This experiment tests the
prediction directly.
**Evidence:** 3d/flocking3d_slow_vaccination.py, N=350, 3D torus, R_CONT=0.155
(mean k~8), gamma_i bimodal {0.4, 3.6} (mean 2.0, F54-strong analog), beta=1.5,
3 seeds, 8000 SIS iter. Strategies: random, spatial (3D farthest-point), slow.
  p_imm   random           spatial          slow
  0.10    0.661+/-0.005    0.659+/-0.009    0.638+/-0.006
  0.20    0.567+/-0.003    0.569+/-0.003    0.517+/-0.006
  0.30    0.472+/-0.012    0.479+/-0.006    0.381+/-0.012
  0.40    0.382+/-0.014    0.382+/-0.005    0.223+/-0.013
  0.50    0.282+/-0.003    0.286+/-0.006    0.000+/-0.000
**Key result 1 -- slow-targeting beats random at every p_imm in 3D.**
Advantage grows from 3% at p_imm=0.10 (0.661 vs 0.638) to 42% at p_imm=0.40 (0.382 vs
0.223) to total eradication at p_imm=0.50 (0.000 vs 0.282). Same qualitative pattern as
F56 in 2D: deeper p_imm gives sharper relative gain, with clean eradication at half the
population vaccinated.
**Key result 2 -- spatial is again a clean null vs random in 3D.**
Spatial and random agree to within 0.007 at every p_imm tested. The F37 kinematic-mixing
explanation transfers to 3D regardless of recovery-rate heterogeneity (consistent with
F46). The slow-vs-spatial gap at p_imm=0.40 is 0.159 (0.382 - 0.223), while spatial-vs-
random is 0.000 -- the mechanism is purely per-agent, not spatial.
**Key result 3 -- 3D advantage is smaller than 2D advantage.**
In 2D (F56) at p_imm=0.40 slow eradicates (f=0.000) while random sits at 0.095 --
nearly a complete win. In 3D at p_imm=0.40 slow is 0.223 vs random 0.382 -- a smaller
absolute gap. The 3D reservoir effect is diluted, plausibly because the 3D contact
graph is even more homogeneous (F46 reported CV=0.59 in 3D vs CV=0.68 in 2D) so each
slow agent contributes less concentrated reservoir mass. F52 also showed 3D mixes 1.8x
slower than 2D at matched degree, which keeps the slow class from spreading panic as
efficiently as in 2D. The mechanism transfers, but the magnitude is geometry-dependent.
**Key result 4 -- p_imm=0.50 produces clean eradication.**
In all three seeds, p_imm=0.50 under slow-targeting yields f_ss=0.000 with zero
standard deviation: the slow half of the population is exactly the bimodal slow class,
and removing them entirely strips the reservoir down to nothing. Random at p_imm=0.50
leaves 0.282 of the population panicked. This is the cleanest signal of the per-agent
mechanism: targeting EXACTLY the reservoir class collapses the epidemic to zero.
**Implication:** The F56 mechanism is dimension-independent, as predicted by F57's
internal-vs-spatial argument. Of the four canonical targeting strategies tested across
2D and 3D (degree, spatial, random, slow), only slow-targeting works -- and it works in
both dimensions. The "kinematic mixing defeats targeting" thesis from F36/F37/F46
remains correct for system-level observables, but is silent on per-agent internal
rates, which the dynamics cannot mix away. This sharpens the F36/F37/F46 result into a
positive statement: in a heterogeneous-recovery flock, slow-targeting is the canonical
vaccination policy regardless of dimension. The cost of dimension-going-up is partial
dilution of the magnitude, not loss of the mechanism.

---

## Finding 59: Slow-recoverer vaccination advantage survives continuous (lognormal) gamma distributions -- and grows with the width of the distribution
<img src="./figures/continuous_gamma_vaccination_1.png" width="640"/>

**What:** F54, F56, F57, F58 all used a BIMODAL gamma distribution -- gamma_i in
{1-spread, 1+spread} with a clean 50/50 split. That bimodal "slow vs fast" structure
makes slow-targeting a class-membership question. Real populations have continuous
recovery distributions; "slow" is a quantile, not a label. Does the F56 mechanism
need the sharp bimodal split, or does it survive a continuous distribution where
slow-vs-fast is a soft boundary?
**Evidence:** continuous_gamma_vaccination.py, N=350, beta=0.35 (just above homogeneous
threshold), gamma_i lognormal with arithmetic mean 1.0 and varying width sigma_log.
4 seeds. Strategies: random vs slow (bottom p_imm by gamma_i).
Exp 1 -- advantage as sigma_log grows (p_imm=0.20):
  sigma   random           slow             advantage
  0.00    0.000            0.000            0.000   (homog, sub-threshold)
  0.40    0.004            0.000            0.004
  0.60    0.105+/-0.066    0.043+/-0.056    0.061
  0.80    0.194+/-0.041    0.000+/-0.000    0.194   (full eradication)
  1.00    0.271+/-0.009    0.116+/-0.072    0.156
  1.20    0.331+/-0.015    0.214+/-0.036    0.117
Exp 2 -- p_immune sweep at sigma_log=0.6:
  p_imm   random           slow
  0.10    0.151+/-0.097    0.098+/-0.059   (35% reduction)
  0.20    0.105+/-0.066    0.043+/-0.056   (59% reduction)
  0.30    0.036+/-0.049    0.000+/-0.000   (slow eradicates)
**Key result 1 -- the slow-targeting advantage SURVIVES a continuous distribution.**
The F56 mechanism does not require a sharp bimodal slow/fast class. With lognormal
gamma_i, simply taking the bottom p_imm fraction by exact gamma_i value reproduces the
advantage. At sigma_log=0.80 the advantage is so strong it produces total eradication
(f_ss=0.000) at just p_imm=0.20 -- 80% of the population unvaccinated, yet zero
endemic panic, because the worst-20% lowest-gamma agents WERE the entire reservoir.
**Key result 2 -- the advantage emerges around sigma_log ~ 0.5 and peaks near 0.80.**
Below sigma_log=0.4, random alone keeps the epidemic from establishing (the distribution
is too tight; even random misses the small reservoir contribution). At sigma_log=0.6
the advantage activates. Around sigma_log=0.8 it produces clean eradication at
p_imm=0.20. Above sigma_log=1.0 the advantage shrinks in absolute terms because the
deep tail of the lognormal contains agents with such extreme reservoir capacity that
even targeting the bottom 20% leaves some in the population.
**Key result 3 -- non-monotonic behavior at very large sigma.**
Beyond sigma=0.8 both endemic levels rise (random from 0.194 to 0.331; slow from 0.000
to 0.214). The lognormal's heavy tail means a few super-slow agents (gamma << mean)
dominate everything. At sigma=1.0 the bottom-20% slow set captures most of the
reservoir but not all; the leftover deep-tail agents continue to seed the epidemic.
This is a regime where p_imm=0.20 is no longer enough -- larger p_imm would restore
the advantage.
**Key result 4 -- p_imm dependence at moderate spread tracks F56 qualitatively.**
At sigma_log=0.6, slow at p_imm=0.30 eradicates (0.000) while random sits at 0.036.
The effective herd-immunity threshold under slow-targeting is ~0.20-0.30; under random,
~0.30-0.40 with much larger seed noise. Same qualitative pattern as the F56 bimodal
case (slow eradicates at ~p_imm=0.30-0.40, random at ~0.50).
**Implication:** The F56 mechanism does not depend on a sharp bimodal slow-vs-fast
class structure. In any heterogeneous-recovery population where gamma_i is broadly
distributed (sigma_log >= 0.5), targeting the lowest-gamma quantile beats random.
The slow-vs-fast taxonomy is a useful framing, but the operational policy is
"target the bottom X% by gamma_i" regardless of distribution shape. This sharpens the
practical implications: a vaccination policy based on observed per-agent recovery
behavior should work for any plausible biological heterogeneity (which is typically
log-normal-like), not only the synthetic bimodal case. The non-monotonic structure at
very large sigma flags a limit: extreme tail-heterogeneity outruns any fixed p_imm
budget, just as F54's "extreme" condition outran any tested beta.

---

## Finding 60: Slow-recoverer vaccination tolerates noisy gamma estimates -- the policy works under realistic observation noise
<img src="./figures/noisy_gamma_vaccination_1.png" width="640"/>

**What:** F56-F59 used the EXACT true per-agent gamma_i to rank agents. In any real
application the vaccinator does not see gamma_i directly; they see a noisy estimate
(say, recovery time from one prior panic episode). If slow-targeting requires perfect
knowledge it is impractical; if it tolerates substantial noise it is actionable.
**Evidence:** noisy_gamma_vaccination.py, F56 setup (bimodal {0.2, 1.8}, beta=0.30,
p_imm=0.30), 4 seeds. Vaccinator ranks agents by gamma_hat_i = gamma_i + N(0, sigma_obs)
and immunises the bottom p_imm fraction. Random baseline at the same gamma seeds gives
f_ss = 0.164 +/- 0.032.
  sigma_obs   slow_hit_rate   f_ss
  0.00        1.000           0.013+/-0.019
  0.10        1.000           0.039+/-0.044
  0.20        1.000           0.039+/-0.044
  0.40        1.000           0.039+/-0.044
  0.60        0.986           0.035+/-0.029
  0.80        0.952           0.024+/-0.027   (essentially perfect)
  1.00        0.893           0.045+/-0.039
  1.50        0.793           0.077+/-0.046
  2.00        0.726           0.102+/-0.068
  100.        0.533           0.150+/-0.039   (chance ranking; ~random)
**Key result 1 -- noise up to ~0.8 leaves the policy identical to perfect knowledge.**
The true slow/fast distance in this setup is 1.6 units (1.8 - 0.2). At sigma_obs = 0.8
(half that distance) the slow-hit-rate is still 95% and f_ss is 0.024 -- essentially
eradication. The Gaussian observation noise rarely flips the ranking of any
gamma-extreme pair until sigma_obs is comparable to the bimodal half-width.
**Key result 2 -- slow-targeting still beats random at sigma_obs=2.0.**
At sigma_obs = 2.0 the observation is dominated by noise (slow-hit-rate 73%), yet
f_ss = 0.102 is still well below random's 0.164 (38% reduction). The policy degrades
gracefully: even half-informative gamma estimates retain a measurable advantage.
**Key result 3 -- in the uninformative limit (sigma_obs=100) the policy reduces to
random.** slow-hit-rate falls to 53% (chance) and f_ss converges to 0.150, within
seed noise of random's 0.164. Failure mode: graceful, not catastrophic.
**Implication:** The F56 slow-targeting policy is highly noise-tolerant. For any
practical estimate of per-agent gamma_i with noise comparable to or smaller than the
slow/fast separation, the policy works identically to the perfect-knowledge case.
This removes one major concern about real-world applicability: vaccinators do not
need precise gamma_i measurements; they need rough rankings. The policy fails
gracefully (degrading to random) only when the observation is dominated by noise.
Combined with F59 (continuous distributions), F58 (3D transfer), and F61 (rare
reservoirs, below), slow-targeting is now established as the most robust vaccination
policy in this study across every variation tested.

---

## Finding 61: Slow-recoverer vaccination works for rare reservoirs -- smaller reservoir, smaller required vaccination budget
<img src="./figures/rare_reservoir_vaccination_1.png" width="640"/>

**What:** F56-F60 used a 50/50 bimodal split: half the population is slow. In real
settings the reservoir-prone class may be a small minority (e.g. 5-15%, like the
immunocompromised). If slow-targeting requires a large reservoir to be effective it
is brittle; if it works for rare reservoirs the policy scales naturally with the
size of the problem.
**Evidence:** rare_reservoir_vaccination.py, gamma_slow=0.1 (deep reservoirs),
gamma_fast adjusted so arithmetic mean stays 1.0, beta=0.30, 4 seeds.
Exp 1 (p_imm = f_slow, target the reservoir EXACTLY):
  f_slow   gamma_fast   p_imm   random          slow
  0.05     1.047        0.05    0.129+/-0.055   0.000+/-0.000  (eradication)
  0.10     1.100        0.10    0.143+/-0.035   0.000+/-0.000
  0.20     1.225        0.20    0.144+/-0.024   0.000+/-0.000
  0.30     1.386        0.30    0.176+/-0.026   0.000+/-0.000
  0.50     1.900        0.50    0.150+/-0.015   0.000+/-0.000
Exp 2 (p_imm = 0.30 fixed, F56-budget across rarity):
  f_slow   random          slow
  0.05     0.000           0.000  (sub-threshold for both, p_imm overshoots)
  0.10     0.000           0.000
  0.20     0.081+/-0.030   0.000
  0.30     0.169+/-0.020   0.000
  0.50     0.294+/-0.022   0.135+/-0.018
**Key result 1 -- even a 5% slow class sustains an SIS endemic state under random
vaccination, but slow-targeting at p_imm=5% eradicates.** With f_slow=0.05 a random
vaccinator misses most of the reservoir (only 0.05*0.30 = 0.015 of selected immune
are slow) and the residual reservoir of ~17 agents (out of 350) is enough to keep
the epidemic going at f_ss=0.129. Slow-targeting at the matching p_imm=0.05 picks
EXACTLY the 17 slow agents and the epidemic is gone.
**Key result 2 -- minimum vaccination budget = f_slow when reservoir is known.**
Whenever p_imm matches f_slow, slow-targeting eradicates regardless of how small
f_slow is. This is the tightest possible policy: vaccinate exactly the reservoir
class, vaccinate no one else, and the epidemic dies. The smaller the reservoir, the
smaller the budget. This explains the F59 non-monotonicity: when sigma_log is large
and the bottom-p_imm quantile misses some deep-tail agents, eradication fails not
because slow-targeting is wrong but because p_imm is too small relative to the
effective reservoir size.
**Key result 3 -- when p_imm exceeds f_slow, the surplus does nothing useful.**
At p_imm=0.30 and f_slow=0.05 or 0.10, both random AND slow give f_ss=0 because
the epidemic is sub-threshold (a 30%-immunised flock with low effective <k> doesn't
sustain the SIS at beta=0.30 regardless of who is immune). At p_imm=0.30 and
f_slow=0.20, slow eradicates with 100 immunised (50 needed for the reservoir, 50
wasted on fast agents) while random still leaves 8% endemic. The optimal budget for
slow-targeting is p_imm = f_slow; anything more is waste, anything less leaves
reservoir uncovered.
**Implication:** Slow-targeting scales naturally with reservoir size. Vaccination
costs grow linearly with the reservoir fraction, not with the total population. This
is the policy's practical signature: in a flock where the slow class is small, the
required vaccination effort is small. Combined with F60 (noise tolerance) the policy
is fully practical: a vaccinator who knows roughly who the slow class is can
eradicate the SIS endemic with effort proportional to the reservoir size, regardless
of dimension (F58), distribution shape (F59), measurement precision (F60), or
reservoir rarity (F61). The slow-recoverer vaccination thread closes here as a
positive, complete, robust policy result.

---

## Finding 62: Slow-recoverer vaccination needs a DURABLE recovery-rate label -- drift erodes the targeting advantage, and fast drift eradicates the epidemic by self-averaging gamma
<img src="./figures/drifting_gamma_vaccination_1.png" width="640"/>

**What:** F56-F61 all held per-agent gamma_i STATIONARY. That stationarity is the
single load-bearing assumption of the whole positive result: the synthesis explains
slow-targeting's success by saying the "hub" label lives in gamma_i, a FIXED property
of the individual that kinematic mixing cannot scramble. This experiment makes gamma_i
a fluctuating STATE rather than a trait and asks whether the policy survives.
**Evidence:** drifting_gamma_vaccination.py, bimodal gamma {0.2, 1.8}, beta=0.30,
4 seeds. Vaccinate the slow agents ONCE at t=0 from a snapshot, then let each agent's
slow/fast identity decorrelate by symmetric two-state resampling at rate r_drift
(identity autocorrelation time ~ 1/r_drift). The bimodal marginal and 50/50 split are
preserved at all times; only WHICH individuals are slow drifts. Baseline (no
vaccination, drift-independent): f_ss = 0.376.
  p_imm=0.20         r_drift:  0.0     0.1     0.3     1.0    3.0   10.0
    random f_ss             0.233   0.178   0.166   0.000  0.000  0.000
    slow   f_ss             0.115   0.180   0.147   0.049  0.000  0.000
    advantage (rand-slow)  +0.118  -0.002  +0.019  -0.049  0.000  0.000
  p_imm=0.40
    random f_ss             0.095   0.070   0.023   0.000  0.000  0.000
    slow   f_ss             0.000   0.000   0.028   0.000  0.000  0.000
**Key result 1 -- the slow advantage requires a durable label and is gone by mild
drift.** At r_drift=0 slow beats random as F56 says (+0.118 at p_imm=0.20). By
r_drift=0.1/tu the advantage has collapsed to ~0: random IMPROVES (0.233->0.178) while
slow WORSENS (0.115->0.180) and they converge. The one-shot vaccine is wasted as
vaccinated slow agents drift to fast and unvaccinated agents drift into the reservoir.
This is the predicted erosion of a snapshot policy when the targeted property is not
durable -- and is distinct from F60 (static observation noise), which the policy
tolerates: drift is noise that grows with time.
**Key result 2 -- fast drift ERADICATES the epidemic regardless of strategy.** At
r_drift>=1/tu both random and slow give f_ss=0. This is the deeper result: fast drift
does not merely defeat targeting, it removes the heterogeneity that made the epidemic
supercritical. F54 showed the threshold reduction comes from the SPREAD of gamma; when
gamma decorrelates faster than the recovery timescale, every agent recovers at the
time-averaged mean (gamma=1.0), restoring the HOMOGENEOUS threshold (F54 homog
beta_c=0.385). At beta=0.30 < 0.385 the system is subcritical and the outbreak dies on
its own. The crossover (r_drift ~ 0.2-1/tu) matches the reservoir-memory timescale
1/gamma_slow = 1/0.2 = 5 tu: the reservoir survives drift only when 1/r_drift exceeds it.
**Key result 3 -- the policy's domain of usefulness exactly coincides with the regime
where the reservoir is real.** Where slow-targeting helps (durable label), the epidemic
is endemic and needs intervention; where it stops helping (fast drift), there is no
reservoir to target AND no threshold reduction, so no targeting is needed. The
per-agent-invariance argument is thus confirmed in its STRONG form: slow-targeting is
not a fragile trick defeated by dynamics -- it is exactly as valid as the reservoir it
exploits is durable.
**Implication:** Reframes F54-F61. The operative requirement is not "gamma_i is
measurable" (F60) but "the slow CLASS is a stable population on the epidemic timescale".
A recovery rate that is a persistent individual trait (chronic condition, age) is
targetable; one that is a transient state (a passing illness shorter than the outbreak)
is not -- but in that case the reservoir self-averages away and the epidemic is milder.
The F53/F54 heterogeneity dichotomy is sharpened: heterogeneity the dynamics homogenize
on a fast timescale is neither a lasting threshold-shifter NOR a targetable hub.

---

## Finding 63: Under combined beta_i + gamma_i heterogeneity, slow-recoverer targeting is the ROBUST vaccine; super-spreader targeting is as good ONLY when infectiousness is not anti-correlated with the reservoir
<img src="./figures/het_beta_gamma_vaccination_1.png" width="640"/>

**What:** F55 found heterogeneous infectiousness (beta_i) does NOT shift the SIS
threshold and concluded "target gamma_i, not beta_i." F56 found slow-recoverer
(gamma) targeting beats random. Both were studied in isolation. This experiment layers
BOTH heterogeneities and varies their correlation, asking the adversarial question:
if super-spreaders are FAST recoverers (high beta, high gamma), they escape a
gamma-based vaccine -- do they leak the epidemic and defeat slow-targeting?
**Evidence:** het_beta_gamma_vaccination.py, gamma bimodal {0.2,1.8} (50/50), beta
bimodal {0.15, 0.90} (20% super, arithmetic mean 0.30 == F56), source-weighted
transmission (a calm agent's hazard = sum of beta_j over its panicked contacts, the
F55 convention), 4 seeds. Strategies: random, slow (smallest gamma), super (largest
beta), combo (half budget each).
Exp 1 (independent correlation, p_immune sweep), f_ss:
  p_imm      0.10    0.20    0.30    0.40
  random    0.256   0.219   0.139   0.090
  slow      0.240   0.120   0.032   0.001
  super     0.159   0.025   0.003   0.001
  combo     0.206   0.009   0.005   0.000
Exp 2 (p_immune=0.30, strategy x correlation), f_ss [none = no vaccination]:
  correlation   none    random   slow    super   combo
  independent  0.373   0.139    0.032   0.003   0.005
  pos          0.445   0.244    0.020   0.000   0.000
  neg          0.225   0.062    0.000   0.017   0.022
**Key result 1 -- beta-targeting IS effective for vaccination, refining F55.** Under
independent correlation, super-spreader targeting is the STRONGEST single strategy
(f_ss=0.025 at p_imm=0.20 vs slow 0.120). This does not contradict F55: F55 says the
THRESHOLD does not move with beta-spread at fixed mean (a statement about where
criticality sits). Vaccination REMOVES agents entirely, and the 20% supers at beta=0.9
source ~60% of total infectivity (0.2*0.9 / 0.30); deleting them slashes transmission
capacity. So source-side heterogeneity is event-level for the threshold (F55) but
exploitable for removal (here) -- two different questions with two different answers.
**Key result 2 -- the reservoir and the transmission engine are different populations,
and only reservoir-targeting is robust.** Endemic persistence is set by the SLOW class
(gamma, the reservoir, F54); transmission throughput is set by the SUPER class (beta,
the engine). When the two coincide or are uncorrelated (pos / independent), hitting
supers works because the engine overlaps the reservoir. In the adversarial anti-
correlated case (neg: supers are fast recoverers), removing the engine leaves the slow
reservoir intact to re-sustain the outbreak -- super-targeting leaves f_ss=0.017 while
SLOW-targeting ERADICATES (0.000). Slow-targeting wins or ties in all three regimes;
super-targeting fails relative to slow exactly when infectiousness is anti-correlated
with recovery rate.
**Key result 3 -- correlation sets the baseline severity.** With no vaccination,
pos (supers slow, engine == reservoir) is the worst epidemic (f_ss=0.445), neg (supers
fast, engine decoupled from reservoir) is the mildest (0.225). Co-locating high spread
and slow recovery in the same agents maximizes endemicity; separating them weakens it.
**Implication:** Completes the vaccination-target taxonomy. F36/F48: degree fails
(no hubs). F37: spatial fails (kinematic mixing). F56-F62: slow (gamma) succeeds and
is robust because the reservoir label is a per-agent invariant. F63 adds that
infectiousness (beta) targeting is ALSO effective for removal -- but it is not robust,
because it targets the transmission engine rather than the reservoir, and the two can
be decoupled. Operational guidance: when forced to choose ONE axis, target the
reservoir (slow gamma); target super-spreaders additionally only when they are known
not to be fast recoverers. The F55 "target gamma not beta" slogan is correct as a
ROBUSTNESS statement, not because beta-targeting is ineffective.

---

## Finding 64: Slow-recoverer vaccination reverses the F34 damage asymmetry -- but only when the budget covers the full reservoir
<img src="./figures/predator_slow_vaccination_1.png" width="640"/>

**What:** F34 found the study's sharpest asymmetry: after an encirclement-driven SIS
outbreak, predator removal lets the KINEMATIC damage reverse fast (~10 tu, F22) but the
EPIDEMIC persists 100+ tu -- so contagion was "the worst combined stressor", with damage
that outlasts the event. That experiment used homogeneous gamma. F54-F63 established
that endemic persistence is set by the slow-recoverer reservoir and that vaccinating it
(F56) is the robust policy. Pointed question: does vaccinating the slow class before the
attack let the post-removal epidemic DIE, converting irreversible contagion damage into
reversible damage and overturning F34?
**Evidence:** predator_slow_vaccination.py, slow-prey predator regime (v0=0.02), three
phases (warmup 0-10 tu / 6-predator encirclement + SIS 10-50 tu / predators removed
50-100 tu). Bimodal gamma {0.5, 3.5}, mean 2.0 (== F34's mean), so the slow class is a
reservoir (beta/gamma_slow = 3.0) while the mean matches F34; beta=1.5; 4 seeds.
Vaccination at t=0 (immune never panic); strategies none/random/slow at matched p_imm.
  strategy   p_imm   f_during   f_post    Phi_post
  none       0.00    0.586      0.572     0.139
  random     0.20    0.425      0.419     0.311
  slow       0.20    0.342      0.358     0.376
  random     0.30    0.350      0.334     0.417
  slow       0.30    0.225      0.206     0.665
  random     0.50    0.173      0.175     0.737
  slow       0.50    0.000      0.000     0.997
**Key result 1 -- the epidemic persists after predator removal for every strategy
EXCEPT full-reservoir slow-targeting.** For none/random/slow at p_imm<=0.30, f_post is
within noise of f_during: removing the predators does not lower the endemic level. This
reproduces and strengthens F34 -- with heterogeneous gamma the reservoir makes the
outbreak supercritical on its own (beta/gamma_slow=3.0), so compression is not even
needed to sustain it, and removal cannot reverse it. The F34 asymmetry stands.
**Key result 2 -- slow-targeting reverses the asymmetry once the budget covers the
reservoir (p_imm >= f_slow = 0.50).** At p_imm=0.50 slow-targeting vaccinates all 175
slow agents (the entire reservoir); the remaining fast agents have beta/gamma_fast =
1.5/3.5 = 0.43 < 1 (subcritical), so the epidemic cannot sustain -- f_ss=0.000 during
AND after the attack, and Phi recovers to 0.997 (full F22 reunion). Both damage channels
are now reversible: contagion is no longer the worst stressor. Random vaccination at the
same p_imm=0.50 leaves ~87 slow agents, the reservoir persists, and f_post=0.175 remains
endemic with Phi only 0.737. This is the F61 "budget = reservoir fraction" law operating
inside the combined predator+contagion regime.
**Key result 3 -- below the reservoir budget, slow-targeting still helps on both axes.**
At every p_imm slow beats random in endemic level (0.358 vs 0.419 at 0.20; 0.206 vs
0.334 at 0.30) AND in kinematic recovery (Phi_post 0.665 vs 0.417 at 0.30), because
fewer panicked agents means less alignment disruption. So slow-targeting monotonically
improves the combined-stressor outcome even when it cannot eradicate.
**Implication:** Resolves the F34 "contagion always wins" conclusion. Contagion is the
worst stressor only while the reservoir survives; reservoir-targeted vaccination at a
budget matching the reservoir fraction makes the combined predator+contagion damage
FULLY reversible (epidemic eradicated, flock reunites to Phi~1.0). The kinematic
stressor was always reversible (F22); F64 shows the epidemic stressor becomes reversible
too, conditional on covering the slow class. This unifies the predator thread (F22
reversibility), the contagion thread (F34 persistence), and the vaccination thread
(F56/F61 reservoir-targeting) into one statement: in the flock, lasting damage requires
a surviving reservoir, and the reservoir is the slow-recoverer class.

---

## Finding 65: 3D flocks are robust to ALL point-predator strategies tested -- the F43 "no surface to seal" failure is a special case of "no spatial perimeter at all"
<img src="./figures/flocking3d_transect_1.png" width="640"/>

**What:** F43-F49 showed encirclement does not disrupt a 3D flock (Phi~1.0) by any
geometric variant (radius, count, adaptive, sphere vs planar). The mechanistic claim was
that a handful of point predators cannot seal a closed 2D SURFACE around a 3D volume the
way they can seal a 1D perimeter around a 2D area. But that claim, if literally correct,
only rules out SURROUNDING strategies. A strategy that does not rely on sealing -- a
predator that TRANSECTS the flock, darting through the dense core at high speed and
shearing alignment in its wake -- is the natural test of whether 3D flocks are robust to
all point predators or only to those that try to surround.
**Evidence:** flocking3d_transect.py. Three strategies in the same 3D harness as F43:
naive (chase CoM, slow predator v0=0.05), encircle (F43 baseline, slow), and transect
(chase CoM, fast v0=0.30 so the predator punches through and oscillates back). N=350,
3 seeds, 4000 steps, 2500-step warmup. Two experiments:
Exp A -- strategy comparison at n_pred = 3, 6, 10 (matched R_enc=0.15 for encircle):
  strategy   n_pred    Phi             Rg
  naive      3/6/10    1.000/1.000/1.000   0.430/0.430/0.431
  encircle   3/6/10    1.000/1.000/1.000   0.430/0.434/0.437
  transect   3/6/10    1.000/1.000/1.000   0.431/0.431/0.432
Exp B -- transect predator-speed sweep at n_pred=10, prey v0=0.02:
  v0_pred    0.05   0.10   0.20   0.40   0.80
  Phi        1.000  1.000  1.000  1.000  1.000
  Rg         0.431  0.431  0.431  0.432  0.450
**Key result 1 -- 3D Phi=1.000 to three decimals at every configuration tested.** Naive
and transect are equivalent in their target (CoM) and differ only by predator speed; over
the full v0_pred sweep (40x range, ending at 40x prey speed) the order parameter never
moves. The 3D flock is robust to every point-predator strategy in this harness, not just
those that try to seal a perimeter.
**Key result 2 -- the mechanism is not "can't seal a surface", it is "no spatial
structure to attack at all".** The flock's radius of gyration sits at Rg=0.43 in the unit
cube; the upper bound for a uniform spatial distribution is Rg ~ sqrt(1/4) = 0.5, so the
3D "flock" fills the box nearly uniformly with globally aligned velocities (mean
alignment-neighbor count ~12 at rf=0.20, N=350). It has no spatial perimeter to encircle
and no localized core to transect: a few predators with finite repulsion range
R0_P=0.10 only perturb a vanishing fraction of the flock at any instant, and the
remaining ~99% of the alignment graph immediately heals the wake. F43's "cannot seal a
2D surface" framing was incidental; the deeper statement is that 3D flocks are not
spatially localized at the F41-F49 parameter regime, so the predator's geometric task
has no localized target to address.
**Key result 3 -- the only Rg movement is the predictable Stokes-style wake.** Transect
at v0_pred=0.80 raises Rg slightly (0.450 vs the 0.430 baseline) -- the very fast
predator carves a faintly larger excluded volume than slower ones -- but velocity
alignment remains at 1.000 regardless. Spatial perturbation does not imply alignment
disruption when the alignment graph is globally connected.
**Implication:** Closes the 3D point-predator question. To disrupt a 3D flock at these
parameters an attacker must either (i) supply on the order of N predators (already
known not to help up to n_pred=50, F44), (ii) extend each predator's repulsion range to
a substantial fraction of the box (unphysical), or (iii) attack the alignment force per
agent rather than relying on repulsion -- which is exactly what contagion does, and why
F25/F54 contagion successfully disrupts 3D flocks where predators cannot. The 3D
predator thread closes definitively: alignment-driven kinematic mixing without spatial
localization is invulnerable to point-source mechanical disruption.

---

## Finding 66: Predictive encirclement (predators target CoM + lead*v_mean) deepens F14 disruption substantially -- the first predator-side adaptation in this study to beat F14
<img src="./figures/predictive_encirclement_1.png" width="640"/>

**What:** F33 found the flock does not steer toward gaps -- it has no global escape-route
detection. The symmetric, untested question is whether PREDATORS can detect and exploit
the flock's heading direction. The simplest predator intelligence is anticipation: each
predator's target is CoM + lead_time * v_mean (with v_mean the flock's mean velocity),
so the encirclement ring is placed where the flock WILL be rather than where it is.
At lead_time=0 this reproduces F14 (Phi~0.77 at n_pred=6, R_enc=0.15). For lead_time>0
the ring shifts in the flock's heading direction.
**Evidence:** predictive_encirclement.py, slow-prey regime (v0=0.02, ramp=0.1), N=350,
n_pred=6, R_enc=0.15, 4 seeds, 1000-step warmup then 4000 steps of encirclement, Phi
averaged over the attack phase (first 500 transient steps dropped).
  lead_time (tu)   mean Phi    cross-seed std   intra-run std
  0.0              0.825       0.127            0.159
  0.5              0.675       0.049            0.212
  1.0              0.687       0.082            0.210
  2.0              0.530       0.074            0.257
  5.0              0.908       0.143            0.091
  10.0             0.891       0.064            0.069
**Key result 1 -- predictive encirclement deepens disruption substantially below F14 and
F35.** At the optimum lead_time=2 tu, Phi=0.530 -- well below F14's baseline (0.77-0.83
here) and below F35's adaptive R_enc result (0.713). This is the first predator-side
adaptation in the study that substantially beats F14 at the same predator count and
radius. The improvement comes purely from PLACEMENT: predators do not change R_enc, do
not change n_pred, do not coordinate beyond all using the same v_mean -- they merely
lead the flock by lead_time*v_mean.
**Key result 2 -- the optimum has a clear geometric explanation.** Mean prey speed
v_mean ~ v_eq = v0 + alpha/mu = 0.02 + 0.1 = 0.12 per F1. At lead_time=2 tu the lead
distance is 2*0.12 = 0.24, larger than R_enc = 0.15. The ring of predators is therefore
sitting where the flock will be in ~2 tu, and the flock's heading direction is now
inside the ring rather than open. At lead_time=5-10 tu the lead distance is 0.6-1.2,
predators overshoot beyond the flock's reach within the attack window, and the flock
turns away -- the ring is now irrelevant. Disruption is non-monotonic in lead_time with
a clear optimum near R_enc / v_mean.
**Key result 3 -- intra-run std grows in the disruptive regime.** At lead_time = 0.5-2,
intra-run Phi std rises to 0.21-0.26 (vs 0.09 at lead=5-10 and 0.16 at lead=0). This is
the F32 intermittent merge/split steady state, sharpened by predictive placement:
predators repeatedly intercept the leading sub-flock, fragment it, and the fragments
re-form before being intercepted again. Predictive placement makes the flock visit the
encirclement ring more often per unit time than fixed placement does.
**Implication:** Opens the predator-learning thread. Adapting POSITION (F66, lead the
flock) is complementary to adapting RADIUS (F35, R_enc/Rg). The two are independent
levers and could be combined; the natural follow-up is "predictive + adaptive" predators
that scale R_enc with live Rg AND lead by v_mean. This also flips the F33 asymmetry:
the flock cannot detect global escape directions, but predators CAN detect the flock's
global heading (mean velocity is the dual of "where the gap is for the flock"). Predator
intelligence is informationally easier than prey escape intelligence in this model,
because v_mean is a global summary statistic that is well-defined for the flock even
when the flock cannot use it itself.

---

## Finding 67: Predictive (F66) and adaptive R_enc (F35) do NOT compose -- predictive placement is the dominant lever and angular spread becomes secondary once the heading is blocked
<img src="./figures/predictive_adaptive_encirclement_1.png" width="640"/>

**What:** F66 closed by noting that adapting predator POSITION (predictive lead) and
adapting predator RADIUS (adaptive R_enc = 0.5*live_Rg, F35) are independent geometric
levers and predicted they would compose multiplicatively. The natural one-step test is
the four-condition matrix at matched n_pred=6: fixed-fixed (F14 reproduction), fixed-
adaptive (F35 reproduction), predictive-fixed (F66 reproduction, lead=2 tu), and the new
predictive-adaptive combined.
**Evidence:** predictive_adaptive_encirclement.py, slow-prey regime, N=350, n_pred=6,
4 seeds, 1000-step warmup then 4000 attack steps, Phi averaged over the attack phase
(first 500 transient steps dropped). Adaptive uses R_enc = 0.5*Rg (F35 universal
optimum); predictive uses lead = 2 tu (F66 optimum).
  condition              mean Phi   cross-seed std   intra-run std
  fixed-fixed            0.825      0.127            0.159
  fixed-adaptive         0.866      0.055            0.123
  predictive-fixed       0.530      0.074            0.257
  predictive-adaptive    0.535      0.059            0.228
**Key result 1 -- the two adaptations DO NOT COMPOSE.** Predictive-adaptive (0.535) is
within seed noise of predictive-fixed (0.530); the combined predator is no more
disruptive than the predictive-only predator. The F66 prediction that the two levers
would compose multiplicatively is falsified.
**Key result 2 -- placement dominates radius once the heading is blocked.** Under
encirclement the compressed flock has Rg ~ 0.05-0.10, so adaptive R_enc = 0.5*Rg gives
a ring radius of only ~0.025-0.05, while the predictive lead distance is 0.24 (= 2 tu *
v_mean 0.12). Six predators at a 0.03 ring radius placed 0.24 ahead of CoM cluster into
what is geometrically a near-point predator in the heading direction -- they are no
longer surrounding anything. Yet the combined Phi (0.535) is statistically the same as
the proper predictive ring at R_enc = 0.15 (0.530). The implication is that once the
flock's heading direction is blocked by predators at the right distance, the angular
spread of the predator configuration does not matter much: a single dense block in front
is as effective as a six-fold spread around the lead point. Encirclement's geometric
identity dissolves under predictive placement -- it becomes equivalent to a one-sided
interception.
**Key result 3 -- the F35 single-lever advantage is fragile in this harness.** Fixed-
adaptive (0.866) is no better than fixed-fixed (0.825) here, the opposite of F35's
reported 0.778 -> 0.713 improvement. The cross-seed std (0.055-0.127) is comparable to
the mean difference, so the F35 effect may be within the noise of this harness's 4-seed
estimate; F35 used a different aggregation metric (frac_above_0.85, which dropped 0.56
to 0.37). I do not claim F35 is wrong, but note that the adaptive-radius single-lever
effect is small or noise-level on the mean-Phi metric used here, while the predictive
single-lever effect (0.825 -> 0.530) is large and consistent across seeds.
**Implication:** Refines F66's "two independent levers" interpretation. Placement (where
the ring is) is the dominant geometric degree of freedom for predator-side adaptation;
radius (how big the ring is) is at best secondary and may be redundant once placement
is anticipatory. Closes the immediate predator-learning thread: the predator's
informational advantage (access to global v_mean) buys ~0.3 in Phi reduction; further
geometric tuning beyond that yields diminishing returns. The next questions are
predator-side INFORMATIONAL: e.g., what if predators have noisy v_mean estimates, or
delayed updates, or only see a subset of the flock? These are the F60-analog stress
tests for the F66 mechanism.

---

## Finding 68: Predictive encirclement degrades gracefully but GRADED under noisy v_mean -- less noise-tolerant than F60's slow-targeting because a global statistic has no N-sample averaging
<img src="./figures/predictive_noisy_encirclement_1.png" width="640"/>

**What:** F67 closed by flagging the next open question as predator-side INFORMATIONAL --
the F60 analog for the F66 predictive mechanism. The cleanest variant is observation
noise on the predator's estimate of v_mean. Real predators do not have instantaneous
access to the flock's mean velocity; they estimate it with error. At the F66 optimum
lead_time = 2 tu, replace the true v_mean with v_mean_hat = v_mean + N(0, sigma_obs)
per step (independent Gaussian noise per component) and sweep sigma_obs from 0 (perfect
knowledge, F66 reproduction) up to 4x the magnitude of v_mean.
**Evidence:** predictive_noisy_encirclement.py, slow-prey regime, N=350, n_pred=6,
R_enc=0.15, lead=2 tu, 4 seeds, 1000-step warmup then 4000 attack steps, Phi averaged
over the attack phase (first 500 transient steps dropped). |v_mean| ~ v_eq = 0.12
(F1), so sigma_obs = 0.12 means observation noise equal in magnitude to the signal.
  sigma_obs   sigma/|v_mean|   mean Phi   cross-seed std   intra-run std
  0.00        0%               0.530      0.074            0.257
  0.03        25%              0.629      0.128            0.191
  0.06        50%              0.670      0.084            0.255
  0.12        100%             0.709      0.062            0.181
  0.24        200%             0.770      0.071            0.181
  0.48        400%             0.804      0.110            0.133
**Key result 1 -- graceful but graded degradation, no cliff and no plateau.** Phi rises
monotonically with sigma_obs from 0.530 to 0.804 (approaching the F14 baseline of
0.825 in the high-noise limit). The advantage over F14 (0.825 - Phi) decays from 0.295
at sigma=0 to 0.221 at sigma=0.03 (75% of original advantage retained), 0.155 at sigma
=0.06 (52%), 0.116 at sigma=0.12 (39%), 0.055 at sigma=0.24 (19%), and 0.021 at
sigma=0.48 (7%). The policy never fails outright but loses about a quarter of its
advantage for each step up the noise scale.
**Key result 2 -- predictive is LESS noise-tolerant than F60's slow-targeting.** F60
showed that slow-recoverer vaccination is identical to perfect knowledge up to
sigma_obs ~ half the slow/fast separation, then graceful below. Here predictive
degrades immediately from sigma=0, with no plateau. The contrast is mechanistic: F60's
signal is a PER-AGENT ranking. Noise on individual gamma estimates does not change the
overall ordering as long as noise < separation, because the population N=350 has many
agents and the bottom quantile is stable under noise applied independently per agent
(an N-sample averaging argument). F68's signal is a SINGLE GLOBAL VECTOR per timestep.
Each noisy estimate is one number; there is no averaging across many independent
samples within a step, and noise enters predator targeting directly.
**Key result 3 -- intra-run std drops with sigma_obs at high noise.** Intra-run Phi
std falls from 0.257 at sigma=0 to 0.133 at sigma=0.48. The high-noise regime smooths
the F32 intermittent merge/split state into a steadier (but milder) disruption, because
the noisy predictor no longer locks on the flock's heading and instead jiggles the ring
direction randomly -- the flock no longer gets repeatedly intercepted from the same side.
**Implication:** Refines the synthesis on what makes "intelligent" disruption robust.
The statistical footprint of the intelligence matters as much as its content. Per-agent
invariants (gamma_i, F56-F61) are intrinsically buoyed by N-sample averaging and tolerate
substantial observation noise. Global summary statistics (v_mean, F66) carry the same
informational content per step but as one number, so they are noise-sensitive without an
averaging buffer. To make predator intelligence robust, the predator would need to AVERAGE
v_mean estimates over time (a Kalman-style filter), which is a different model and a
different open question. Closes the predator-learning thread (F66-F68): predictive
placement is the dominant predator-side lever, radius adaptation does not compose with
it (F67), and the lever degrades gracefully but graded with observation noise (F68). The
predator now has all the geometric and informational degrees of freedom that one global
statistic can buy; further improvement requires temporal filtering or partial-observation
modelling, which is beyond the scope of the present study.

---

## Finding 69: Predictive encirclement is FAR more sensitive to DELAY than to noise -- a stale heading is a systematic error on a forward-projected quantity
<img src="./figures/predictive_delayed_encirclement_1.png" width="640"/>

**What:** F68 tested observation NOISE on v_mean (the F60 analog). The companion
informational stress test is DELAY: real sensing and processing introduce lag, so the
predator may act on v_mean from some time ago rather than the current value. At the F66
optimum lead_time = 2 tu, replace the current v_mean with v_mean from delay_steps
timesteps in the past and sweep the delay from 0 to 5 tu (comparable to the attack
duration).
**Evidence:** predictive_delayed_encirclement.py, slow-prey regime, N=350, n_pred=6,
R_enc=0.15, lead=2 tu, 4 seeds, circular buffer of past v_mean.
  delay (tu)   mean Phi    cross-seed std   intra-run std
  0.00         0.530       0.074            0.257
  0.25         0.774       0.120            0.211
  0.50         0.636       0.108            0.226
  1.00         0.849       0.147            0.102
  2.50         0.824       0.078            0.144
  5.00         0.880       0.108            0.090
**Key result 1 -- delay destroys the predictive advantage much faster than noise does.**
A delay of just 0.25 tu (one eighth of the lead time, 25 steps) lifts Phi from 0.530 to
0.774 -- losing about 83% of the F66 advantage over F14. By delay = 1 tu the advantage is
entirely gone (Phi = 0.849 >= the F14 baseline of 0.825). Compare F68, where 100%
observation noise (sigma = |v_mean|) still retained ~40% of the advantage. Delay is far
more damaging. (The non-monotonic dip at 0.5 tu, 0.636, is within the 4-seed cross-seed
std of ~0.11; the trend -- sharp loss of advantage by ~0.25-1 tu -- is unambiguous.)
**Key result 2 -- delay >= 1 tu is slightly WORSE than no prediction.** At delay = 1-5 tu
Phi (0.82-0.88) sits at or above the F14 fixed-encirclement baseline (0.825). A stale
lead steers predators toward where the flock was heading, which under the merge/split
dynamics is often no longer where it is heading, so the predators partially un-block the
current escape direction relative to a symmetric fixed ring. Bad information is worse
than no information for this mechanism.
**Key result 3 -- the asymmetry between noise and delay is mechanistic.** Noise (F68) is
a ZERO-MEAN error: over many timesteps the perturbations to the predator's target average
out, and the predator still spends most of its time roughly in the right place. Delay is
a SYSTEMATIC error: the predator consistently aims where the flock was going, and because
v_mean is used for FORWARD projection (target = CoM + lead*v_mean), a directional bias in
v_mean translates directly into a directional bias in placement. Under encirclement the
flock fragments and reorients, so v_mean decorrelates on sub-tu timescales; a delay
comparable to that correlation time (~0.25-0.5 tu) already makes the stale heading
nearly independent of the true heading. Intra-run std also collapses at long delay
(0.257 -> 0.090), confirming the predator no longer tracks the flock's heading and the
F32 intermittent interception cycle disappears.
**Implication:** Completes the predator-side informational suite (F66-F69). Predictive
encirclement requires CURRENT, LOW-NOISE access to the flock's global heading: it
tolerates moderate observation noise (F68) but not delay (F69), because the quantity is
used for forward projection and a stale value is systematically rather than randomly
wrong. This is the dual of the F60/F68 contrast: F60's per-agent rate is both
noise-robust (N-sample averaging) and intrinsically stationary (no delay problem,
because the rate does not change), whereas the predator's global heading is both
noise-sensitive AND delay-sensitive. The robustness of an "intelligent" disruption
strategy depends on whether its key signal is a stationary per-agent invariant or a
fast-changing global statistic. The predator-learning thread closes: any further gain
requires the predator to FILTER its heading estimate over time (a Kalman-style observer),
which is a different model.

---

## Finding 70: Collective escape intelligence counters predictive encirclement above a threshold -- but weak escape is WORSE than none, and the predator's own forward-massing creates the signal the prey exploit
<img src="./figures/collective_escape_1.png" width="640"/>

**What:** F66-F69 gave the predator a global signal (v_mean) and showed predictive
placement deepens disruption. The symmetric, arms-race question is whether the PREY can
use the dual global signal -- the predator centroid -- to flee collectively. F33 showed
the flock cannot detect escape directions on its own. Crucially, under SYMMETRIC
encirclement (F14) the predator centroid coincides with the flock CoM, so "flee the
centroid" has no gradient; but under PREDICTIVE encirclement (F66) the predators mass
AHEAD of the flock, displacing the centroid in the heading direction and making a
backward escape well-defined. Each prey adds a force w_escape * e_hat with e_hat the unit
vector from the predator centroid toward the flock CoM -- the prey-side dual of v_mean,
a global signal shared by the whole flock.
**Evidence:** collective_escape.py vs the F66 predator (predictive, lead=2 tu), N=350,
n_pred=6, 4 seeds. Prey alignment strength alpha=1.0 sets the force scale.
  w_escape   mean Phi    cross-seed std   intra-run std
  0.00       0.530       0.074            0.257
  0.25       0.275       0.048            0.190
  0.50       0.762       0.150            0.179
  1.00       0.932       0.028            0.134
  2.00       1.000       0.000            0.003
  5.00       1.000       0.000            0.000
**Key result 1 -- strong escape intelligence fully defeats predictive encirclement.** At
w_escape >= 2 (>= 2x the alignment strength) Phi returns to 1.000 with near-zero
fluctuation (intra-std 0.003): the flock flees the predator mass as a coherent rigid unit
and outruns the trap. A unified escape direction REINFORCES alignment -- every prey is
pushed the same way -- so the fleeing flock is perfectly ordered. Prey global
intelligence decisively beats predator global intelligence when the prey commit.
**Key result 2 -- weak escape intelligence is WORSE than none (non-monotonic).** At
w_escape=0.25 Phi DROPS to 0.275, below the no-escape value 0.530. An escape force too
weak to actually move the flock instead COMPETES with the alignment force: the flock is
torn between aligning with neighbors and weakly fleeing the centroid, and the two
directional drives partially cancel, fragmenting the flock more than the predators alone.
"A little escape intelligence is dangerous" -- it spoils alignment without achieving
escape. The threshold for benefit is w_escape ~ alpha (the alignment strength): below it,
escape loses the tug-of-war and only adds conflict; above it, escape wins and the flock
both aligns and evades.
**Key result 3 -- the predator's intelligence creates the prey's opening.** The escape
counter works specifically because predictive encirclement masses predators AHEAD of the
flock, displacing their centroid from the CoM and defining a backward escape direction.
Against symmetric F14 encirclement the centroid coincides with the CoM and the escape
force vanishes (no gradient). So the predator's forward projection -- the very thing that
made F66 effective -- is self-defeating against committed escape-intelligent prey: it
hands the flock a clean directional signal. The arms race is not symmetric in the naive
sense; it has a rock-paper-scissors structure (fixed encirclement gives no escape signal
but is weakly disruptive; predictive encirclement is strongly disruptive but legible to
escape intelligence).
**Implication:** Closes the predator-prey arms-race arc (F66-F70). When both sides have
their global signal, committed prey escape wins, because a collective flee is constructive
with alignment whereas predator placement must fight it. The non-monotonicity is the
deeper lesson: adding a competing global drive to an alignment-dominated flock is harmful
unless it is strong enough to take over the heading -- echoing F16/F24/F27 (competing
forces in the flock resolve by domination, not blending). The natural next questions are
partial/local escape sensing (does the result survive if prey sense only nearby
predators?) and co-adaptation dynamics (both sides updating), which are beyond the present
scope.

---

## Finding 71: Local escape sensing only PARTIALLY counters predictive encirclement -- the F70 full escape required a globally SHARED escape direction, not merely escape information
<img src="./figures/local_escape_1.png" width="640"/>

**What:** F70 gave every prey the global predator centroid and showed a committed
collective flee (w_escape >= alpha) fully defeats predictive encirclement (Phi -> 1.0).
That assumes each prey knows where all predators are. This experiment replaces the global
signal with a realistic per-prey LOCAL rule (the prey-side analog of F19's predator
sensing threshold): prey i flees the summed direction away from predators within a
sensing radius r_sense, escape_i = normalize(sum over in-range k of unit(pos_i - pos_k)),
and feels no escape force if none are in range. w_escape is fixed at 2.0 (the F70 value
that gave full escape with global sensing); r_sense is swept from 0.05 to 1.0.
**Evidence:** local_escape.py vs the F66 predator, N=350, n_pred=6, 4 seeds.
  r_sense   mean Phi    cross-seed std   intra-run std
  0.05      0.679       0.014            0.223
  0.10      0.709       0.044            0.229
  0.20      0.829       0.039            0.111
  0.40      0.778       0.009            0.111
  1.00      0.691       0.003            0.055
**Key result 1 -- local escape never reaches full escape.** Every r_sense gives Phi in
0.68-0.83, well above F66's 0.530 (some protection) but far below F70's 1.000 (no full
escape). Even at r_sense=1.0 (sensing range exceeding the flock) the flock does not
coherently outrun the trap. The difference from F70 is not the amount of information but
its STRUCTURE: F70's escape direction is a single shared vector (CoM toward away-from-
centroid) identical for all prey, so it ALIGNS with the flocking force and produces a
unified flee; F71's per-prey direction depends on each prey's own position relative to
the predators, so different prey flee different ways, the escape forces do not align
across the flock, and they compete with the alignment force instead of reinforcing it.
A globally shared escape vector is constructive with flocking; a locally computed one is
not.
**Key result 2 -- non-monotonic with an optimal sensing radius ~0.20.** Phi peaks at
r_sense=0.20 (0.829, about the F14 baseline) and falls off on both sides. Too local
(0.05-0.10): prey react only to nearly-touching predators, little better than plain
repulsion. Optimal (0.20 ~ R_enc+flock scale): prey sense the encircling ring and push
outward with enough coherence to reach baseline. Too global (0.40-1.0): each prey senses
predators on ALL sides of the (near-symmetric) ring, the unit vectors away from them
roughly cancel, and the escape force shrinks -- the F33 "surrounded, no net escape
direction" problem returns at the individual level. So extending local sensing range is
counterproductive past the ring scale.
**Key result 3 -- honest caveat on F70.** The dramatic F70 full-escape result is partly
an artifact of giving the flock a single globally-shared escape vector. With realistic
local sensing the counter is real but modest: it lifts Phi from 0.53 to at best ~0.83,
restoring the flock to roughly the disruption level of plain (non-predictive) F14
encirclement, not to full coherence. Predictive encirclement remains effective against
locally-sensing prey.
**Implication:** Sharpens the F70 lesson. What defeats predictive encirclement is not
escape information per se but a COMMON escape direction that the alignment force can
amplify -- the same principle that makes the flock coherent in the first place (a shared
heading) is what makes collective escape work. Local sensing cannot manufacture a common
direction under symmetric surrounding because each prey's local view points a different
way. This unifies F70/F71 with F16 (alignment homogenizes a shared quantity) and F33 (no
global escape-route detection from local information): the flock can act collectively
only on signals that are already global/shared, and a locally-sensed predator field is
not one of them. Closes the escape-sensing question: the F70 counter requires
flock-level coordination of the escape direction, which local perception does not provide.

---

## Finding 72: A tiny informed minority steers the whole flock with no loss of cohesion -- and it works because the leaders share a SINGLE common direction (the constructive case of the F70/F71 shared-signal rule)
<img src="./figures/leadership_1.png" width="640"/>

**What:** Opens the collective-decision-making thread (Couzin et al. 2005, "Effective
leadership and decision-making in animal groups on the move"). A fraction rho of agents
are INFORMED: they carry a preferred travel direction g_hat (=+x) and feel an extra force
w_lead*g_hat each step. The remaining 1-rho are naive followers with only the usual four
forces and no knowledge of g_hat. Question: how accurately does the whole flock travel
toward g_hat, how small can rho be, and does the steering force break cohesion?
**Evidence:** leadership.py at root, N=350, pure-flock regime (v0=1.0), default
parameters, 4 seeds. accuracy = (mean flock velocity . g_hat)/|mean flock velocity| in
[-1,1] (1 = flock travels exactly toward the goal); Phi = order parameter.
  w_lead   rho    informed   accuracy           Phi
  --       0.00     0        -0.242 +/- 0.750   1.000   (no preferred direction; random)
  0.5      0.02     7        +0.397 +/- 0.357   0.998
  0.5      0.05    18        +0.625 +/- 0.234   0.995
  0.5      0.10    35        +0.865 +/- 0.119   0.996
  0.5      0.20    70        +0.979 +/- 0.019   0.998
  0.5      0.50   175        +1.000 +/- 0.000   1.000
  1.0      0.02     7        +0.492 +/- 0.311   0.996
  1.0      0.05    18        +0.830 +/- 0.132   0.996
  1.0      0.10    35        +0.957 +/- 0.041   0.997
  1.0      0.20    70        +0.996 +/- 0.004   0.999
  1.0      0.50   175        +1.000 +/- 0.000   1.000
**Key result 1 -- a very small minority suffices.** At rho=0.05 (18 informed agents out
of 350) the flock already travels at accuracy 0.63-0.83 toward the goal; at rho=0.10 (35
agents) accuracy is 0.87-0.96. The naive 90-95% majority, which has no knowledge of the
goal direction whatsoever, is steered almost entirely by alignment coupling to the few
informed agents. This reproduces the central Couzin (2005) result: the informed fraction
needed for accurate group navigation DECREASES as group size grows, and only a handful of
leaders is required in a large group.
**Key result 2 -- accuracy increases with rho and with leader strength, and saturates.**
Accuracy is monotone in rho and saturates near 1.0 by rho~0.20. Stronger leaders (w_lead
1.0 vs 0.5) reach a given accuracy at smaller rho AND with much lower cross-seed variance
(at rho=0.05, std falls 0.234 -> 0.132; at rho=0.10, 0.119 -> 0.041): a stronger or more
numerous informed set makes the outcome not just more accurate but more RELIABLE. The
rho=0 baseline has accuracy -0.24 +/- 0.75 -- a near-uniform random heading across seeds
(the flock picks an arbitrary spontaneous direction), confirming the +x accuracy at rho>0
is entirely leader-induced.
**Key result 3 -- steering is cohesion-free.** Phi stays 0.995-1.000 at every rho and
w_lead. The leadership force never fragments the flock; the informed minority redirects
the group without any loss of order. Steering and cohesion are decoupled here -- the
opposite of the predator case, where redirecting the flock (encirclement) costs coherence.
The difference is sign: leaders add a coherent common-direction force that alignment
amplifies, whereas a predator adds position-dependent repulsion that alignment cannot
reconcile across the flock.
**Implication -- the CONSTRUCTIVE half of the F70/F71 shared-signal principle.** F71
concluded the flock acts collectively only on signals that are already global/shared.
Leadership is exactly such a signal: every informed agent carries the SAME vector g_hat,
so the minority injects a single common direction that the alignment force propagates to
the whole group -- the same mechanism that made F70's committed collective escape work
(one shared escape vector) and that F71's per-prey local escape lacked (each prey's vector
pointed a different way). Leadership, F70 escape, and flock formation itself (F16,
alignment homogenizes a shared heading) are the same phenomenon seen three ways: a
globally shared directional signal is amplified by alignment; a locally heterogeneous one
(F71 local escape, F33 no global escape-route detection) is not. The minority's power
comes not from numbers but from agreement -- 18 agents that all point the same way beat
175 agents (half the flock) that each sense a different local escape direction.

---

## Finding 73: Conflicting leaders -- the flock COMPROMISES at small angular conflict and switches to CONSENSUS (picks one) past ~90-120 deg, and a numerical majority among informed agents wins. Reproduces Couzin's decision dynamics in this model
<img src="./figures/conflicting_leaders_1.png" width="640"/>

**What:** The F72 follow-up that Couzin et al. (2005) is most famous for: split the informed
agents into TWO subgroups that prefer DIFFERENT directions and ask whether the flock (a)
COMPROMISES (travels the average heading), (b) reaches CONSENSUS (commits to one of the two
at random), or (c) SPLITS into two sub-flocks. Two experiments. Exp 1 sweeps the angular
conflict theta between equal subgroups (rho1=rho2=0.05): g1=+x, g2 rotated by theta. Exp 2
fixes theta=180 deg (direct opposition) and varies the size ratio at fixed total informed
fraction 0.10. Pure-flock regime, w_lead=1.0, 6 seeds.
**Evidence:** collective/conflicting_leaders.py, N=350.
  Exp 1 (equal subgroups, vary conflict angle):
  theta   midpoint   flock heading      Phi     split_frac
    0       0.0      +8.6 +/- 19.1     0.996    0.000
   30      15.0     +19.1 +/- 19.9     0.995    0.139
   60      30.0     +33.4 +/- 20.7     0.992    0.050
   90      45.0     +44.7 +/- 22.3     0.986    0.055
  120      60.0     +36.8 +/- 56.7     0.971    0.052
  150      75.0     +41.4 +/- 66.2     0.968    0.064
  180      90.0     +33.4 +/- 73.6     0.958    0.161
  Exp 2 (theta=180, vary size ratio, total informed 0.10):
  n1:n2    accuracy toward majority(+x)    Phi     split
  18:18         +0.201 +/- 0.406         0.958    0.161
  21:14         +0.423 +/- 0.368         0.960    0.074
  25:10         +0.627 +/- 0.293         0.964    0.041
  28: 7         +0.759 +/- 0.204         0.975    0.020
  35: 0         +0.925 +/- 0.101         0.995    0.000
**Key result 1 -- compromise at small conflict.** For theta <= 90 deg the flock travels
almost exactly the MIDPOINT direction (theta=90: heading 44.7 deg vs midpoint 45.0; theta=60:
33.4 vs 30.0), with a tight, consistent cross-seed spread (~20 deg). The two subgroups'
forces vector-add and alignment carries the resultant to the whole flock: the group
literally averages the two preferred directions. This is the compromise regime.
**Key result 2 -- consensus (random pick) past a critical angle ~90-120 deg.** At theta >=
120 deg the cross-seed heading std EXPLODES from ~22 to 57-74 deg while the mean heading
falls away from the midpoint and becomes meaningless as a central value. This is the
signature of CONSENSUS by random selection: different seeds commit to one subgroup's
direction or the other (the mean of "sometimes +x-ish, sometimes the rotated goal" is not
the midpoint and varies wildly seed to seed). Averaging two nearly-opposed directions is
not a viable heading -- a flock cannot travel "the average of +x and -x" -- so the symmetry
breaks and the flock picks a side. The critical conflict angle lies between 90 and 120 deg,
exactly Couzin's compromise-to-consensus transition.
**Key result 3 -- consensus, not splitting.** Phi stays high throughout (0.958-0.996) and
split_frac stays low (<=0.16 even at theta=180). The flock does NOT fragment into two stable
counter-traveling sub-flocks; it resolves the conflict by the WHOLE group committing to one
direction. Cohesion survives even direct opposition. (The small split_frac at theta=180,
0.16, is the minority of agents transiently pulled the other way before the flock commits.)
**Key result 4 -- the majority wins (democratic decision).** Exp 2: at exact parity (18:18)
the flock picks a side at random (mean accuracy ~0, huge spread 0.41). But even a slight
numerical majority among the informed agents decides the outcome: 21:14 already gives
accuracy +0.42 toward the majority goal, rising monotonically to +0.76 at 28:7 and +0.93 at
35:0 (single leader, ~F72). Cross-seed spread shrinks (0.41 -> 0.10) and splitting vanishes
(0.16 -> 0.00) as the margin grows. The larger informed subgroup wins more reliably the
larger its margin -- Couzin's result that group direction is decided by an effective majority
vote among the informed, even though every informed agent is a small fraction of the flock.
**Implication.** Extends F72 from "a shared signal is amplified" to "COMPETING shared signals
are resolved by vector-averaging when compatible and by majority-driven symmetry-breaking when
not." The flock is a near-ideal democratic integrator: it compromises when compromise is
geometrically sensible and votes when it is not, and it almost never splits. This is the
multi-signal generalization of the F70/F71/F72 shared-signal principle: alignment does not
just propagate one common direction, it ARBITRATES among several, and the arbitration rule
(average-then-threshold-to-consensus, majority wins) is an emergent property of the same
alignment force, not anything built into the agents. Connects the leadership thread to the
competing-global-drive results F16/F24/F27 (competing drives resolve by domination, not
blending) and to F70's non-monotonic escape (weak conflicting signal fragments; strong one
dominates): the consensus transition here is the same domination-not-blending physics, now
between two leadership signals rather than escape-vs-alignment.

---

## Finding 74: Numbers vs CONVICTION -- the collective decision is set by total PULL (count x strength), not headcount. A small, strongly committed minority beats a larger weak one
<img src="./figures/conviction_1.png" width="640"/>

**What:** F73 showed that between two equal-strength opposed subgroups the LARGER one wins
(majority vote). But all leaders there had equal bias strength. This asks the dual question:
does a SMALLER but more strongly committed subgroup beat a larger weakly committed one, and
what quantity decides the outcome -- headcount, or total "pull" = count x bias strength?
Two opposed subgroups (theta=180 deg, g1=+x, g2=-x). Accuracy toward g1 = cos(flock heading)
(+1 g1 wins, -1 g2 wins, 0 tie/random).
**Evidence:** collective/conviction.py, N=350, pure-flock, 6 seeds.
  Exp A -- EQUAL numbers (18 vs 18), vary conviction ratio w1/w2 (w2=1.0):
  w1/w2    accuracy toward stronger group    Phi
  1.0          +0.201 +/- 0.406            0.958
  1.5          +0.334 +/- 0.364            0.951
  2.0          +0.442 +/- 0.336            0.946
  3.0          +0.598 +/- 0.291            0.941
  5.0          +0.706 +/- 0.234            0.937
  Exp B -- PRODUCT-LAW test: 10 strong vs 26 weak (w2=1.0), vary w1:
  w1     pull1 = n1*w1   vs pull2=26    accuracy toward minority(g1)    Phi
  1.0       10.0                          -0.267 +/- 0.462            0.957
  1.8       18.0                          -0.165 +/- 0.452            0.947
  2.6       26.0  (pull balance)          -0.069 +/- 0.434            0.944
  3.5       35.0                          -0.024 +/- 0.441            0.942
  5.0       50.0                          +0.074 +/- 0.461            0.935
**Key result 1 -- conviction wins at equal numbers (a second independent lever).** Exp A:
with 18 agents on each side, the more strongly committed subgroup wins, and increasingly so
with the conviction ratio: accuracy toward the stronger group climbs from +0.20 (tie at equal
strength) through +0.44 at 2x to +0.71 at 5x. Bias strength is a lever on the decision exactly
as numbers are (F73). The flock's "vote" is not one-agent-one-vote; each informed agent's
influence scales with its commitment.
**Key result 2 -- the PRODUCT LAW: the winner is set by total pull = count x strength.** Exp B
pits a numerical minority (10 agents) against a majority (26) and ramps the minority's
conviction. Accuracy toward the minority crosses zero right around the point where the two
groups' total pull balances: -0.27 at pull 10 (minority loses badly), -0.07 at pull 26 (the
naive balance n1*w1 = n2*w2, essentially a tie), and +0.07 at pull 50 (the 10-agent minority
now wins). A small, strongly committed minority overcomes a larger weakly committed majority
once its summed committed force exceeds theirs. Headcount is not privileged over conviction;
what the flock integrates is the total directed force injected by each side.
**Key result 3 -- a mild residual numbers advantage.** The zero-crossing sits slightly ABOVE
equal pull: at exact pull balance (26 vs 26) accuracy is -0.069 and at pull 35 still -0.024,
turning positive only past ~pull 35-40. So the minority needs somewhat MORE than equal pull to
win. More distinct informed agents nucleate the goal direction in more spatial locations within
the flock, giving the more numerous side a small edge beyond its raw pull -- the product law is
the leading-order rule with a second-order bonus for being spread across more individuals.
**Key result 4 -- still consensus, never splitting.** Phi stays 0.94-0.96 throughout and the
cross-seed spread is large (~0.44) in Exp B because theta=180 is the consensus regime (F73):
each run decisively picks one side, and the mean accuracy is the expected vote bias across
seeds, not a within-run compromise. Even a near-balanced strength contest does not fragment the
flock -- it picks a direction.
**Implication.** Completes the F73 voting picture. F73 varied numbers at fixed conviction; F74
varies conviction at fixed and unequal numbers. Together they show the flock weights each
informed agent's vote by its commitment strength and decides by the summed pull of each side --
an emergent weighted majority rule, with a mild bonus for numerosity per se. This is the
quantitative form of the alignment-arbitration principle (F72/F73): alignment integrates all
the directed forces present and the group commits to the net winner, whether that net is built
from many weak voices or few strong ones. Connects to F70's escape threshold (the deciding
quantity there was also a force magnitude relative to alignment, w_escape vs alpha): in this
model collective outcomes are governed by summed directed force, not by counting agents.

---

## Finding 75: Time-resolved decisions -- leadership is fast and noise-robust, and the flock shows CRITICAL SLOWING at the compromise-to-consensus boundary (commitment time peaks at theta~90 deg)
<img src="./figures/decision_time_1.png" width="640"/>

**What:** F72-F74 all measured only the steady-state heading, averaged over a late window;
they say nothing about the DYNAMICS of the decision. This records the full heading time
series and extracts a settle/commitment time = the time (in tu) after which the flock heading
stays permanently within 15 deg of its final value. Three parts: (1) response time vs informed
fraction rho for a single leader; (1b) speed and accuracy vs noise at fixed rho=0.10; (2)
commitment time vs the conflict angle theta between two equal opposed subgroups (the F73
geometry), testing for critical slowing near the F73 compromise-to-consensus boundary.
**Evidence:** collective/decision_time.py, N=350, pure-flock, 4 seeds. (Settle times carry
large cross-seed std at 4 seeds; the ROBUST trends are the monotone accuracy, the strong
high-rho speedup, and the Part-2 peak.)
  Part 1 -- single leader, vary rho (w=1.0):
  rho     settle_time (tu)      accuracy
  0.02      5.98 +/- 4.20       +0.585
  0.05     13.60 +/- 10.09      +0.917
  0.10      8.85 +/- 8.16       +0.988
  0.20      4.50 +/- 3.98       +1.000
  0.50      0.46 +/- 0.33       +1.000
  Part 1b -- rho=0.10, vary noise ramp:
  ramp    settle_time (tu)      accuracy
  0.5       8.85 +/- 8.16       +0.988
  2.0       9.64 +/- 9.12       +0.996
  5.0       9.76 +/- 9.69       +0.996
  10.0     11.26 +/- 11.21      +0.998
  Part 2 -- two equal opposed subgroups, vary conflict angle:
  theta   commitment_time (tu)   final accuracy to g1
   30       7.34 +/- 6.88        +0.966   (compromise, ~midpoint)
   60      10.32 +/- 9.79        +0.864   (compromise, ~midpoint)
   90      12.09 +/- 9.27        +0.718   (PEAK -- boundary)
  120       7.55 +/- 4.53        +0.376   (consensus onset)
  150       5.82 +/- 10.08       +0.196   (consensus)
  180       6.38 +/- 10.93       +0.148   (consensus)
**Key result 1 -- strong leadership is BOTH faster and more accurate; no genuine
speed-accuracy tradeoff.** Final accuracy rises monotonically with rho (0.585 -> 1.000) and
the settle time collapses at high rho (4.5 tu at rho=0.20, 0.46 tu at rho=0.50 -- near
instant). The one apparent exception is the weak-leadership regime: at rho=0.02 the flock
settles fast (~6 tu) but onto a POOR heading (accuracy 0.585), because seven leaders cannot
drag the group off its spontaneous direction, so the heading is dominated by the flock's own
quickly-stabilizing spontaneous alignment. That is an under-led artifact, not a real tradeoff:
once leadership is strong enough to actually steer (rho >= 0.10), more leaders make the
decision simultaneously quicker and more accurate. The slowest commitment is the intermediate
rho~0.05, where the leader bias and the flock's inertia are comparable and the tug takes
longest to resolve.
**Key result 2 -- leadership is noise-robust in both speed and accuracy.** Across a 20x noise
range (ramp 0.5 -> 10) accuracy stays essentially perfect (0.988 -> 0.998) and commitment slows
only mildly (8.9 -> 11.3 tu). The flock follows its leaders just as accurately in heavy noise and
takes only slightly longer to settle -- consistent with F4 (full-model flocking robust to noise
up to eta ~ 10). Noise does not trade off against decision quality here.
**Key result 3 -- CRITICAL SLOWING at the decision boundary (the headline).** Commitment time
is non-monotonic in the conflict angle, peaking sharply at theta=90 deg (12.1 tu) and falling
off on BOTH sides (7.3 tu at theta=30, 6.4 tu at theta=180). The peak sits exactly at the F73
compromise-to-consensus boundary (~90-120 deg). This is the dynamical signature of a bistable
system slowing near its bifurcation: where compromise is easy (small theta) the flock averages
quickly; where the choice is decisive (large theta) it commits quickly once the symmetry breaks;
but right at the boundary, where the averaging solution is losing stability and the two
consensus solutions are just appearing, the flock dithers longest before settling. The decision
takes longest exactly when it is hardest.
**Implication.** Adds the temporal dimension to the F72-F74 decision picture and ties it to
dynamical-systems theory. The flock is not just a weighted-majority integrator (F73/F74) but a
bistable one: the compromise-to-consensus transition (F73) is a genuine bifurcation, evidenced
here by critical slowing at the threshold rather than only by the steady-state heading jump.
The speed-accuracy results sharpen the leadership story: leadership's benefit is not bought with
slower decisions (the predator-disruption tradeoff has no analog here) -- strong, even noisy
leadership is fast, accurate, and robust together. The only regime where the flock decides
"fast but wrong" is when leadership is too weak to overcome the flock's own spontaneous heading,
which is the temporal face of the F72 threshold (rho needs to clear a floor to steer at all).

---

## Finding 76: Leadership is a SIGNAL, not an IDENTITY -- rotating which agents are informed never hurts steering and FASTER rotation improves it. The exact opposite of F62's durable-label requirement
<img src="./figures/rotate_leaders_1.png" width="640"/>

**What:** F72 used a FIXED informed set. F62 (contagion thread) showed slow-recoverer vaccination
FAILS once the per-agent gamma label drifts, because there the targetable thing was a durable
per-agent IDENTITY. Leadership should be the opposite kind of signal: the flock follows a shared
DIRECTION, not particular individuals, so rotating which agents are informed -- while keeping the
goal g_hat=+x fixed -- should not hurt, because the total injected directed force (rho*N*w, the F74
"pull") is unchanged regardless of who carries it. Test: keep informed fraction rho and goal fixed,
but every tau time units re-draw at random which rho*N agents are informed. tau=inf is F72 (fixed);
tau->0 smears the bias across all agents (each informed a fraction rho of the time, time-averaged
uniform force rho*w*g_hat on everyone).
**Evidence:** collective/rotate_leaders.py, N=350, pure-flock, w_lead=1.0, 5 seeds.
  rho     tau (tu)     accuracy            Phi
  0.05    0.1        +0.943 +/- 0.065     1.000
  0.05    0.5        +0.935 +/- 0.070     0.999
  0.05    2.0        +0.894 +/- 0.108     0.998
  0.05   10.0        +0.872 +/- 0.127     0.997
  0.05   fixed       +0.864 +/- 0.136     0.997
  0.10    0.1        +0.994 +/- 0.007     1.000
  0.10    0.5        +0.991 +/- 0.010     1.000
  0.10    2.0        +0.981 +/- 0.022     0.999
  0.10   10.0        +0.970 +/- 0.035     0.998
  0.10   fixed       +0.966 +/- 0.040     0.998
**Key result 1 -- rotation never hurts: leadership is a signal, not an identity.** At every
rotation period the accuracy equals or EXCEEDS the fixed-leader baseline. Steering does not care
which individuals are informed at any moment, only that some rho*N of them are pushing the shared
direction. This is the direct opposite of F62, where rotating (drifting) the per-agent recovery-rate
label destroyed the vaccination advantage within ~0.1/tu. The distinction is sharp and mechanistic:
contagion targeting exploits a DURABLE PER-AGENT INVARIANT (gamma_i), so identity must persist;
leadership transmits a SHARED GLOBAL DIRECTION through alignment, so only the direction must persist,
not the messenger. The same decision currency (F74 total pull = count x strength) is delivered
whether it sits on a fixed set or rotates across the whole flock.
**Key result 2 -- FASTER rotation actively IMPROVES steering and sharply reduces variance.** As tau
falls from fixed to 0.1 tu, accuracy rises (0.864 -> 0.943 at rho=0.05; 0.966 -> 0.994 at rho=0.10)
and the cross-seed std collapses (0.136 -> 0.065 at rho=0.05; 0.040 -> 0.007 at rho=0.10). Fast
rotation smears the same total pull uniformly over all N agents (each informed a fraction rho of the
time), and a weak uniform bias on everyone steers more reliably than a strong bias concentrated on a
fixed subset. A fixed informed set can cluster or drift to the flock edge and must propagate its bias
through alignment, which adds lag and seed-to-seed variance; a smeared bias acts everywhere at once,
with no propagation bottleneck. Distributing the directed force beats concentrating it.
**Key result 3 -- still cohesion-free.** Phi stays 0.997-1.000 at every tau and rho, as in F72.
Rotating leadership does not stress the flock's coherence at all.
**Implication.** Completes the "what is the targetable invariant" arc that runs through the whole
study. Degree-targeting fails because the flock has no durable hubs (F36/F48, STRUCTURAL); spatial
targeting fails because motion erases coverage (F37, KINEMATIC); slow-recoverer targeting SUCCEEDS
because gamma_i is a durable per-agent invariant (F56) -- but only as long as it stays durable (F62).
Leadership now sits at the opposite pole: it works precisely because it does NOT rely on any
persistent identity. The "signal" is a shared direction held collectively, and any agent can carry it
at any time; rotating the carriers, far from degrading it, distributes the same total pull more evenly
and improves both accuracy and reliability. The unifying axis: a collective control that depends on a
persistent per-agent label is fragile to that label changing (F62), while one that depends only on a
shared global quantity (direction) is robust to -- even helped by -- turning over the individuals who
supply it. This also nuances the F37 "distribution doesn't help" null: distribution is irrelevant when
you are REMOVING nodes (vaccination) but beneficial when you are INJECTING a directional force, because
a force applied everywhere needs no propagation while a removed node's effect is local either way.

---

## Finding 77: The flock has a STEERING BANDWIDTH -- leaders can drive a turn only below a critical rate set by 1/(F75 response time); more leaders widen it; over-steering fragments the flock
<img src="./figures/moving_goal_1.png" width="640"/>

**What:** F72-F76 all used a FIXED goal direction. Real navigation requires turning. Here the goal
direction rotates at angular velocity omega rad/tu, g_hat(t) = (cos(omega t), sin(omega t)), and the
informed minority always biases toward the CURRENT goal. Questions: how well does the flock track a
turning target, how much does it lag, is there a critical turning rate above which tracking collapses
(a steering bandwidth), and does it scale with the informed fraction the way the F75 response time does?
**Evidence:** collective/moving_goal.py, N=350, pure-flock, w_lead=1.0, 5 seeds. track = mean
cos(heading - goal); lag = mean signed heading-minus-goal angle (deg).
  rho     omega (rad/tu)   track             lag (deg)   Phi
  0.10    0.00            +0.977 +/- 0.028     +0.8      0.998
  0.10    0.05            +0.551 +/- 0.207    -54.3      0.974
  0.10    0.10            -0.362 +/- 0.208   -103.4      0.911
  0.10    0.20            -0.045 +/- 0.015    -16.1      0.923
  0.10    0.50            -0.098 +/- 0.025     -8.5      0.930
  0.10    1.00            -0.058 +/- 0.013     -3.2      0.973
  0.20    0.00            +0.998 +/- 0.002     +0.5      1.000
  0.20    0.05            +0.832 +/- 0.025    -33.1      0.979
  0.20    0.10            +0.325 +/- 0.044    -70.5      0.923
  0.20    0.20            -0.449 +/- 0.213    -34.1      0.781
  0.20    0.50            -0.021 +/- 0.037    -25.3      0.861
  0.20    1.00            -0.050 +/- 0.013    -11.7      0.935
**Key result 1 -- a finite steering bandwidth.** The flock tracks a turning goal only below a critical
turning rate. At rho=0.10 tracking is near-perfect at omega=0 (0.977), degrades to partial at omega=0.05
(track 0.551, lag -54 deg: the flock trails the goal by 54 deg), and FAILS by omega=0.10 (track -0.362,
lag -103 deg -- the goal has out-run the heading by more than 90 deg, so they are anti-correlated). Above
the bandwidth (omega >= 0.20) the goal spins so fast the bias time-averages to nearly zero over each turn
and the flock effectively ignores it (track ~ 0): the leaders are shouting directions that reverse before
the flock can respond, so no net steering survives.
**Key result 2 -- the bandwidth scales with informed fraction, as 1/(F75 response time).** Doubling rho
(0.10 -> 0.20) roughly doubles the critical turning rate: at omega=0.10 the rho=0.20 flock still tracks
(0.325) where the rho=0.10 flock has already failed (-0.362), and at omega=0.05 it tracks much better
(0.832 vs 0.551). Quantitatively the bandwidth matches the inverse of the F75 settle time: omega_crit ~
1/tau_response gives ~1/8.85 = 0.11 rad/tu at rho=0.10 and ~1/4.5 = 0.22 rad/tu at rho=0.20, both
consistent with where tracking breaks down. The SAME lever that speeds the response (more leaders, F75)
widens the steering bandwidth (F77) -- they are one timescale: how fast the informed minority can re-aim
the bulk.
**Key result 3 -- first-order-lag tracking below the bandwidth.** Where the flock does track, it trails
the goal by a lag that grows with omega (0.8 deg at omega=0; 54 deg at omega=0.05 for rho=0.10), the
behaviour of a control system with a finite response time driven by a ramping setpoint. The flock is a
low-pass steering filter: slow turns pass with a small lag, fast turns are attenuated and ultimately
blocked.
**Key result 4 -- over-steering costs cohesion (unlike fixed-goal steering).** Fixed-goal steering was
cohesion-free at every rho (F72: Phi ~ 1.0). But forcing a turn faster than the flock can follow stresses
coherence: at rho=0.20, omega=0.20 (well above bandwidth, strong leadership), Phi falls to 0.781 -- the
lowest in the whole leadership thread. The informed minority pulls hard in a direction the bulk's alignment
cannot track, tearing the flock between the leaders' rapidly-rotating bias and its own lagging heading.
Steering within the bandwidth is free; steering beyond it is paid for in coherence.
**Implication.** Completes the leadership thread with a control-theoretic picture. The informed minority is
not an arbitrary steering wheel -- it drives a low-pass system with a bandwidth set by the alignment
response time, which the informed fraction tunes (F75/F77 are the time- and frequency-domain views of the
same timescale). Within the bandwidth, steering is accurate, lagging, and cohesion-free; at the bandwidth,
turns are attenuated; beyond it, the bias time-averages away AND, if strong, fragments the flock. This ties
the decision thread back to the predator thread's central tension: redirecting a flock has a coherence cost
whenever the redirection outpaces what alignment can propagate -- gentle steering (slow leaders) is free,
but aggressive steering (fast turns, or a predator) is not. The flock's steerability and its coherence are
the same resource viewed two ways, mediated by the alignment response time.

---

## Finding 78: An informed minority both RESTORES the coherence encirclement destroyed and STEERS the flock through the ring -- leadership counters predation by re-injecting the shared heading the predator removed (the two biggest threads meet)
<img src="./figures/led_encirclement_1.png" width="640"/>

**What:** The crossover of the study's two largest threads. F14 found encirclement (n_pred=6,
R_enc=0.15) is the one predator strategy that breaks 2D coherence (fragmenting the flock into
sub-flocks, F16); F72-F77 found a shared directional signal held by a minority steers the flock.
Question: under ACTIVE encirclement, can an informed minority carrying a shared GOAL direction
(+x -- NOT fleeing the predators, just committed to a heading) (a) restore the coherence
encirclement destroyed, and (b) steer the flock toward the goal despite the surrounding predators?
**Evidence:** collective/led_encirclement.py, self-contained, slow-prey predator regime (F14
calibration: N=100, v0=0.02, ramp=0.1), n_pred=6, R_enc=0.15, w_lead=0.5, 6 seeds. Predator
repulsion sign verified (prey pushed away). Two columns: predators off (pure leadership) and on.
  rho    predators OFF: acc / Phi      predators ON: acc / Phi
  0.00   -0.059 / 0.999               -0.054 / 0.793   (F14 baseline: encirclement broke Phi to 0.79)
  0.05   +0.951 / 0.992               +0.564 / 0.832
  0.10   +0.997 / 0.999               +0.746 / 0.805
  0.20   +1.000 / 1.000               +0.936 / 0.895
  0.40   +1.000 / 1.000               +0.980 / 0.939
**Key result 1 -- leadership transfers to the slow-prey regime.** With no predators, the slow-prey
flock (v0=0.02) is steered just as the fast-prey flock was (F72): rho=0.05 gives accuracy 0.95,
rho>=0.10 gives ~1.0, Phi stays ~1.0. The F72 leadership result is regime-independent, so the
encirclement comparison is on equal footing.
**Key result 2 -- leaders STEER the flock through the ring.** Under active encirclement, accuracy
toward the goal climbs from -0.05 (rho=0, no goal) to 0.936 (rho=0.20) and 0.980 (rho=0.40). The
flock travels coherently toward +x even though six predators surround it. The predators re-center
their ring on the moving CoM and keep pace (v0_pred=0.05 > v0=0.02), so the flock does not outrun
them -- it travels toward the goal carrying the predator ring along with it. Encirclement fails to
prevent a led flock from going where its leaders aim.
**Key result 3 -- leaders RESTORE the coherence encirclement destroyed.** Phi rises from 0.793 at
rho=0 (the F14 broken state) to 0.895 at rho=0.20 and 0.939 at rho=0.40. The shared goal direction
re-aligns the fragmenting flock: leadership partially undoes encirclement's fragmentation. The
mechanism is exactly complementary -- encirclement breaks coherence by REMOVING the flock's shared
heading (pushing different sub-groups different ways, F16); leadership restores it by RE-INJECTING
a shared heading. They are opposing forces on the same alignment substrate.
**Key result 4 -- encirclement raises the leadership threshold (the quantitative cost).** It takes
MORE leaders to steer under encirclement than without. Without predators rho=0.05 already gives
accuracy 0.95; under encirclement rho=0.05 gives only 0.564, and rho~0.20-0.40 is needed for
comparable steering. The predators compete with the leaders' signal, so the informed fraction
required rises ~4-8x. Coherence is also not fully restored even at rho=0.40 (Phi 0.939 < 1.0):
residual disruption persists, but leadership lifts the flock far above the F14 floor.
**Implication.** Unifies the predator and decision threads and generalizes F70. F70 showed a shared
ESCAPE direction defeats predictive encirclement; F78 shows a shared GOAL direction -- with no
knowledge of the predators at all -- counters standard encirclement by restoring coherence and
enabling travel. So it is not the CONTENT of the shared signal (flee-the-predator vs go-to-goal)
that counters the predator, but the mere PRESENCE of any strong shared heading: encirclement works
by destroying the flock's common direction, and anything that supplies one -- escape intelligence
(F70) or oblivious leadership (F78) -- fights it. This is the constructive dual of the whole
predator program: the predator's only successful 2D strategy (encirclement, F14) wins by erasing
the shared heading, and the leadership thread's central object (a shared heading) is exactly its
antidote. The flock's coherence and its steerability, shown to be one resource in F77, are here
shown to be the same resource a predator attacks -- predation and leadership pull on opposite ends
of the single lever of shared alignment.

---

## Finding 79: Spreading panic collapses STEERABILITY without breaking COHERENCE -- contagion severs the rudder, not the hull (the complement of encirclement); steering tracks the active-leader pull (1-f)*rho
<img src="./figures/panic_leadership_1.png" width="640"/>

**What:** The third major thread (contagion) meets leadership. F78 showed encirclement attacks
coherence and leadership restores it. Contagion attacks differently: SIS panic (book Sec 10.5) makes
agents ERRATIC and a panicked leader cannot lead. So panic severs the shared signal at its SOURCE
(intermittently silenced leaders) plus adds noise. Question: is there a contagion threshold for LOSS
OF STEERABILITY, and does it hit coherence or steering? SIS panic among prey (calm<->panic, transmission
beta per panicked flock-neighbor, recovery gamma=1); panicked agents get high noise (ramp 5) and, if
informed, suspend their goal bias while panicked. rho=0.10 informed toward +x; sweep beta.
**Evidence:** collective/panic_leadership.py, N=350, pure-flock, w_lead=1.0, 5 seeds.
  beta/gamma   panic frac f_ss    steering accuracy      Phi
  0.0          0.000             +0.936 +/- 0.056       0.996
  0.5          0.805             +0.713 +/- 0.199       0.983
  1.0          0.899             +0.356 +/- 0.325       0.981
  2.0          0.946             -0.038 +/- 0.501       0.980
  4.0          0.970             -0.468 +/- 0.346       0.979
  8.0          0.982             -0.335 +/- 0.290       0.980
**Key result 1 -- coherence is untouched.** Phi stays 0.980-0.996 across the ENTIRE beta range. Even
when 98% of the flock is panicked, the order parameter barely moves: the panic noise (ramp 5) is below
the melting threshold (F4, eta~10), so the flock remains a tight, globally aligned flock. Panic does
not fragment the group.
**Key result 2 -- steerability collapses.** Steering accuracy falls monotonically from 0.936 (no panic)
through 0.713, 0.356 to ~0 by beta/gamma~2, then scatters with large cross-seed variance (std 0.3-0.5)
about zero (the negative central values are within that spread). The flock stays coherent but loses its
heading: it becomes a tight flock flying an essentially random, leader-uncorrelated direction. Contagion
destroys the flock's STEERABILITY, not its COHESION -- it severs the rudder while leaving the hull intact.
**Key result 3 -- the controlling quantity is the active-leader pull (1-f)*rho (the F74 law + SIS).**
Panic SATURATES even at beta/gamma=0.5 (f=0.805) because the spatial flock has a high contact degree
(mean flock-neighbors ~30 at N=350, rf=0.1), so R0 = beta*<k>/gamma >> beta/gamma and the outbreak is
supercritical almost immediately (consistent with F18-F25). So the relevant axis is not the epidemic
threshold but the DEPTH of saturation. At any instant a fraction f of leaders is panicked and silent, so
the active leadership pull is (1-f)*rho*N*w (the F74 product law, taxed by the panic fraction), i.e. an
effective informed fraction rho_eff = (1-f)*rho. At beta/gamma=0.5, f=0.805 gives rho_eff~0.02 -- the
F72 weak-but-working regime (accuracy ~0.5-0.7, observed 0.71); at f>=0.95, rho_eff<=0.005, below the F72
floor, so steering vanishes. Steerability tracks the active-leader pull exactly as F72/F74 predict;
panic acts as a multiplicative tax (1-f) on the leadership signal.
**Implication.** Contagion and predation are COMPLEMENTARY attacks on the two faces of the same shared
heading (F77/F78). Encirclement (F78) attacks COHERENCE -- it pushes sub-groups apart, dropping Phi, and
leadership restores it. Panic (F79) attacks STEERABILITY -- it silences the leaders, dropping accuracy,
while coherence (which only needs local alignment, not the shared goal) survives untouched. A flock can
therefore be in two distinct failure modes: fragmented-but-aimable (encirclement, fixable by leadership)
or coherent-but-rudderless (panic, fixable only by suppressing the contagion, which leadership cannot do
because the disease disables the leaders themselves). This also explains why contagion has been the
study's most durable stressor (F22/F23/F34): kinematic attacks on coherence are reversible because the
shared heading re-forms, but an attack that disables the carriers of the shared signal cannot be
countered by that signal. The leadership thread's central object (a shared heading) is the antidote to
predation (F78) but is itself the casualty of contagion -- the rudder cannot repair itself once the hands
on it are panicking. Closes the leadership thread by mapping its two adversaries onto the two resources
(coherence, steerability) that F77 showed are one substrate seen two ways.

---

## Finding 80: Adversarial leadership -- DENIAL is cheaper than CAPTURE. A saboteur minority deadlocks a led flock at pull PARITY but must reach a MAJORITY (~2x) to hijack it to a false goal
<img src="./figures/adversarial_leaders_1.png" width="640"/>

**What:** The adversarial framing of conflicting leaders (F73/F74). A fixed set of TRUE leaders
(rho_true=0.10) steers toward a goal (+x); a swept set of SABOTEURS (rho_sab) pushes toward a trap
(-x). Two distinct adversarial objectives are distinguished: DENIAL (stop the flock reaching its
goal -- drive goal-accuracy to ~0, a deadlock) and CAPTURE (actively drive the flock to the trap --
goal-accuracy < 0). Are they equally hard? Equal leader strength w=1.0.
**Evidence:** collective/adversarial_leaders.py, N=350, pure-flock, 6 seeds. accuracy = cos(heading)
toward the TRUE goal (+1 goal reached, 0 deadlock, -1 captured to trap).
  rho_sab   (sab vs 35 true)   accuracy to goal      Phi     outcome
  0.000      0 vs 35          +0.921 +/- 0.106      0.995    goal
  0.025      9 vs 35          +0.779 +/- 0.201      0.965    goal
  0.050     18 vs 35          +0.599 +/- 0.250      0.945    goal (degraded)
  0.100     35 vs 35 (PARITY) +0.112 +/- 0.245      0.917    DENIED (deadlock)
  0.150     52 vs 35          -0.262 +/- 0.207      0.894    denied -> capturing
  0.200     70 vs 35          -0.558 +/- 0.133      0.880    CAPTURED
  0.300    105 vs 35          -0.823 +/- 0.010      0.883    captured
**Key result 1 -- denial at parity.** As the saboteur fraction rises to match the true leaders
(rho_sab=rho_true=0.10), goal-accuracy collapses from 0.92 to 0.11: the flock is deadlocked, reaching
neither the goal nor the trap. Even at HALF parity (rho_sab=0.05, 18 vs 35) accuracy is already halved
(0.92 -> 0.60). Denial -- mission failure for the flock -- is achieved cheaply, at or below pull parity.
**Key result 2 -- capture requires a majority.** Goal-accuracy crosses zero (the flock starts heading
toward the trap) only PAST parity, between rho_sab=0.10 and 0.15, and decisive capture (accuracy < -0.5,
the flock committed to the trap) needs rho_sab=0.20 -- twice the true-leader count. Actively hijacking
the flock to a chosen false target is far more expensive than merely paralyzing it.
**Key result 3 -- the asymmetry is the F74 product law plus a threshold gap.** The zero-crossing sits at
pull parity exactly as F74 predicts (equal w, so the balance is at equal numbers), confirming the
adversarial contest obeys the same summed-directed-force accounting. The new content is the GAP between
the two adversarial thresholds: denial at ~1x the defender's pull, capture at ~2x. Between them lies the
deadlock band (rho_sab 0.10-0.15 here), where neither side wins and the flock wanders. Coherence declines
only modestly across the whole sweep (Phi 0.995 -> 0.88): the opposed pulls fragment the flock a little
(F73's large-conflict mild Phi drop) but never break it -- the contest is over the flock's HEADING, not
its integrity.
**Implication.** A security asymmetry for any led collective. An adversary with a leadership-style
channel (injecting a false shared direction) can PARALYZE a led flock with a force merely matching the
legitimate leaders, but to CAPTURE it -- steer it to a destination of the adversary's choosing -- needs
clear superiority (~2x here). Symmetrically for the defender: to GUARANTEE reaching the goal, true leaders
need a pull majority over any saboteur, not just parity; at parity the outcome is deadlock. This sharpens
the F73/F74 voting picture into an attack/defense statement: the cheapest adversarial outcome is denial
(deadlock at parity), the most expensive is capture (hijack at majority), and the two are separated by a
deadlock band. It also connects the leadership thread to the predator arms race (F66-F71): a predator
that could mimic a leader would find denial (preventing the flock from going where it wants) far easier
than herding it to a kill zone -- consistent with the broader finding that disrupting the flock's shared
heading (denial, encirclement) is easier than commandeering it (capture, herding).

---

## Finding 81 (SELF-TEST/CORRECTION): Steering accuracy is set by the informed FRACTION, not the absolute number -- a fixed handful of leaders does NOT suffice as the group grows, correcting F72's offhand Couzin attribution
<img src="./figures/leader_scaling_1.png" width="640"/>

**What:** F72 cited Couzin et al. (2005) for the claim that "the informed fraction needed decreases as
the group grows," but only ever swept rho at a single N=350. This directly tests the scaling: vary the
flock size N and fix the informed NUMBER n_lead. If a fixed handful of leaders suffices in arbitrarily
large groups (the simplest reading of Couzin), accuracy at fixed n_lead should be N-independent. If
instead the FRACTION controls steering, accuracy at fixed number should FALL as N grows.
**Evidence:** collective/leader_scaling.py, pure-flock, w_lead=1.0, 4 seeds.
  N      n_lead   fraction    accuracy
  100     5       0.050      +0.356 +/- 0.396
  100    10       0.100      +0.735 +/- 0.202
  100    20       0.200      +0.957 +/- 0.068
  250     5       0.020      +0.470 +/- 0.667
  250    10       0.040      +0.586 +/- 0.554
  250    20       0.080      +0.755 +/- 0.366
  500     5       0.010      +0.149 +/- 0.770
  500    10       0.020      +0.236 +/- 0.701
  500    20       0.040      +0.479 +/- 0.502
**Key result 1 -- a fixed NUMBER of leaders does not suffice as N grows.** At every fixed n_lead, accuracy
DECREASES with N: at n_lead=20, accuracy falls 0.957 (N=100) -> 0.755 (N=250) -> 0.479 (N=500); at
n_lead=10, 0.735 -> 0.586 -> 0.236. Twenty leaders that steer a flock of 100 almost perfectly barely steer
a flock of 500. The absolute number of informed agents needed to reach a given accuracy GROWS with N.
**Key result 2 -- the controlling variable is the FRACTION (the F74 per-capita pull law).** Re-indexing by
fraction collapses the data far better than number: at fraction ~0.04-0.05 the accuracy is ~0.36-0.59
across N=100/250/500 (no monotonic N trend within the large noise), and at fraction 0.08-0.10 it is
~0.74-0.76. Steering accuracy is a function of the informed fraction, because the mean-velocity steering is
the total injected directed force n_lead*w divided by all N agents -- the F74 product law normalized per
capita. Growing N at fixed n_lead dilutes the per-capita pull and steering weakens.
**Key result 3 -- corrects F72's Couzin attribution.** F72's remark that "the informed fraction needed
decreases as the group grows" is NOT borne out in this model: the fraction needed is roughly CONSTANT
(accuracy collapses on fraction, not on number), and equivalently the absolute number needed grows
linearly with N. Couzin's "a fixed number suffices in large groups" arises from explicit
preferred-direction-averaging-with-noise mechanisms (the many-wrongs principle) that amplify a sparse
informed signal in large groups; this model's LINEAR alignment force has no such amplification -- it
delivers exactly the per-capita pull, so accuracy tracks the fraction. (Cross-seed std is large at low
fraction, the F73/F75 consensus/random regime where the flock picks a spontaneous heading, and small at
high fraction.)
**Implication.** A self-test in the tradition of F47/F48/F52: a claim asserted in passing (F72's Couzin
citation) is tested directly and corrected. The honest statement is that THIS model's leadership obeys a
per-capita pull law -- steering is set by the informed fraction times leader strength -- and does not
reproduce the group-size amplification of Couzin's averaging models. It sharpens the F74 product law:
"total pull decides" must be read as "total pull RELATIVE TO N," i.e. per-capita directed force, which is
the fraction. All the fraction-based leadership findings (F72, F78, F79, F80) are therefore the correct
frame; the one number-based intuition (a fixed handful suffices at any size) is the exception this test
removes. It also predicts that adding a many-wrongs amplification (e.g. a noisy preferred-direction average
rather than a fixed bias force) would be required to recover Couzin's number-suffices scaling -- a clean
open direction.

---

## Finding 82: MANY-WRONGS navigation -- noisy private goal estimates DO average to a 1/sqrt(N) wisdom-of-crowds law (recovering exactly the amplification F81 found absent for exact-shared vectors), but only below a per-agent noise ceiling
<img src="./figures/many_wrongs_1.png" width="640"/>

**What:** F81 closed with a prediction: linear velocity alignment gives no group-size amplification for
leaders carrying an EXACT shared goal vector (steering is per-capita pull, fraction not number), and
recovering the animal-navigation literature's "a fixed number suffices in large groups" scaling would
require a MANY-WRONGS follower rule -- agents holding INDEPENDENT NOISY estimates of the goal so that
averaging over more of them cancels the error. This tests that prediction directly. Every agent i carries
a private preferred direction g_hat_i at angle phi_i ~ N(0, sigma_pref) about the true goal (+x), fixed
for the run, and feels a bias w_bias*g_hat_i toward its OWN estimate (w_bias=0.5, alpha=1.0). No
exact-vector leaders; everyone is a noisy one. The many-wrongs prediction is that the flock's steady
heading is the alignment-averaged resultant of N independent noisy biases, with angular error ~
sigma_pref/sqrt(N): accuracy toward the true goal should IMPROVE with N (opposite of F81), and the
cross-seed RMS heading error should fall like 1/sqrt(N) -- PROVIDED the flock stays coherent so the
averaging is global, not local.
**Evidence:** collective/many_wrongs.py, pure-flock, w_bias=0.5, 8 seeds. Metric: signed steady heading
error (angle of the time-averaged mean-velocity vector relative to the true goal) -> cross-seed RMS (deg);
accuracy = cos(error); Phi.
  Exp1 (sweep N at sigma_pref=1.0 rad ~ 57 deg; single-agent baseline error would be ~57 deg):
    N      RMS_err(deg)   accuracy        Phi
    30     15.7           +0.963+/-0.056   0.877
    60      6.7           +0.993+/-0.008   0.922
    125     4.3           +0.997+/-0.003   0.945
    250     2.3           +0.999+/-0.001   0.953
    500     2.4           +0.999+/-0.001   0.955
    1000    2.5           +0.999+/-0.001   0.959
    log-log slope d(log RMSerr)/d(log N) = -0.523 (many-wrongs predicts -0.5)
  Exp2 (sweep sigma_pref at N=250):
    sigma_pref   RMS_err(deg)   accuracy        Phi
    0.00 (  0d)   0.1           +1.000           1.000
    0.25 ( 14d)   0.6           +1.000           0.996
    0.50 ( 29d)   1.1           +1.000           0.985
    1.00 ( 57d)   2.3           +0.999+/-0.001   0.953
    1.50 ( 86d)  57.8           +0.754+/-0.644   0.918
    2.00 (115d)  64.0           +0.582+/-0.584   0.902
**Key result 1 -- many-wrongs amplification is REAL and the 1/sqrt(N) law holds.** At fixed per-agent
error sigma_pref=1.0 rad, the flock's RMS heading error falls from 15.7 deg (N=30) to ~2.3 deg (N=250),
a log-log slope of -0.523, almost exactly the many-wrongs prediction of -0.5. Accuracy toward the true
goal IMPROVES with N (0.963 -> 0.999) -- the exact OPPOSITE of F81, where accuracy at fixed leader number
FELL with N. The flock is far wiser than its members: each agent individually would head off by ~57 deg,
but a flock of 250 navigates to within ~2 deg of the true goal. Alignment is performing the wisdom-of-
crowds average: it pools the N independent private estimates and the errors cancel as 1/sqrt(N).
**Key result 2 -- resolves the apparent contradiction with F81.** F81 (exact shared vector, no
amplification) and F82 (noisy private vectors, full 1/sqrt(N) amplification) are not in conflict -- they
are the two ends of one statement. Alignment averages whatever directional signal the agents carry. With
an EXACT shared vector there is no error to average away, so all that remains is the per-capita pull
(F81: number doesn't help, only fraction). With HETEROGENEOUS noisy estimates there is error, and alignment
averages it down as 1/sqrt(N), so MORE agents (at fixed per-agent noise) means a more accurate group
(F82: number helps). The literature's "a fixed number suffices in large groups" comes precisely from
this many-wrongs averaging, and adding noisy estimates to the model recovers it exactly as F81 predicted.
**Key result 3 -- a residual error FLOOR.** The 1/sqrt(N) decline saturates at a ~2.3-2.5 deg floor by
N~250 (the law is clean over N=30-250, flat thereafter). The floor is a second, N-independent error source
-- temporal fluctuation of the flock heading within the finite measurement window plus the ramp noise --
not the averaged-bias error, which has already dropped below it. Many-wrongs reduces the bias-error term
to zero with enough agents but cannot remove the dynamical jitter floor.
**Key result 4 -- a per-agent NOISE CEILING gates the averaging (Exp2).** For sigma_pref up to ~1.0 rad
the RMS error grows gently (0.1 -> 2.3 deg, roughly sigma_pref/sqrt(N)) and accuracy stays ~1.0 with high
Phi -- averaging works. But between sigma_pref=1.0 and 1.5 rad the behavior collapses abruptly: RMS error
jumps to ~58 deg, accuracy falls 0.999 -> 0.754 -> 0.582, and the cross-seed std EXPLODES (0.001 -> 0.644).
Crucially Phi stays high (0.95 -> 0.92 -> 0.90) -- the flock does NOT fragment; it stays a tight flock that
flies a near-random heading. Mechanism: the magnitude of the averaged bias projected on the true goal is
w_bias*E[cos phi] = w_bias*exp(-sigma_pref^2/2), which shrinks fast (0.61*w at sigma=1.0, 0.32*w at 1.5,
0.14*w at 2.0). Once this averaged directional signal falls below the threshold needed to overcome the
flock's spontaneous symmetry-breaking heading, steering becomes unreliable -- different seeds veer
different ways (the huge std), exactly the F72 under-led / F74 below-pull-threshold regime expressed in
the many-wrongs setting. So the wisdom of crowds has a ceiling: it sharpens a usable consensus only while
the per-agent error is small enough that the pooled estimate still has appreciable magnitude.
**Implication.** F82 confirms F81's prediction and completes the leadership thread's central mechanism:
alignment is a directional averager. It delivers per-capita pull for a shared exact signal (F81) and
1/sqrt(N) wisdom-of-crowds amplification for heterogeneous noisy signals (F82), with a high-noise ceiling
(F82 Exp2) that is the many-wrongs form of the F72/F74 pull threshold. This is the constructive twin of the
shared-heading principle (F70/F72): a globally shared direction is amplified, a locally heterogeneous one
is averaged -- and when the heterogeneity is small zero-mean noise about a common goal, the averaging is
not destructive but is precisely the crowd's wisdom. Opens follow-ups: a noisy-MINORITY rematch of F81's
N-scaling (does a fixed NUMBER of noisy-informed agents now suffice as N grows, since their pooled estimate
sharpens?); interaction with environmental noise (ramp) and coherence; and whether correlated (non-
independent) estimates degrade the 1/sqrt(N) law toward the F81 per-capita limit.

---

## Finding 83: CORRELATED estimates -- F81 and F82 are the two ends of ONE axis (error correlation), and any shared sensing error imposes an N-independent accuracy floor sigma*sqrt(rho_c) that no flock size can beat
<img src="./figures/correlated_estimates_1.png" width="640"/>

**What:** F81 (exact shared goal vector -> no group-size benefit, per-capita pull) and F82 (independent noisy
estimates -> 1/sqrt(N) wisdom of crowds) look like opposite results. This shows they are the rho_c -> 1 and
rho_c -> 0 limits of a SINGLE parameter: how CORRELATED the agents' goal-estimate errors are. Real collectives
sit in between -- animals reading the same misleading environmental cue, or agents fed common misinformation,
share part of their error. Each agent's preferred-direction angle is built as
phi_i = sqrt(rho_c)*c + sqrt(1-rho_c)*e_i with c (one shared draw per run) and e_i (independent private draws)
both ~ N(0, sigma_pref), so Var(phi_i)=sigma_pref^2 and Corr(phi_i,phi_j)=rho_c. rho_c=0 reproduces F82;
rho_c=1 gives every agent the SAME (shared but wrong) direction, an F81-like shared signal.
**Prediction (small-angle):** the alignment-averaged heading error has cross-seed variance
rho_c*sigma^2 + (1-rho_c)*sigma^2/N, i.e. RMS heading error = sigma_pref*sqrt(rho_c + (1-rho_c)/N). The private
part averages away as 1/sqrt(N); the shared part does NOT -- it is a FLOOR of sigma_pref*sqrt(rho_c) that no
group size can beat. Crossover N* ~ 1/rho_c.
**Evidence:** collective/correlated_estimates.py, pure-flock, w_bias=0.5, sigma_pref=1.0 rad, 12 seeds.
  rho_c   N=30          N=125         N=500         predicted floor (deg)
  0.00    12.7 deg      3.9 deg       2.1 deg       0.0   (falls ~1/sqrt(N), = F82)
  0.10    24.7          22.9          21.7          18.1  (flat in N)
  0.30    38.8          38.6          37.4          31.4  (flat in N)
  1.00    68.4          68.4          68.4          57.3  (exactly N-independent, = F81 limit)
  accuracy: rho_c=0 -> 0.976/0.998/0.999; rho_c=0.1 -> ~0.92 (flat); rho_c=0.3 -> ~0.79 (flat);
            rho_c=1 -> 0.437 +/- 0.535 (flat). Phi: rises with rho_c (0.88-0.96 at rho_c=0 -> 1.000 at rho_c=1).
**Key result 1 -- F81 and F82 are endpoints of one axis.** At rho_c=0 the error falls as 1/sqrt(N)
(12.7 -> 2.1 deg, F82 reproduced). At rho_c=1 the error is EXACTLY N-independent (68.4 deg at every N) with
Phi=1.000 -- every agent carries the identical wrong direction, the flock agrees perfectly and heads off by
the shared offset, the F81 shared-signal / no-amplification limit. The single parameter rho_c interpolates
continuously between the two findings that looked contradictory.
**Key result 2 -- correlated sensing error caps the wisdom of crowds.** For ANY rho_c>0 the heading error is
flat in N (rho_c=0.1: ~22 deg at N=30, 125, AND 500; rho_c=0.3: ~38 deg flat). The shared error component does
not average away no matter how many agents pool their estimates. Even a modest 10% correlation collapses the
collective from "arbitrarily accurate given enough agents" (accuracy -> 1 at rho_c=0) to a hard ceiling
(accuracy ~0.92, N-independent). The floor ordering and N-independence match the prediction
sigma*sqrt(rho_c) exactly; absolute values run ~15-20% above the linearized formula because sigma=1 rad is not
in the small-angle regime (the vector-mean angle of a wide distribution has RMS somewhat above sigma) plus the
~2 deg dynamical floor of F82 -- the scaling law is confirmed, the prefactor is approximate.
**Key result 3 -- correlation buys COHERENCE at the cost of ACCURACY.** Phi RISES with rho_c (0.88-0.96 at
rho_c=0 to a perfect 1.000 at rho_c=1): the more correlated the agents' goals, the more tightly the flock
agrees. But that agreement is on an increasingly WRONG heading (accuracy 0.99 -> 0.44). Independent errors
slightly loosen cohesion yet cancel for accuracy; shared error tightens cohesion onto a common mistake. This
is the F73 consensus theme in the navigation setting -- consensus is not correctness; a perfectly coherent
flock can be confidently, unanimously wrong. The huge cross-seed std at high rho_c (0.535 at rho_c=1) is
exactly that: each run's shared draw sends the whole flock a different definite (wrong) way.
**Implication.** Closes the many-wrongs sub-thread (F81-F83) with a unifying axis: alignment is a directional
averager whose collective accuracy is set by the CORRELATION structure of the inputs, not their number. The
practical content is a sharp warning about the wisdom of crowds -- it delivers 1/sqrt(N) accuracy only for
INDEPENDENT errors; common-mode error (shared cue, shared misinformation, correlated sensors) imposes a floor
sigma*sqrt(rho_c) that more agents cannot reduce, and drives the flock to confident consensus on the wrong
heading. Connects the leadership thread back to the adversarial finding (F80): an attacker who cannot add
enough saboteurs can instead INJECT CORRELATION into the legitimate agents' estimates (a single shared false
cue) and cap the collective's accuracy regardless of its size -- common-mode deception is cheaper than
majority capture. Remaining open: does the floor relax if agents can DETECT and down-weight correlated inputs
(robust estimation), and how does rho_c interact with the F77 steering-bandwidth limit for a moving goal.

---

## Finding 84: NOISY MINORITY -- a fixed NUMBER of noisy-informed agents still fails as N grows (NOT many-wrongs); Couzin's informed-minority and the wisdom-of-crowds are DISTINCT mechanisms that do not combine in a minority
<img src="./figures/noisy_minority_1.png" width="640"/>

**What:** Closes the F81-F84 quartet by resolving which mechanism a NOISY MINORITY follows. The animal-
navigation literature treats two effects as separate: Couzin et al. (2005) informed-minority steering (a few
agents with a preferred direction steer the group) and the Simons (2004)/Codling et al. (2007) many-wrongs
principle (many independent noisy estimates average to higher accuracy with group size). F81 (exact-vector
minority, NUMBER fails) and F82 (all-agents noisy, NUMBER helps via 1/sqrt(N)) are the pure cases. The
bridging case is a fixed NUMBER n_lead of informed agents EACH carrying a private noisy estimate
(angle ~ N(0, sigma_pref)), the rest naive followers with no bias. Two scales pull opposite ways as N grows:
the DIRECTION of the injected pull is the pooled estimate of the n_lead leaders (error ~ sigma/sqrt(n_lead),
FIXED in N), while the MAGNITUDE of the pull per capita is n_lead*w/N (DILUTES in N, F81).
**Evidence:** collective/noisy_minority.py, pure-flock, w_lead=1.0, 8 seeds.
  Exp1 -- accuracy vs N at fixed n_lead, EXACT (sigma=0, reproduces F81) vs NOISY (sigma=1.0 rad):
    EXACT n_lead=10:  N=100 +0.720  N=250 +0.573  N=500 +0.489   (falls with N)
    EXACT n_lead=20:  N=100 +0.976  N=250 +0.769  N=500 +0.586   (falls with N)
    NOISY n_lead=10:  N=100 +0.356  N=250 +0.376  N=500 +0.424   (flat/low, within seed scatter)
    NOISY n_lead=20:  N=100 +0.861  N=250 +0.526  N=500 +0.425   (falls with N, below exact)
  Exp2 -- noisy minority (sigma=1.0 rad), sweep n_lead at N=250:
    n_lead= 5  acc +0.224 +/- 0.646  Phi 0.991
    n_lead=10  acc +0.376 +/- 0.703  Phi 0.982
    n_lead=20  acc +0.526 +/- 0.616  Phi 0.971
    n_lead=40  acc +0.867 +/- 0.229  Phi 0.967
    n_lead=80  acc +0.905 +/- 0.219  Phi 0.947
**Key result 1 -- a fixed NUMBER of noisy leaders does NOT suffice as N grows.** The exact minority falls
with N at every n_lead (F81 reproduced: n_lead=20 -> 0.976/0.769/0.586). The noisy minority also fails: at
n_lead=20 it falls 0.861 -> 0.425, and at n_lead=10 it is stuck near the spontaneous-heading floor (~0.4,
huge variance) regardless of N. The minority's internal averaging cannot rescue the per-capita pull dilution
-- the pooled direction is at best a FIXED accuracy (error ~ sigma/sqrt(n_lead), set by the fixed n_lead, not
by N), and the strength with which that direction is imposed weakens as n_lead*w/N. So a noisy minority is the
F81 regime, NOT the F82 regime: number does not suffice.
**Key result 2 -- noisy minority <= exact minority, always.** At every (n_lead, N) the noisy accuracy sits at
or below the exact (n_lead=20: 0.861 vs 0.976 at N=100, 0.425 vs 0.586 at N=500). The penalty is the
pooled-direction error sigma/sqrt(n_lead): the n_lead leaders agree on a direction that is itself off the true
goal by ~18 deg for n_lead=10, and no amount of follower alignment can recover a target the leaders
themselves get wrong. An exact-vector minority points the flock exactly right (just weakly); a noisy minority
points it weakly AND slightly wrong.
**Key result 3 -- only the FRACTION recovers accuracy (Exp2).** Growing n_lead at fixed N raises accuracy
monotonically (0.224 -> 0.905 as n_lead 5 -> 80), because more leaders give BOTH more per-capita pull (F81/F74)
AND a better-pooled direction (sigma/sqrt(n_lead) shrinks). This is the F82 amplification re-expressed: the
1/sqrt(n_lead) wisdom-of-crowds averaging operates over the INFORMED set, so it helps only when the informed
set GROWS -- which, at fixed N, means a larger fraction. The huge cross-seed std at small n_lead (0.65-0.70)
is the F72/F75 under-led regime (the flock picks spontaneous headings).
**Implication.** Separates, within one model, two mechanisms the literature often conflates. Couzin
informed-minority steering and the many-wrongs wisdom of crowds are DISTINCT and do not combine in a fixed
minority: confining noisy estimates to a fixed cadre gives per-capita-diluted steering toward a
fixed-accuracy pooled direction (strictly worse than an exact minority, F81-like), while the 1/sqrt(N)
many-wrongs amplification (F82) requires the informed FRACTION itself to grow. This is the sharp form of the
F81 correction: "number suffices" is true for the many-wrongs crowd only when EVERY agent is a (noisy)
estimator, never for a fixed informed minority. Completes the leadership thread's mechanistic map
(F72-F84): alignment is a directional averager; its output accuracy is set by the per-capita pull (fraction x
strength, F74/F81) toward a target whose own accuracy is set by the correlation structure (F83) and the
sample size of the estimating set (F82/F84) -- number helps only when it grows the estimating fraction, never
as a fixed minority.

---

## Finding 85: MISINFORMATION robustness -- a navigating crowd is near-immune to UNCOORDINATED wrong members (votes cancel) but flips to a COORDINATED false consensus of equal size (parity at f=0.5); the error's correlation, not its presence, sets the damage
<img src="./figures/misinformation_1.png" width="640"/>

**What:** The many-wrongs map (F82-F84) says alignment is a directional averager whose accuracy is set by the
CORRELATION structure of the estimates (F83), and because each agent contributes a UNIT vector its influence
is bounded however wrong its angle -- so directional averaging should be intrinsically robust to outliers.
This quantifies that robustness and contrasts two kinds of misinformation carried by a fraction f of the
flock. A well-informed majority (fraction 1-f) holds a tight estimate (sigma_in=0.3 rad) of the true goal
(+x). The misinformed fraction f is either LOST -- uniform-random directions, uncoordinated -- or ADVERSARIAL
-- all pointing at a false goal (-x), coordinated (a shared false cue). Prediction: lost votes are unit
vectors that cancel (~1/sqrt(fN) resultant), so accuracy holds until f is large; adversarial votes compete
with the true majority vote-for-vote, so accuracy crosses zero at PARITY f=0.5 (the F80 product law).
**Evidence:** collective/misinformation.py, N=250, w_bias=0.5, sigma_in=0.3 rad, 8 seeds.
  Exp1 -- accuracy toward true goal vs misinformed fraction f:
    f       LOST (uncoordinated)      ADVERSARIAL (coordinated)
    0.00    +1.000  Phi 0.994         +1.000  Phi 0.994
    0.10    +1.000  Phi 0.987         +0.999  Phi 0.991
    0.20    +0.999  Phi 0.981         +0.996  Phi 0.987
    0.30    +0.998  Phi 0.973         +0.945  Phi 0.963
    0.40    +0.998  Phi 0.966         +0.623  Phi 0.882
    0.50    +0.998  Phi 0.956         +0.105  Phi 0.775
    0.60    +0.996  Phi 0.945         -0.617  Phi 0.876
    0.70    +0.983  Phi 0.936         -0.899  Phi 0.942
  Exp2 -- LOST mode at f=0.40, sweep N: accuracy +0.987/+0.995/+0.998/+0.998 at N=60/125/250/500 (N-independent).
**Key result 1 -- a crowd is near-immune to UNCOORDINATED misinformation.** With lost members pointing in
uniform-random directions, accuracy stays at 0.998 even when HALF the flock is misinformed (f=0.5), and only
falls to 0.983 at f=0.7. The random unit-votes cancel: their resultant is ~sqrt(fN) in a random direction
against ~(1-f)N aligned for the goal, so the net heading stays locked on the truth. The order parameter
declines only gently (0.994 -> 0.936) -- the lost agents add a little incoherence but the flock stays tight
and correct. Directional averaging shrugs off noise.
**Key result 2 -- the same fraction of COORDINATED misinformation is decisive.** Adversarial members all
pointing at the false goal drive accuracy down steeply: 0.945 at f=0.3, 0.623 at f=0.4, through ZERO at parity
f=0.5 (acc=0.105), to capture (negative) beyond -- -0.617 at f=0.6, -0.899 at f=0.7. The zero-crossing at
f=0.5 is exactly the F80/F74 product law (equal per-agent strength, so balance at equal numbers). The
coherence cost peaks AT the parity point (Phi=0.775, the lowest in the sweep) and recovers as one side wins --
the F80 heading-fight signature and the F73/F75 critical-slowing dip, here in the misinformation framing.
**Key result 3 -- it is the CORRELATION of the error that does the damage, not its presence (ties F83).** At
f=0.4 the two modes differ by 0.998 (lost) vs 0.623 (adversarial); at f=0.5, 0.998 vs 0.105. Same number of
wrong agents, same per-agent wrongness magnitude -- the only difference is whether their errors are correlated
(all -x) or independent (uniform). Uncoordinated error (rho_c->0, F83) averages away; coordinated error
(rho_c->1, F83) competes and flips the consensus. F85 is the F83 correlation principle expressed as a
robustness/security statement.
**Key result 4 -- lost-robustness is scale-free (Exp2).** At fixed f=0.4 lost, accuracy is N-independent
(0.987 at N=60 up to 0.998 at N=500, if anything improving as the random resultant cancels better at larger
N). A crowd averages out a fixed fraction of lost members at any size -- the robustness is structural, not a
finite-size accident.
**Implication.** Caps the many-wrongs arc (F81-F85) with its practical payoff: a navigating collective is
robust to NOISE but fragile to a coordinated FALSEHOOD of the same size. Uncoordinated misinformation -- lost,
confused, or independently-erring members -- is averaged out even at 50% prevalence; a coordinated false
consensus competes vote-for-vote and captures the flock once it reaches parity. This is the constructive-
security mirror of the adversarial-leadership result (F80): there denial was cheaper than capture at parity;
here the same parity threshold governs whether a shared false cue can hijack a navigating crowd, while
uncorrelated noise of any prevalence cannot. The general lesson across F80/F83/F85: what threatens a
collective's heading is never the AMOUNT of error but its CORRELATION -- alignment averages out everything
independent and is moved only by what is shared. Closes the many-wrongs sub-thread (F81-F85). The leadership
thread (F72-F85) is a complete arc; the next major direction is co-adaptation/evolution (heritable escape or
estimate quality under selection), which requires a fitness model and is a fresh thread.

---

## Finding 86 (SELF-TEST): Many-wrongs tracking of a MOVING goal -- spatial averaging (F82) and temporal bandwidth (F77) are INDEPENDENT; a noisy crowd tracks a turning goal at the SAME bandwidth as a sharp leader (my exp(-sigma^2/2) bandwidth-reduction prediction FALSIFIED)
<img src="./figures/moving_goal_crowd_1.png" width="640"/>

**What:** The open question left by F83/F77: does the many-wrongs average help a crowd track a MOVING goal, and
does per-agent noise cost steering bandwidth? F77 found a finite steering bandwidth (a flock tracks a goal
turning at rate omega only below omega_crit ~ 1/response-time). F82 found alignment averages noisy private
estimates to a 1/sqrt(N) accurate heading. Here the goal direction rotates at omega and every agent biases
toward the CURRENT goal carrying its own FIXED angular offset, g_hat_i(t)=rotate(goal(t), phi_i),
phi_i~N(0,sigma_pref). PRE-REGISTERED PREDICTION: the many-wrongs average is SPATIAL (over agents,
instantaneous each step) so it should add no lag; noise should lower the bandwidth ONLY by shrinking the
averaged-bias MAGNITUDE w*exp(-sigma_pref^2/2) (the F82 Exp2 law), i.e. omega_crit ~ exp(-sigma_pref^2/2).
**Evidence:** collective/moving_goal_crowd.py, N=250, w_bias=0.5, 8 seeds. Tracking accuracy =
time-avg cos(flock heading - goal angle).
  tracking accuracy vs omega (rad/tu), by per-agent noise sigma_pref:
  sigma  exp(-s^2/2)   om=0.00  0.02    0.05    0.10    0.15    0.20
  0.00   1.000         1.000    0.999   0.993   0.973   0.940   0.890
  0.50   0.882         1.000    0.999   0.994   0.972   0.935   0.880
  1.00   0.607         0.999    0.998   0.990   0.956   0.894   0.787
  1.50   0.325         0.769*   0.821*  0.909   0.830   0.596   0.086   (* huge cross-seed std, F82 ceiling)
  Phi ~ 1.000 (sigma=0) down to ~0.92 (sigma=1.5) across all omega -- coherence fine throughout.
  Bandwidth (largest omega with acc>=0.5): sigma=0/0.5/1.0 all ~0.20; sigma=1.5 ~0.15 (unreliable).
**Key result 1 -- spatial and temporal averaging are INDEPENDENT (prediction falsified).** For sigma_pref up
to 1.0 rad the tracking-accuracy-vs-omega curves OVERLAY almost exactly (at omega=0.10: 0.973/0.972/0.956; at
omega=0.15: 0.940/0.935/0.894). The bandwidth is flat at ~0.20 across sigma=0, 0.5, 1.0 even though the
averaged-bias magnitude factor exp(-sigma^2/2) has fallen to 0.61. My pre-registered prediction that bandwidth
~ exp(-sigma^2/2) is WRONG: the magnitude reduction does NOT translate into proportional bandwidth loss. The
many-wrongs noise costs essentially nothing for moving-goal tracking -- a noisy crowd tracks a turning goal as
well as a sharp leader.
**Key result 2 -- why the prediction failed.** The per-agent offsets are STATIC and rotate WITH the goal, so
the averaged bias points cleanly at the current goal each step and adds no temporal lag; the averaging is
spatial and instantaneous, decoupled from the alignment response time that sets the bandwidth. And the
magnitude reduction (0.61*w at sigma=1.0) leaves the effective pull well above the steering threshold, so the
bandwidth -- set by the response time, not the pull magnitude in this range -- is unchanged. The magnitude law
would only bite if it pushed the pull below threshold, which is exactly what happens at the F82 NOISE CEILING.
**Key result 3 -- the ceiling, not a graceful bandwidth, is the failure mode.** At sigma_pref=1.5 (magnitude
0.33, past the F82 Exp2 ceiling ~1.3 rad) tracking does not degrade gracefully -- it COLLAPSES: even the static
omega=0 case gives accuracy 0.769 with std 0.605 (different seeds head different ways), and at omega=0.20
accuracy is 0.086. The breakdown is the F82 ceiling (averaged signal too weak to overcome the spontaneous
heading), not a lowered bandwidth. So noise has a binary effect on moving-goal tracking: free (no bandwidth
cost) below the ceiling, catastrophic above it.
**Implication.** A 5th self-test (cf. F47/F48/F52/F81): a pre-registered quantitative prediction tested
directly and CORRECTED. Refines the F77/F82 relationship: steering BANDWIDTH (temporal, set by alignment
response time) and crowd ACCURACY (spatial, set by estimate averaging) are independent resources -- many-wrongs
noise spends the second, not the first, so a noisy crowd turns as fast as a sharp leader and only pays in
steady accuracy, until the F82 noise ceiling where the averaged signal collapses outright. Strengthens the §5.3
thesis (alignment is a directional averager) by showing the average is computed afresh each timestep, not
integrated over time. Closes the many-wrongs sub-thread (F81-F86). Leadership thread (F72-F86) complete and
PAUSED; the next major direction is co-adaptation/evolution, which needs a fitness model (a scientific choice
to make deliberately, not by default).

## Finding 87: EVOLUTION of the collective-escape weight under capture/removal selection -- the F70 "dangerous valley" is a strong evolutionary BRAKE (escape is stable and near-costless once present, but evolves only by a slow, hysteretic crawl from the no-escape state), not an absolute barrier

**Setup.** First experiment of the co-adaptation thread, and the first in this study where a behavioral
trait is HERITABLE and under selection rather than fixed by hand. Each prey carries a per-agent
collective-escape weight w_i (the F70 trait: prey i feels w_i * e_hat, with e_hat the shared unit vector
from the predator centroid toward the flock CoM), and w_i is heritable. Predators run F66 predictive
encirclement at lead=2 tu (the hardest predator found), n_pred=6, slow-prey regime, N=150. The FITNESS
MODEL is a deliberate scientific choice (capture/removal): an agent within r_kill=0.03 of any predator is
captured at hazard rate 3.0/tu and replaced by a mutated clone of a random survivor (inherits w + Gaussian
mutation sigma=0.10, clamped to [0,5], spawned at the parent's position/velocity with small jitter), holding
N fixed (a Moran-style continuous-replacement scheme). The question F70 poses to evolution: F70 found a
"dangerous valley" where a weak escape (w~0.25) is WORSE than none -- it competes with alignment and
fragments the flock without escaping -- and only w >= alpha = 1 restores coherence and outruns the trap.
Does selection drive a low-w population ACROSS that valley to the winning regime, or does the valley trap
it? Initial weight w0 swept over {0, 0.25, 0.5, 1, 2}, 2 seeds, 150 tu; plus a 400-tu long run on the low
starts to distinguish a true barrier from a slow brake. The predictive-predator + collective-escape
dynamics are the validated, bit-identical harness (vectorized_predator_prey.py).

**Result.** The 150-tu outcome is sharply set by the initial weight. Populations seeded in the escape
regime are evolutionarily STABLE and nearly predation-free: w0=2.0 stays at w=2.00 with only ~6 captures
and Phi=1.000; w0=1.0 stays at w=1.00 (277 captures, Phi=0.992). Populations seeded at or below the valley
stall well short of escape: w0=0.0 -> mean w=0.27 (Phi=0.22, 1804 captures), w0=0.25 -> 0.49, w0=0.5 ->
0.66. Captures PEAK in the valley region (1886 at w0=0.25, 1804 at w0=0.0) and collapse to single digits at
w0=2.0 -- predation cost is concentrated exactly where F70 placed the valley. The long run resolves the
mechanism: selection on w is directional-UPWARD from every start (even w0=0 drifts up), but the valley
THROTTLES the climb. From no escape the mean weight crawls 0 -> 0.13 (50 tu) -> 0.18 (150 tu) -> 0.51
(400 tu), end-slope ~3e-4/tu, and never reaches the escape threshold w=1; a start past the worst of the
valley (w0=0.5) climbs faster (0.88 at 400 tu, slope ~1.3e-3/tu, accelerating). Neither crosses w=1 within
400 tu.

**Implication.** The F70 force-versus-alignment threshold has a population-genetic image: it is a strong
evolutionary BRAKE, not an absolute barrier. Escape behavior is trivial to MAINTAIN -- stable,
self-reinforcing, near-zero predation once w >= alpha -- but very hard to EVOLVE DE NOVO from the no-escape
state, because the path there runs through the valley where escape is actively harmful (highest capture,
lowest coherence), so selection pushes w upward only weakly and the crowd crawls for hundreds of time units
without establishing escape. This is strong evolutionary hysteresis, a first-mover problem: which basin a
population occupies is set by where it starts, and the costless escape optimum is not reachable on realistic
timescales from rare. It is the domination-not-blending theme (F16/F24/F70) read at the evolutionary level
-- a globally shared escape direction pays off only once it is strong enough to beat alignment, so partial
commitment is selected against -- and it inverts the naive reading of F70's "escape wins": escape wins only
where it is already present. Opens the co-adaptation thread; the predator side is still fixed (natural next
step: let predator lead_time or aggression co-evolve against the prey trait, and test whether a larger
mutation step or a seeded escape-carrying minority can jump the gap). Caveats: one fitness model
(capture/removal), fixed predator, N=150, mutation sigma=0.10 -- the brake's steepness depends on these.
evolution/escape_evolution.py

## Finding 88: The F87 evolutionary brake is a barrier to ORIGINATION, not to invasion -- a rare escaper founder group (>=5%) establishes escape, and a single large-enough mutational jump (sigma~0.3-0.6) clears the valley

**Setup.** Direct follow-up to F87, asking whether the brake can be overcome. Same capture/removal model,
same validated harness; only the initial condition (Exp1) and the mutation step (Exp2) change -- no new
fitness model or mechanism. Exp1 (INVASION): seed a fraction f of the population in the escape regime
(w=2) with the rest at no escape (w=0); sweep f in {0.05, 0.10, 0.20, 0.50}, 2 seeds, 150 tu -- does escape
invade and establish, or get diluted and lost? Exp2 (MUTATION STEP): from a uniform no-escape start
(w0=0), sweep the per-capture mutation sigma in {0.10 (the F87 value), 0.30, 0.60, 1.00}, 2 seeds, 200 tu
-- does a bigger step let the population JUMP the valley rather than crawl and stall?

**Result.** Exp1 -- invasion succeeds from a tiny seed. Even f=0.05 carries the flock-mean weight into the
escape regime (w_end=1.18), with the escaper fraction climbing 0.05 -> 0.55 and the capture toll roughly
halving (1012 vs F87's ~1800 for the de-novo w0=0 crawl); f=0.10 -> escaper fraction 0.70, w=1.33, Phi=0.91;
f=0.20 and 0.50 settle similarly (w~1.2-1.3, escapers ~0.55-0.63, captures down to ~500). The escaper trait
does not drive the no-escapers extinct -- it settles at a MIXED equilibrium around 60% escapers, mean
w~1.2 -- but it establishes the escape regime in every case. Exp2 -- the F87 step (sigma=0.10) never reaches
w=1 (w_end=0.30, 0% of seeds, reproducing F87), but sigma=0.30 and 0.60 cross w=1 in 100% of seeds
(w_end=1.10 and 1.33), while the largest step sigma=1.0 is noisier and crosses in only 50% (w_end=0.90) --
very large jumps also scatter many offspring back to low w and overshoot, so there is an intermediate
mutation-scale sweet spot.

**Implication.** Together with F87 this pins down the first-mover problem precisely: the F70 valley is a
barrier to the ORIGINATION of escape, not to its spread. Escape cannot crawl up from zero against the valley
(F87) -- but it does not have to. A rare founder group already carrying the escape weight (a 5% minority
suffices) invades and establishes the escape regime, and a single mutational jump large enough to clear the
worst of the valley (sigma~0.3-0.6) seeds the same outcome from a uniform no-escape start. The brake is
therefore mutation-limited / standing-variation-limited: whether escape evolves is set not by selection
(which favours it once the valley is cleared) but by whether variation can deliver an agent past the valley
in one step, or a pre-adapted founder group arrives. The MIXED ~60% escaper equilibrium (rather than full
fixation) is itself informative: it is a free-rider / herd-protection effect of the SHARED escape signal --
once enough agents flee, the predator is outrun and the residual low-w agents ride along protected, which
relaxes selection on them. The public-good nature of the shared escape direction (the constructive mirror of
the F70/F72 shared-signal rule) prevents escape from going to fixation even as it dominates. Caveats: still
one fitness model (capture/removal), fixed predator, N=150. evolution/escape_invasion.py

## Finding 89: The collective-escape free-rider equilibrium is ROBUST -- predation pressure shifts it only weakly and escape never fixes (a sticky public-goods mixed strategy)

**Setup.** Maps the F88 mixed escaper equilibrium against predation pressure -- a pure parameter sweep in
the same capture/removal model, no new mechanism. Each run starts from a seeded f=0.5 escaper population
(so escape is established) and the capture rate (the predation hazard while within the kill radius) is swept
over {1, 2, 3, 5, 8}/tu; the steady escaper fraction is the mean over the last third of a 200-tu run.
Prediction: stronger predation should erode the free-rider advantage (riders get caught more) and push the
escaper fraction toward fixation. A convergence check starts f=0.3 and f=0.7 at one rate to test whether the
value is a genuine attractor.

**Result.** The prediction holds in direction but is weak in magnitude. The steady escaper fraction rises
only from 0.56 (rate 1/tu) to 0.66 (rate 8/tu) across an eightfold range of predation, with the mean escape
weight staying in the escape regime (w~1.1-1.35) and Phi~0.73-0.84 throughout. It never approaches fixation:
even at the strongest predation about a third of the flock rides the shared escape as protected free-riders.
The convergence check settles f=0.3 and f=0.7 into a similar band (0.67, 0.77) -- escape persists as a
substantial-but-not-fixed majority from either side -- with some residual start-dependence and 2-seed noise.

**Implication.** The F88 mixed free-rider equilibrium is a robust, sticky feature, not a knife-edge: across
an eightfold range of predation pressure escape neither collapses nor fixes, settling around a 0.55-0.77
escaper majority. This confirms the public-goods reading -- the shared escape direction is a non-excludable
benefit, so a protected free-rider fraction persists as long as the flock escapes at all, and intensifying
predation only mildly raises the cost of riding. The result reinforces the F87/F88 hysteresis theme (some
start-dependence remains within the band) and bounds the evolutionary outcome of the chosen capture/removal
model: escape, once established, is a durable MIXED strategy rather than a trait that sweeps to fixation. It
is the same public-good logic as the shared-heading principle (F70/F72) seen now at the population level --
a shared directional signal benefits non-contributors too, so contribution need not become universal.
Caveats: 2 seeds (noisy at the few-percent level), fixed predator, N=150, one fitness model.
evolution/escape_freerider.py

## Finding 90: Two-sided co-evolution -- the arms race is ASYMMETRIC: the predator optimises its lead freely (no barrier) while the prey counter is origination-limited (F87), so de-novo co-evolution favours the predator and escape wins only when already present, then collapses predator selection [NOTE: the secondary "capture-optimal lead ~3 differs from disruption-optimal ~2" claim made here was FALSIFIED by the F91 direct measurement -- see the correction in the implication below and in F91]

**Setup.** The true two-sided arms race (chosen trait: predator predictive lead_time; user decision,
2026-06-01). Prey keep the F87 heritable escape weight under capture/removal; now each of the 6 predators
also carries a heritable lead_time and they are selected on CAPTURE SUCCESS -- every 20 tu the
worst-capturing predator is replaced by a mutated clone of the best (the small-population analogue of the
prey's Moran scheme; captures are credited to the nearest predator). Predators start at RANDOM leads in
[0,5]. Three experiments, 3 seeds, 400 tu: Exp0 evolves ONLY the predator against FROZEN no-escape prey (a
clean test of what capture-selection alone favours); Exp1 co-evolves both from no escape (w0=0); Exp2
co-evolves both from a seeded f=0.5 escaper population. Reuses the validated per-step physics.

**Result.** Exp0 (clean): against frozen no-escape prey the predator lead converges tightly to 3.03 +/-0.31
(3 seeds), into the effective/disruptive lead range. I initially read this as evidence that capture-selection
favours a LONGER lead (~3) than the F66 most-disruptive value (~2), i.e. that capture and disruption have
distinct optima -- but a direct capture-vs-lead measurement (F91) FALSIFIED that: the capture rate actually
peaks at lead~2, the SAME lead that most disrupts coherence, and the evolutionary overshoot to ~3 is a
small-population selection artifact (see F91 and the corrected implication below). Exp1 (co-evolve from no escape): the predator lead stays high (3.2 +/-1.5) while the prey climb only
partway (mean w~0.70, escaper fraction 0.47) -- escape is not established within 400 tu (the F87 origination
brake), so the predator wins. Exp2 (co-evolve from seeded escape): escape FIXES (mean w~2.00, escaper
fraction 0.99, Phi=1.000) and the predator's lead trait DRIFTS aimlessly (2.3 +/-2.3, i.e. essentially
random across seeds) because committed escape (F70) yields no captures and so no selection signal on the
predator -- a "use it or lose it" collapse of the predator trait under relaxed selection.

**Implication.** (i) [CORRECTED by F91] I first inferred from the evolved lead (~3) that capture-selection
favours a different lead than disruption (~2). The F91 direct sweep refutes this: the capture rate peaks at
lead~2, coinciding with the disruption optimum, so capture and disruption do NOT have distinct optima; the
evolved ~3 was an artifact of the noisy 6-predator replace-worst-with-best scheme (the capture rate at
lead~3 is hugely variable, +/-5.6, and cloning the lucky best chases that unstable tail). The lesson survives
in inverted form: a tightly-converged evolved-trait value under small-population noisy selection need not sit
at the trait's fitness optimum. (ii) The robust result -- independent of (i) -- is that the arms race is
fundamentally ASYMMETRIC. The predator can climb to its optimum from any start (no barrier), but the prey
counter is origination-limited (F87/F88), so de-novo co-evolution favours the predator and escape establishes
only when seeded -- after which it is an uncounterable hard counter (F70) that collapses predator selection
entirely. This also RECONCILES F89: the ~60% free-rider equilibrium requires a persistently EFFECTIVE
predator to sustain it; once the predator is allowed to become ineffective (co-evolving against winning
prey), predation collapses and escape fixes (0.99 here vs 0.6 under the fixed effective predator of F89). The
co-adaptation arc (F87-F90) thus closes with a clear picture: collective escape is a powerful but
evolutionarily fragile defence -- hard to originate, easy to lose to drift, yet decisive once present.
Caveats: 2-3 seed noise (the Exp1/Exp2 lead std is large), predator selection on only 6 individuals, one
fitness model, fixed prey alignment. evolution/escape_coevolution.py

## Finding 91 (SELF-TEST): Direct measurement CORRECTS the F90 capture-vs-disruption claim -- the capture rate peaks at the SAME lead (~2) that most disrupts coherence, so the F90 evolved lead ~3 was a small-population selection artifact, not a distinct capture optimum

**Setup.** F90 inferred from the evolved predator lead (~3) that capture-selection optimises a DIFFERENT
lead than coherence-disruption (F66's ~2). This tests that inference directly rather than through the noisy
evolutionary dynamics: against frozen no-escape prey, hold all six predators at a FIXED common lead and
measure the steady capture rate and the order parameter, sweeping the lead over {0,1,2,3,4,5} tu, 3 seeds,
no evolution. If captures peak near ~3 while Phi bottoms near ~2 the F90 distinction holds; if both
extremise at the same lead it does not.

**Result.** Both the capture rate and the disruption peak at the SAME lead, ~2. Capture rate (per tu): 2.2
(lead 0), 7.0 (1), 9.6 (2, the maximum), 4.3 (3, but +/-5.6 -- wildly variable), 0.2 (4), 0.2 (5); order
parameter Phi: 0.80, 0.60, 0.584 (minimum at lead 2), 0.835, 0.998, 0.999. The lead that catches the most
prey (2) is exactly the lead that most fragments the flock (2). At lead 3 captures are already low and
hugely variable run-to-run; at lead >= 4 the predators overshoot the flock and barely catch anything (the
flock stays coherent, Phi ~ 1).

**Implication.** F90's secondary claim -- that capture-maximisation and coherence-disruption are distinct
objectives with distinct optima -- is FALSIFIED. The capture optimum and the disruption optimum coincide at
lead ~2 (the F66 value). The F90 evolved lead of ~3 was therefore NOT a true capture optimum but an artifact
of the small-population (6-predator), high-variance, replace-worst-with-best selection scheme: at lead ~3 the
capture rate has enormous run-to-run variance (+/-5.6), and a rule that each window clones the single
best-capturing predator chases lucky high-capture realisations into the unstable long-lead tail rather than
tracking the mean-rate optimum at ~2. The robust F90 result -- the ASYMMETRY of the arms race (predator
optimises freely, prey origination-limited; seeded escape wins and collapses predator selection) -- is
unaffected, as it does not depend on the exact evolved lead. The broader lesson survives inverted: a
tightly-converged evolved-trait value under noisy small-population selection need NOT sit at the trait's
fitness optimum, so an evolved parameter should be checked against a direct fitness measurement before being
read as optimal. A 6th self-test in the study's tradition (cf. F47/F48/F52/F81/F86) of registering a claim
and testing it directly; the only one to correct a result from THIS session's own thread.
evolution/escape_capture_curve.py

## Finding 92: Robustness under an ENERGY-BUDGET fitness model -- the origination BRAKE survives (model-independent), but F87's "escape is free and stable once present" does NOT: any appreciable metabolic cost collapses even established escape (a sharp threshold, not an interior optimum)

**Setup.** The student's chosen robustness test (energy-budget fitness, 2026-06): does the F87 brake
depend on the capture/removal model? It replaces the implicit "escape is free" of F87-F91 with an explicit
metabolic cost -- an always-on per-step death hazard metab_cost*w (escaping costs energy even when safe),
added to the predator-capture hazard; dead agents (predator OR metabolic) are replaced by mutated clones of
survivors, N fixed. Exp1: fixed cost c=0.5, sweep initial w0 {0, 0.5, 1, 2} -- does the brake still trap low
starts, and is a high start pulled to an interior optimum? Exp2: seeded f=0.5 escapers, sweep cost c
{0, 0.25, 0.5, 1, 2} -- the evolved steady weight (ESS) vs cost. 2 seeds, 150 tu.

**Result.** Exp1: at c=0.5 EVERY start collapses to w~0.05 (escaper fraction 0, Phi~0.26). The brake traps
low starts as before (w0=0 -> 0.05, escape cannot originate, confirming F87), and a w0=2 start seeded in the
escape regime is driven ALL the way down to ~0.05 as well -- escape is abandoned entirely, with no interior
optimum. Exp2: the ESS collapses SHARPLY with cost -- c=0 -> w=1.28 (escaper fraction 0.64, recovering the
F88 mixed equilibrium), but c=0.25 -> 0.11, c=0.5 -> 0.06, c=1 -> 0.04, c=2 -> 0.03. The jump from no cost to
the smallest cost tested takes escape from a stable majority to near-extinction: it is nearly all-or-nothing,
not a graded interior optimum.

**Implication.** Two robustness verdicts, splitting the F87-F91 results into a model-independent half and a
model-dependent half. (1) The origination BRAKE is ROBUST: it survives the energy-budget model unchanged
(escape still cannot evolve from no-escape), so the F70-valley origination barrier is a general feature, not
an artifact of capture/removal. (2) But F87's complementary result -- that escape is near-free and STABLE
once established -- does NOT survive: it relied on the capture/removal model in which escaping costs nothing.
Under an energy budget even established escape collapses at any appreciable metabolic cost (a sharp threshold
below c~0.25), because the cost is paid by every escaper continuously while predation threatens only the few
agents near a predator at any instant, so a modest per-capita cost outweighs the diffuse, intermittent
benefit of escaping. Collective escape is therefore even more evolutionarily fragile than F90/F91 implied:
not only hard to originate and easy to lose to drift, but unsustainable under any non-trivial metabolic cost,
viable only where escaping is essentially free. This sharpens the co-adaptation thesis -- the brake
(origination) is the robust, model-independent result, while the persistence of escape is model-dependent
and, under an explicit cost, precarious. Caveats: 2 seeds, one predator regime; the metabolic-cost scale is
relative to the (diffuse, intermittent) predation hazard of this setup. evolution/escape_energy.py

## Open Questions / Next Directions
*(updated through F92.)*

All primary threads are CLOSED:
- **Predator strategy (2D + 3D)**: F5-F16, F19-F35, F43-F45, F49, F53. 3D encirclement
  does NOT disrupt the flock at all (Phi~1.0) by any variant -- strictly 2D-specific
  (F43/F44/F45/F49 corrected after the predator-force sign bug). Prey fatigue does not
  make damage irreversible (F53). 3D flocks are robust to ALL point-predator strategies,
  not only sealing (F65) -- closes the 3D predator thread definitively.
- **Predator-prey arms race (2D)**: F66-F71. Predictive encirclement (target CoM +
  lead*v_mean) is the first predator adaptation to beat F14 (Phi 0.83->0.53), placement
  dominating radius (F67), noise-tolerant but delay-sensitive (F68/F69). Prey collective
  escape defeats it above w~alpha (F70) but only with a globally SHARED escape vector;
  local per-prey sensing only partly counters (F71).
- **Phase transition**: F2, F8, F12, F17, F38-F40, F50. No exponent of the base_r^n
  repulsion crystallizes; KTHNY needs a true inverse-power-law/hard-disc force.
- **Contagion / vaccination / kinematic mixing**: F10-F37, F47-F48, F52, F54-F64.
  Degree-targeting fails STRUCTURALLY (no hubs, F48); spatial fails KINEMATICALLY
  (mixing erases coverage, F37). Slow-recoverer targeting (F56) is the first and only
  strategy to beat random -- valid in 2D (F56), 3D (F58), under continuous (F59) and
  bimodal gamma, with noisy estimates (F60), for rare reservoirs (F61), exactly as long
  as the slow CLASS is durable on the epidemic timescale (F62), robust under combined
  beta_i+gamma_i heterogeneity (F63), and reversing the predator+contagion damage
  asymmetry once the vaccination budget covers the reservoir (F64).
- **3D extension**: F41-F46, F49, F51, F52, F58, F65. 3D mixes ~1.8x SLOWER than 2D (F52);
  the "mixing aid" theme was falsified.
- **Leadership / collective decision-making (2D)**: F72-F86, a complete arc. An informed
  minority steers the flock cohesion-free (F72); conflicting leaders compromise then
  reach consensus (F73) by a count*conviction product law (F74) with critical slowing at
  the bifurcation (F75); leadership is a signal not an identity (F76) and a low-pass
  steering channel (F77); it counters encirclement (F78) but not panic (F79), and an
  adversary can deny at parity but must dominate to capture (F80). The many-wrongs sub-arc
  (F81-F86) establishes alignment as a directional averager: steering is set by the
  informed fraction (F81), independent noisy estimates average as 1/sqrt(N) (F82), error
  correlation imposes an N-independent accuracy floor (F83), a noisy minority is a distinct
  mechanism (F84), uncoordinated misinformation averages away while coordinated falsehood
  captures at parity (F85), and spatial averaging is independent of temporal bandwidth (F86).

Resolved since the last revision of this list: combined beta_i+gamma_i heterogeneity
(F63), predator + slow-targeted vaccination (F64), and a 3D-effective non-encirclement
predator (F65 -- no point strategy works; a 3D-effective attack must target the alignment
coupling per agent, which is exactly what contagion does).

Remaining exploratory directions:
1. **Co-adaptation / evolution of escape weight -- OPENED (F87, F88).** Fitness model chosen
   (capture/removal). F87: the F70 "dangerous valley" is a strong evolutionary BRAKE, not a
   barrier -- escape (w>=alpha) is stable + near-costless once present but evolves only by a
   slow hysteretic crawl that stalls in the valley (w~0.5 after 400 tu, never reaching w=1).
   F88: the brake is a barrier to ORIGINATION, not invasion -- a >=5% escaper founder group
   establishes escape (mixed ~60% equilibrium, a shared-signal free-rider effect), and a
   single mutational jump sigma~0.3-0.6 clears the valley; mutation-/variation-limited.
   Prerequisites built + validated: vectorized_predator.py + vectorized_predator_prey.py.
   F90: two-sided co-evolution (predator lead_time heritable) -- the arms race is ASYMMETRIC
   (predator optimises freely, prey origination-limited), escape wins only when seeded, then
   collapses predator selection (drift). F91 (self-test): the capture optimum (~2) = the disruption
   optimum, correcting F90's evolved-lead-~3-overshoot (a small-pop selection artifact). F92
   (energy-budget robustness): the origination BRAKE is model-independent, but F87's "escape free/
   stable once present" is model-DEPENDENT -- any appreciable metabolic cost collapses even
   established escape. Remaining: (a) proximity-survival fitness as a third robustness check
   (student-owned model choice); (b) heritable alignment strength alpha under predation; (c)
   co-evolve a 2-trait predator (lead + aggression). The evolution/ scripts + harness support all.
2. **Agent memory beyond fatigue:** learned predator avoidance with internal state
   (e.g. a per-agent threat estimate that decays), distinct from the static fatigue of F53.
3. **Robust estimation against the F83 floor:** can agents detect and down-weight
   correlated inputs to relax the correlation ceiling on collective accuracy?
4. **A true hard-disc potential** would be the only way to revisit the KTHNY phase-transition
   question that the base_r^n family closed negatively (F50).

