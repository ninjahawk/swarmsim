# Findings -- Sandpiles / Self-Organized Criticality
Started 2026-06-14

Charbonneau, *Natural Complexity* (2017), Chapter 5. A **new chapter**, distinct
from the flocking work (F1-F92 in `findings.md`). To keep the two cleanly
separated, sandpile findings use their own **S-series** (S1, S2, ...).

The arc: validated 1-D core (S1-S2) -> rigorous 1-D critical exponents by
finite-size scaling (S3) -> a 2-D slope sandpile and the universality question
(S4: do 1-D and 2-D share exponents, and how do they compare to the canonical
abelian/BTW sandpile?) -> which SOC ingredient is necessary (S5, S7: bulk
conservation) -> a self-test of the universality result (S6) and the chapter's
falloff exercise (S8). Status: S1-S8 complete and pushed.

Headline: self-organized criticality as a *phenomenon* (scale-free avalanches
with no parameter tuning) is robust, but the critical *exponents* are not -- they
change with dimension and with the toppling rule, and the scale invariance is
exact only while the bulk conserves sand.

Code lives in `sandpile/`. Figures in `figures/sandpile_*.png`, run logs in
`outputs/sandpile_*.txt`.

---

## Index

- S1  Validated 1-D slope-sandpile reproduces the chapter's SOC signatures
- S2  Angle of repose: the MEAN slope sits ~16% below Zc (eps-independent); only
      the PEAK (pre-avalanche) slope approaches Zc as eps -> 0
- S3  1-D critical exponents by finite-size scaling: tau_E ~ 1.03, D_E ~ 2.0,
      D_T ~ 1.0; avalanches are quantized families, not simple fractals
- S4  The Grand Challenge: a 2-D slope sandpile is SOC but with DIFFERENT
      exponents from 1-D, and a different universality class from canonical BTW
- S5  Bulk conservation is necessary for true SOC: dissipation imposes a
      characteristic avalanche size and the cutoff stops scaling with system size
- S6  (self-test) The duration-exponent cross-check of S4 is INCONCLUSIVE -- the
      slope model's avalanches are too short-lived to pin tau_T; S4 stands on the
      size exponent
- S7  The conservation-necessity result (S5) transfers to 2-D: conservative
      cutoff scales as L^~2, dissipation suppresses it (qualitatively, over a
      limited size range)
- S8  (Exercise 2) Boundary "falloff" avalanches are scale-invariant but with a
      shallower exponent than bulk avalanches; only ~44% of avalanches reach the
      edge and the drained fraction shrinks with system size
- S9  (Exercise 5) An active-list, numba-compiled engine reproduces the 1-D and
      2-D models exactly (avalanche series bit-identical under shared forcing) and
      runs ~600x faster in 2-D -- the speedup that unlocks the larger lattices S6
      and S7 needed
- S10 (resolves S6) Redone at L up to 512 with the fast engine, the 2-D slope
      model's DURATION exponent is stable at tau_T ~ 0.56, far from BTW's ~1.22 --
      the convention-free duration measure now confirms the S4 universality split

---

## S1 -- Validated 1-D slope-sandpile reproduces the chapter's SOC signatures

**Question.** Before exploring anything new, does a faithful implementation of
the Charbonneau 1-D slope sandpile (eqs 5.1-5.10) reproduce the predictable
outcomes in the chapter? This is the "check limiting cases / known results"
validation step.

**Model.** Continuous nodal heights `S[0..N-1]`; slow forcing adds a grain
`s ~ U(0,eps)` at a random node only on fully-stable iterations (stop-and-go
timescale separation); a nodal pair is unstable when its slope
`z = |S[j+1]-S[j]| >= Zc`; the redistribution rule (eq 5.4) halves an unstable
pair's slope, moving `z/4` of sand downslope; synchronous update; open right
boundary `S[N-1]=0`, wall on the left. Implemented vectorized per iteration in
`sandpile/sandpile1d.py`; validation in `sandpile/validate1d.py`.

**Evidence.**
- **Conservation (eq 5.6).** Over a full instrumented run, grains added minus
  sand drained at the open boundary equals the final pile mass to a residual of
  **3.3e-11** -- redistribution moves sand without creating or destroying it,
  and the boundary is the only sink.
- **Slope halving.** A single isolated unstable pair has its slope reduced by
  exactly a factor of 2 (self-test). A node shared by two unstable pairs topples
  both ways at once (a larger reduction) -- correct synchronous behaviour.
- **Mass series (fig 5.4A).** From an empty N=100 pile, total mass grows
  linearly through a transient then saturates to a statistically stationary
  plateau (M ~ 2.1e4), interrupted by intermittent avalanche discharges
  (`figures/sandpile_validate_mass.png`).
- **E-T wedge (fig 5.6).** N=1000 avalanches fall inside a wedge bounded by
  slope +1 (line avalanches, E ~ T) and +2 (wedge avalanches, E ~ T^2) in a
  log-log plot (`figures/sandpile_validate_ET.png`).
- **Power-law PDFs, size-independent slope (fig 5.7).** The avalanche-energy PDF
  is a power law whose logarithmic slope is essentially independent of lattice
  size: **PDF(E) slope = -0.99 / -1.00 / -1.01 for N = 100 / 300 / 1000**; the
  distribution simply extends farther right as N grows
  (`figures/sandpile_validate_pdf.png`). This is the defining SOC signature --
  scale-free avalanches that emerge with no fine-tuning of a control parameter.
  (Our slope ~ -1.0; the book quotes ~ -1.09. The small difference is within
  fit-window sensitivity; the scientifically important fact -- N-independence --
  is robust.)
- **Initial-condition independence (Exercise 3).** Started from an empty pile, a
  triangular pile at repose, and a uniformly-loaded pile (N=100, 3M iterations),
  the lattice converges to the same stationary slope (**4.213 / 4.209 / 4.211**)
  and the same energy-PDF slope (**-0.98 / -1.00 / -0.96**) -- the SOC state is an
  attractor independent of how the pile started.

**Interpretation.** The implementation is sound: it conserves sand exactly,
reaches the SOC attractor from any initial condition, and reproduces the
chapter's three quantitative signatures (the stationary plateau, the E-T wedge
geometry, and N-independent power-law avalanche statistics). This is the trusted
foundation for the exponent and 2-D work that follows.

---

## S2 -- Angle of repose: mean slope ~16% below Zc; only the peak slope -> Zc as eps -> 0

**Question.** The chapter states the stationary slope settles a few percent
*below* the critical slope Zc (it quotes ~7% for eps=0.1), approaching Zc only as
eps -> 0, because stochastic forcing tips some pairs over before the whole pile
reaches Zc. Does our pile reproduce this, quantitatively?

**Evidence (N=100, time-averaged stationary slope).**

| eps  | stationary slope | deficit (Zc-slope)/Zc |
|------|------------------|-----------------------|
| 0.01 | 4.089            | 18.2%                 |
| 0.05 | 4.176            | 16.5%                 |
| 0.10 | 4.207            | 15.9%                 |
| 0.50 | 4.232            | 15.4%                 |
| 1.00 | 4.257            | 14.9%                 |

Two honest discrepancies with the chapter, neither of which is an implementation
error:

1. **Magnitude.** Our deficit at eps=0.1 is ~16%, not the book's stated ~7%. But
   the book's *own* mass figure (5.4A: M ~ 2.1e4 at N=100) back-calculates via
   mass = slope * N(N-1)/2 to slope ~4.24, i.e. ~15% below Zc -- which agrees
   with us, not with the "7%" in the text. We take ~16% to be correct for this
   model and read the book's "7%" as a loose or differently-defined figure.

2. **eps-trend.** The book says the slope approaches Zc as eps -> 0 (deficit ->
   0). We find the opposite weak trend: the deficit *widens* slightly as eps
   shrinks (14.9% at eps=1 -> 18.2% at eps=0.01). The PDF slopes at the eps
   extremes are noisy (-1.36, -1.61), a sign those runs are under-converged, so
   the trend is not clean; but it is clearly not closing toward zero.

**Resolution (`sandpile/repose_peak.py`, `figures/sandpile_repose_peak.png`).**
Measuring, in the stationary state, both the spatial-MEAN bond slope and the
spatial-MAX bond slope as functions of eps settles it cleanly:

| eps  | mean slope | mean deficit | max slope | max excess over Zc |
|------|-----------|--------------|-----------|--------------------|
| 0.01 | 4.131     | 17.4%        | 4.650     | -0.350             |
| 0.10 | 4.199     | 16.0%        | 4.934     | -0.067             |
| 0.20 | 4.202     | 16.0%        | 5.123     | +0.123             |
| 1.00 | 4.230     | 15.4%        | 5.584     | +0.584             |

The **max** bond slope tracks Zc: its excess over Zc falls roughly linearly with
eps (+0.58 at eps=1 down to -0.35 at eps=0.01), crossing Zc near eps ~ 0.13. The
**mean** slope is essentially flat at ~16% below Zc across two decades of eps.

**Interpretation.** The redistribution rule *halves* an unstable bond's slope,
removing far more sand than the small threshold overshoot a single grain
produces. So the time-mean slope -- the actual angle of repose -- is dragged well
below Zc and stays there regardless of grain size (our ~16% floor). The book's
statement that "the slope approaches Zc as eps -> 0" is correct only for the
**peak** slope: the overshoot a grain can produce is O(eps) and vanishes as
grains get finer, so the steepest bond the pile sustains approaches Zc. Mean and
peak are two different quantities with opposite eps-behaviour; conflating them is
what makes the chapter's ~7% claim look inconsistent with its own fig-5.4A mass.
This is a small but genuine clarification of the chapter.

---

## S3 -- 1-D critical exponents by finite-size scaling

**Question.** The chapter shows the avalanche PDFs are power laws whose slope is
size-independent. Made quantitative: what are the critical exponents, and does a
single set describe every lattice size (a data collapse)?

**Method (`sandpile/fss1d.py`).** The SOC ansatz for an avalanche observable x
(energy E, duration T) is P(x) = x^{-tau_x} g(x/x_c) with cutoff x_c ~ N^{D_x}.
We read tau_x from the power-law slope of the largest lattice, estimate the
cutoff robustly from the moment ratio <x^2>/<x> ~ x_c, fit D_x from x_c vs N over
N = 64, 128, 256, 512, 1024, and verify the rescaled curves x^{tau_x} P(x) vs
x/N^{D_x} collapse (`figures/sandpile_fss.png`).

**Evidence.**
- **Energy:** tau_E = **1.03** (the PDF is close to the marginal 1/E; book quotes
  ~1.09). Cutoff E_c ~ N^{1.94} from the moment ratio; the largest avalanche
  scales as E_max ~ N^{2.01} directly (2.85e3 -> 7.44e5 as N 64 -> 1024), so
  **D_E ~ 2.0**. The rescaled energy PDFs collapse onto one curve.
- **Duration:** the largest avalanche spans the lattice, T_max ~ N^{1.01} (125 ->
  2045 as N 64 -> 1024), so **D_T ~ 1.0** (moment ratio gives 0.90, slightly low
  -- moment estimators converge slowly for the shallow duration distribution).
  tau_T ~ 0.6.

**Interpretation.**
- **D_E ~ 2 and D_T ~ 1 are geometric.** The biggest avalanche reaches across the
  whole lattice, so its duration scales as N (one-node-per-iteration propagation,
  S1's E-T wedge). A lattice-spanning avalanche of the wedge type has E ~ T^2 ~
  N^2, fixing the energy cutoff at N^2.
- **The avalanches are quantized families, not simple fractals.** S1's E-T wedge
  (bounded by slopes +1 and +2) means E and T are *not* related by a single power
  law, so the usual SOC scaling relation tau_T = 1 + (tau_E-1) D_E/D_T does not
  apply (it would predict tau_T ~ 1.06, far from the measured 0.6). This is
  consistent with the chapter's own remark that E, P, T are "correlated only in a
  statistical sense": the 1-D model's locality forces avalanches into discrete
  line/wedge families rather than a single self-affine family.
- These exponents (tau_E ~ 1.0, D_E ~ 2.0) are the **1-D baseline** for the
  universality question: do the 2-D bond-slope sandpile's exponents differ, and
  how do they compare to the canonical abelian (BTW) sandpile?

---

## S4 -- The Grand Challenge: a 2-D slope sandpile, and the universality question

**Question (Charbonneau Exercise 6).** Generalize the slope model to two
dimensions, confirm it self-organizes to a critical state with power-law
avalanches, and ask: are the exponents the same as in 1-D? And -- our own
extension -- is this slope-based sandpile in the same universality class as the
canonical abelian (Bak-Tang-Wiesenfeld) sandpile?

**Model (owned design choice; `sandpile/sandpile2d.py`).** The 1-D model is a
*bond* model: every adjacent pair carries a slope and an unstable bond halves it.
The 2-D generalization keeps that exact rule but lets bonds run in both lattice
directions -- each site links to its right and upper neighbours by an x-bond and
a y-bond, every bond obeying the identical 1-D pair rule -- with open (draining)
boundaries on all four edges. This is chosen over the book's hinted "2x2-block
slope" precisely because it reduces to the 1-D model exactly when one dimension
is collapsed: the *local dynamics are unchanged*, so a 1-D-vs-2-D exponent
difference is a clean effect of dimensionality alone, not of also changing the
rule. (A toppling-count "size" series S = number of bond topplings was added so
we can compare to the BTW size exponent, which counts site topplings.)

**Evidence -- the 2-D slope model is SOC (`sandpile/fss2d.py`,
`figures/sandpile_fss2d.png`).** Run from a pyramid initial pile over L = 32, 48,
64, 96, 128, it reaches a stationary state with a mean bond slope ~2.5 (about 50%
below Zc=5 -- a much larger deficit than 1-D's 16%, because four neighbours give
each site more relaxation channels). Avalanche PDFs of energy E and size S are
power laws whose rescaled curves collapse:

| observable | 1-D exponent | 2-D exponent |
|------------|--------------|--------------|
| energy tau_E   | 1.03 | **0.87** |
| size  tau_S    | (n/a) | **0.89** |
| duration tau_T | 0.60 | 0.67 |
| energy cutoff D_E | 2.0 | ~2.2 |

**Evidence -- canonical BTW under the same pipeline (`sandpile/btw_compare.py`,
`figures/sandpile_btw.png`).** The standard 2-D abelian sandpile (integer
heights, site topples at h>=4 shedding one grain to each neighbour, open edges)
measured with the identical log-binned-PDF / FSS analysis over L = 32..128 gives
a clean 5-decade power law and a tight data collapse with **tau_S(BTW) = 1.14**,
D_S ~ 2.55 -- consistent with the literature value ~1.2 (our slightly lower
number is the well-known finite-size/multifractal effect at these L).

**Two conclusions.**

1. **1-D and 2-D slope sandpiles are NOT in the same universality class.** The
   energy exponent drops from tau_E = 1.03 (1-D) to 0.87 (2-D) under the
   *identical* local toppling rule -- so the change is purely dimensional. (D_E
   stays ~2 in both, set by the same geometry: the biggest avalanche spans the
   lattice.)

2. **The slope (gradient) rule is NOT in the BTW universality class.** At the
   same dimension (2-D) and under the same measurement, the slope model's size
   exponent tau_S = 0.89 differs clearly from BTW's 1.14. So *how* a site decides
   to topple matters: toppling on a height *difference* exceeding a threshold
   (slope/gradient model) yields a different critical behaviour than toppling on
   absolute *height* (BTW). Both are SOC -- both produce scale-free avalanches
   with no parameter tuning -- but they are distinct critical classes.

**Honest caveats.** (a) Exponents come from log-binned power-law fits over a
modest L range (32-128) with finite-size curvature; the digits carry ~+/-0.05-0.1
uncertainty. The *differences* (1.03 vs 0.87 vs 1.14) are several times that, so
the universality conclusions are robust even though the precise values are not
nailed down. (b) The slope-model size counts *bond* topplings and BTW counts
*site* topplings; these differ by an O(1) factor that cannot change a power-law
exponent, so the comparison is valid, but a duration-based cross-check (tau_T,
defined identically for both) would remove the objection entirely -- attempted in
S6, and found inconclusive (the slope model's duration range is too short to pin
tau_T), so S4 rests on the size exponent. (c) The slope-model collapses are
noisier than BTW's because it produces ~5-10x fewer avalanches per run; longer
runs would tighten them.

**Why this is interesting.** It gives a direct, method-matched answer to the
Grand Challenge -- the 2-D indices are *not* the same as 1-D -- and adds a result
the chapter does not address: the textbook's slope sandpile and the historically
canonical BTW sandpile, often discussed under the one banner of "the sandpile
model", are actually in different universality classes. The SOC *phenomenon*
(scale-free avalanches as a dynamical attractor) is robust to the toppling rule;
the SOC *exponents* are not.

---

## S5 -- Bulk conservation is necessary for true SOC

**Question.** Charbonneau lists four ingredients that appear sufficient for SOC:
a system that is open and dissipative, slowly forced, with a local threshold
instability and local relaxation. The slope sandpile is *conservative in the
bulk* -- the redistribution moves sand without destroying it, and sand leaves only
at the open boundary. Is that bulk conservation actually necessary, or incidental?
(This is the controversial ingredient in the Olami-Feder-Christensen earthquake
model.)

**Method (`sandpile/dissipation.py`).** We break conservation with a tunable bulk
dissipation d (sandpile1d.run_sandpile(dissip=d)): a toppling higher node still
sheds |slope|/4, but the lower node receives only (1-d)*|slope|/4 and the rest is
destroyed. Same threshold, same relaxation, same slow forcing -- only conservation
is removed. The decisive criticality test is not whether avalanches still occur
but whether their cutoff still SCALES with system size: a truly critical system
has no characteristic avalanche size (cutoff ~ N^2, S3), whereas a finite
correlation length makes the cutoff SATURATE to an N-independent value. We measure
the energy cutoff (moment ratio <E^2>/<E>) over N = 100, 200, 400, 800 at
d = 0, 0.02, 0.05, 0.10, 0.20.

**Evidence (`figures/sandpile_dissipation.png`).**

| dissipation d | cutoff scaling exponent (d log cutoff / d log N) | cutoff at N=800 |
|---------------|--------------------------------------------------|-----------------|
| 0.00 (conservative) | **2.00** | 2.7e5 |
| 0.02 | 0.77 | 1.2e4 |
| 0.05 | 0.94 | 5.3e3 |
| 0.10 | 0.65 | 1.2e3 |
| 0.20 | 0.51 | 4.2e2 |

The conservative case scales as N^2.00, exactly the critical behaviour of S3.
Every dissipative case has its cutoff-scaling exponent slashed to below 1 and its
absolute cutoff suppressed by one to three orders of magnitude, and the
avalanche-energy PDF is truncated at a characteristic size that shrinks as d
grows (right panel). The stronger the dissipation, the shorter the
dissipation-set correlation length and the smaller the largest possible
avalanche.

**Interpretation.** Bulk conservation is necessary for true, scale-free SOC in
this model. With conservation, the only sink is the boundary, so the only length
that can cut off an avalanche is the system size N -- hence the N^2 cutoff and
genuine scale invariance. Break conservation and sand can vanish anywhere; this
introduces a finite correlation length (a typical distance a disturbance travels
before being dissipated away), which becomes the characteristic avalanche size and
destroys scale invariance. The system is then at best *approximately* critical,
power-law only up to the dissipation scale. This singles out conservation among
the four SOC ingredients as the one that makes criticality exact rather than
approximate, and it reproduces -- in the simplest possible sandpile -- the
non-conservation sensitivity that makes the OFC earthquake model perennially
debated.

**Honest caveat.** Over our N range (100-800) the dissipative cutoff-scaling
exponents land at ~0.5-0.9 rather than cleanly at 0. That is expected: for weak
dissipation the correlation length is large, so N must exceed it before the cutoff
visibly saturates, and the moment-ratio estimator is noisy. The qualitative
dichotomy is unambiguous (d=0 gives exponent 2.0 and orders-of-magnitude larger
cutoffs; every d>0 gives a strongly suppressed, sub-N^2 cutoff), but pinning the
asymptotic value to 0 would need larger lattices, especially at d=0.02.

---

## S6 -- (self-test) The duration-exponent cross-check of S4 is inconclusive

**Motivation.** S4's universality verdict rests on the avalanche-SIZE exponent,
where the slope model counts bond topplings and BTW counts site topplings (an
O(1) definitional difference). To remove that objection, S4 flagged a cleaner
discriminator: avalanche DURATION, the number of parallel relaxation sweeps,
which is defined identically for both models. This is that test
(`sandpile/duration_compare.py`).

**Evidence (`figures/sandpile_duration_compare.png`).**

| model | tau_T (L=64) | tau_T (L=128) |
|-------|--------------|---------------|
| 2-D slope | 1.43 | 0.80 |
| 2-D BTW   | 1.15 | 1.23 |

BTW's duration distribution is a clean power law over three-plus decades and its
exponent is stable across size (tau_T ~ 1.2, consistent between L=64 and 128). The
slope model's is not: its duration spans barely two decades (the longest
avalanche lasts only ~125 sweeps even at L=128, because a 2-D slope avalanche
propagates at most ~L sweeps and the model produces far fewer avalanches), and the
fitted exponent swings from 1.43 to 0.80 between the two sizes (and was 0.67 in
the S3-style fss2d run). A quantity that changes by a factor of two between
adjacent lattice sizes is not a measured exponent.

**Verdict.** The duration cross-check is **inconclusive**: not because the two
duration exponents agree, but because the slope model's tau_T cannot be pinned
down at accessible lattice sizes. So it can neither confirm nor refute S4. The
size-based comparison (tau_S = 0.89 slope vs 1.14 BTW, each from distributions
spanning 3-5 decades) remains the reliable evidence that the two models are in
different universality classes. A decisive duration comparison would require 2-D
slope lattices far larger than L=128 to extend the duration range, which is beyond
what the pure-Python loop reaches here.

**Why record a negative result.** Reporting this honestly -- a planned check that
did not work, with the reason it did not -- is the same discipline as the flocking
self-tests (F47, F48, F91): test your own caveat, and if the test is too weak to
decide, say so rather than read a number off a bad fit. The auto-generated verdict
in the script ("durations agree") is exactly the trap, and it is wrong, because it
averages an unstable fit.

**Update (resolved by S10).** The limitation named above -- "would require 2-D
slope lattices far larger than L=128 ... beyond what the pure-Python loop reaches"
-- is exactly what the active-list engine (S9) removes. At L = 64-512 with ~2e5
avalanches per size, the slope tau_T is stable at ~0.56 and clearly differs from
BTW's ~1.22 (S10). So the duration cross-check is no longer inconclusive; the
record here stands as the honest account of why it could not be settled at L<=128.

---

## S7 -- Conservation-necessity (S5) transfers to two dimensions

**Question.** S5 established in 1-D that bulk conservation is necessary for true
SOC. A robust result should not be a 1-D accident. Does it transfer to the 2-D
bond-slope sandpile? (Same robustness instinct as the flocking thread's
cross-dimension checks.)

**Method (`sandpile/dissipation2d.py`).** The 2-D model gained the same dissip
parameter (lower node of each unstable bond receives only (1-dissip) of the shed
sand; conservative path kept bit-identical via a branch on dissip==0). We measure
the energy-cutoff scaling with L over L = 32, 48, 64, 96 at d = 0, 0.05, 0.20.

**Evidence (`figures/sandpile_dissipation2d.png`).**

| dissipation d | cutoff-scaling exponent (d log cutoff / d log L) |
|---------------|--------------------------------------------------|
| 0.00 (conservative) | **2.16** |
| 0.05 | 1.46 |
| 0.20 | 1.21 |

The conservative 2-D cutoff scales as L^2.16, close to the geometric ~L^2 of a
lattice-spanning avalanche -- genuine criticality, as in S4. Adding bulk
dissipation monotonically suppresses the cutoff-scaling exponent (2.16 -> 1.46 ->
1.21) and truncates the energy PDF at a smaller characteristic size as d grows.
The direction is the same as 1-D (S5): conservation gives size-scaling
criticality, dissipation removes it.

**Interpretation and honest limitation.** The conservation-necessity result
*transfers* to 2-D qualitatively. It is weaker quantitatively than the 1-D
demonstration: the dissipative slopes drop to ~1.2-1.5 rather than toward 0,
because the accessible 2-D lattice range is only about half a decade (L=32 to 96,
limited by the O(L^2) cost of the pure-Python loop) whereas 1-D reached a full
decade (N=100 to 800). Over so short a range the saturation cannot fully develop,
especially since the conservative slope itself (2.16) means the dynamic range of
the cutoff is large. So the 2-D evidence is consistent with S5 and shows the same
monotonic suppression, but a clean 2-D saturation (slope -> 0) would need larger
lattices via the list-based / compiled speedup the chapter suggests in Exercise 5.

---

## S8 -- Boundary "falloff" avalanches vs bulk avalanches (Exercise 2)

**Question (Exercise 2).** Track the mass that actually falls off the open edge
as its own series, distinct from the toppled-mass (bulk) series. Are falloff
avalanches scale-invariant? How does falloff energy correlate with bulk energy?

**Method (`sandpile/falloff.py`).** The 1-D run now records, each avalanche
iteration, the mass evacuated at node N-1 before it is zeroed (a 'falloff'
series). A subtlety: applying the section-5.4 run-detector to the raw falloff
series is not meaningful, because the boundary drains in many short bursts within
a single bulk avalanche (~60 bursts per boundary-reaching avalanche), fragmenting
it into thousands of one-iteration events. The meaningful unit is falloff PER
BULK AVALANCHE: for each avalanche (run of disp>0), sum the mass that left the
edge during it. Measured at N=300 and N=1000.

**Evidence.**
- **Only ~44% of avalanches reach the boundary** (44% at N=300, 45% at N=1000);
  the majority are interior avalanches that rearrange sand downslope but evacuate
  nothing. This is the S1/S3 wedge picture: most avalanches die before the edge.
- **Falloff avalanches are scale-invariant** -- the per-avalanche falloff-energy
  PDF is a power law with an N-independent slope (-0.63 at N=300, -0.65 at
  N=1000). So the boundary-evacuation process is itself self-similar.
- **But with a different exponent than the bulk:** the falloff slope (~ -0.64) is
  clearly shallower than the bulk toppled-energy slope (~ -1.02 / -1.01). Falloff
  and bulk avalanching are distinct self-similar processes, not the same
  distribution rescaled.
- **Falloff energy correlates strongly with bulk energy** on boundary-reaching
  avalanches (log-log correlation 0.91 at N=300, 0.96 at N=1000): a bigger
  avalanche evacuates more sand. The scatter (figure, right) shows a floor of
  avalanches that just touch the edge and drain a single quantum, rising to the
  lattice-spanning avalanches that drain the most.
- **The evacuated fraction is tiny and shrinks with N:** mean falloff/bulk energy
  ratio 0.023 at N=300, 0.005 at N=1000. Most toppling is internal rearrangement;
  only a thin surface layer leaves -- exactly the chapter's remark that even a
  large avalanche lowers the total mass by only ~0.2%, and that the fraction falls
  as the pile grows.

**Interpretation.** Boundary evacuation in a SOC sandpile is itself a scale-free
process, but a softer one than bulk toppling (shallower exponent): large
evacuations are relatively more common among falloff events than large topplings
are among bulk events, because reaching the boundary already selects for big,
spanning avalanches. The two are tightly coupled (high correlation) yet
statistically distinct (different exponents), and the bulk does the overwhelming
majority of the "work" (sand motion) while only a sliver actually exits -- the
hallmark of a system whose interior is frozen at the angle of repose with all the
action in a thin avalanching surface layer.

---

## S9 -- (Exercise 5) An active-list, compiled engine, validated bit-for-bit

**Motivation.** The reference engines (`sandpile1d.py`, `sandpile2d.py`) rescan
every nodal pair / bond on every temporal iteration, so a run costs O(N) per
iteration in 1-D and O(L^2) in 2-D regardless of how little of the lattice is
moving. That ceiling is what left S6 inconclusive (durations needing L >> 128) and
S7 quantitatively weak (a 2-D size range of only half a decade). Charbonneau's
Exercise 5 names the cure: keep an explicit list of the currently active sites and
touch only those plus their immediate neighbours each step.

**Method (`sandpile/sandpile_fast.py`).** A list-based engine for both 1-D and 2-D,
with the inner loop JIT-compiled by numba. The model is unchanged -- same
synchronous update, same boundaries, same stop-and-go forcing, same optional bulk
dissipation. Only the bookkeeping differs: a quiet loading step touches O(1) pairs
(just those around the new grain), and an avalanche step touches O(active bonds)
rather than O(lattice), by rebuilding the candidate list from only the bonds
adjacent to a node that actually moved. Each step therefore costs what the physics
costs, so the whole run scales with the total number of topplings instead of with
lattice size times iterations.

**Validation -- the important part.** To prove the new engine is the same model
and not merely a similar one, both reference engines gained an optional `forcing`
argument (additive; default reproduces the original RNG path exactly) so the fast
and reference engines can be driven by one identical stream of loading events. Run
that way they are bit-for-bit identical:

| test | result |
|------|--------|
| 1-D avalanche series (disp), shared forcing | max abs diff ~1e-13 |
| 1-D falloff series and final state | exactly 0 difference |
| 1-D avalanche count and every duration | identical |
| 2-D avalanche series (disp) and toppling-count (act) | ~1e-15 / exactly 0 |
| 2-D final state, counts, durations | identical |
| dissipative path (d = 0.05, 0.20), 1-D and 2-D | max abs diff ~1e-14 |

(The only quantity that is not bit-identical is the per-iteration total mass, which
the fast engine accumulates incrementally rather than re-summing the lattice; it
agrees to ~1e-9 floating-point accumulation error and is not used for any
exponent.) The avalanche STRUCTURE -- which bonds topple, in what order, for how
long -- is identical, so any result computed from the avalanche series is
unchanged.

**Speedup (steady state, triangle / pyramid IC).**

| dim | size | reference | fast | speedup |
|-----|------|-----------|------|---------|
| 1-D | N=256  | 9.6 s  | 0.07 s | ~135x |
| 1-D | N=1024 | 23 s   | 0.68 s | ~34x  |
| 2-D | L=64   | 14.5 s | 0.03 s | ~536x |
| 2-D | L=128  | 29 s   | 0.05 s | ~617x |

The 2-D gain is the largest because the reference there pays O(L^2) per iteration
while almost all of the lattice is idle between localized avalanches. This is the
enabling result for S10 (and for any future larger-lattice 2-D work); the science
it unlocks, not the engineering, is the point.

---

## S10 -- (resolves S6) The 2-D duration exponent, settled at large L

**Question.** S6 tried to confirm the S4 universality split (slope vs canonical
BTW) with the avalanche DURATION -- the number of parallel relaxation sweeps,
defined identically for both models, hence free of the bond-vs-site size-counting
caveat. It failed at L <= 128: the slope model's duration spanned barely two
decades and the fitted exponent swung from 1.43 to 0.80 between sizes. With the
S9 engine the slope model reaches L = 512 with ~2e5 avalanches per size, which is
exactly what S6 said it needed.

**Method (`sandpile/duration_fss2d.py`).** Measure tau_T by finite-size scaling
for the slope model at L = 64, 128, 256, 512 (fast engine) and for BTW at
L = 48, 64, 128 (same pure-Python pipeline as S4/S6), under one matched
log-binned-PDF fit over the window [8, 0.3 T_max]. The lower bound skips the steep
small-T head that comes from duration quantization at T = 1, 2, 3; the upper bound
skips the noisy extreme tail. The slope-model data collapse is then tested
directly.

**Evidence (`figures/sandpile_duration_fss.png`).**

| model | tau_T by size | mean tau_T | duration cutoff D_T (Tc ~ L^D_T) |
|-------|---------------|-----------|----------------------------------|
| 2-D slope | 0.58, 0.55, 0.57, 0.54 (L=64..512) | **0.56** | 1.07 |
| 2-D BTW   | 1.14, 1.27, 1.26 (L=48..128)       | **1.22** | 1.43 |

The slope model's tau_T is now stable -- it varies by less than 0.04 across a
factor of eight in lattice size, where at L<=128 it had swung by a factor of two.
The duration PDFs collapse onto a single curve under T^0.56 P(T) vs T / L^1.07 over
the scaling region (figure, top right). BTW's exponent is the familiar ~1.2,
stable as before.

**Verdict.** tau_T(slope) = 0.56 and tau_T(BTW) = 1.22 differ by 0.67, more than
twenty times the per-fit scatter (~0.03). The convention-free duration measure now
agrees with the size measure: the slope rule and the BTW rule are in different
universality classes. S6 is **resolved** -- not by overturning its honest "cannot
tell at L<=128" but by removing the size limitation that caused it. As a bonus the
two models even differ in HOW the duration cutoff grows with size (slope L^1.07, a
duration that scales with the linear lattice as a ballistically spreading front;
BTW L^1.43, the larger dynamic exponent expected of the abelian model), a second,
independent signature of the class difference.

**Caveat (own it).** BTW's D_T = 1.43 comes from only three lattice sizes (48-128)
and its T_max is sensitive to rare giant avalanches, so that number is the softest
in the table; the tau_T values, which come from the body of the distribution over
3-5 decades, are firm. The slope tau_T ~ 0.56 itself depends on excluding the
quantized small-T head, a window choice applied identically to both models.
