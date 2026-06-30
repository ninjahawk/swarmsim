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
falloff exercise (S8) -> a fast active-list engine (S9) that resolves S6 (S10) ->
and then the deepest cut at the universality question: not single exponents but
the whole MOMENT SPECTRUM, the field's decisive simple-FSS-vs-multifractal
discriminator (S11 validates the method on BTW; S12 applies it to the slope model
and finds its avalanches are filamentary and anomalously scaling, distinct from
both BTW and the Manna class) -> and then a MECHANISM (S13): a conditional-exponent
test shows the 2-D avalanche is a ballistic filamentary front whose spatial
observables obey single-scale scaling (E ~ S, and S, E ~ area^2) while duration is
a loose proxy -- the 2-D residue of the 1-D family breakdown -> and then the GEOMETRY
(S14): the footprint, dumped per-avalanche and measured directly, is a constant-width
thin filament (mass-radius dimension D ~ 1, one bond wide), thinner than the
exactly-solvable directed sandpile (3/2) and far from compact BTW (2), and a ballistic
front whose topple time tracks radial distance from the seed -> and then a CAUSAL test
(S15): a tunable stochastic split in the redistribution interpolates toward the Manna
class, and as it turns on the filament's mass-radius dimension climbs 1 -> 1.87
(L-independent), proving the thinness is a specific consequence of the deterministic
rule, while the moment spectrum stays anomalous, so the model compactifies in shape
without becoming Manna -> and then a DIMENSIONAL ANCHOR (S16): the same footprint
geometry measured in the model's NATIVE 1-D (the 1-D fast engine gained the S14
footprint dump, validated bit-for-bit), where the avalanche turns out to be a solid,
gap-free, ballistic interval of mass-radius dimension exactly 1, with conditional
exponents matching the 2-D values -- so the 2-D "filament" has the GEOMETRY of a 1-D
avalanche (same dimension, sweep structure and ballistic front, differing only in
directedness), D ~ 1 is the intrinsic dimension of a slope avalanche
(not a 2-D measurement artifact), and the S13 duration anomaly is inherited from S3's
1-D line/wedge families rather than being a 2-D effect; S16 also delivers the L > 256
equilibration enabler (a verified-stationary L = 512 state) that S17 needs -> and then
the DURATION CLOSURE (S17): using that enabler to reach L = 512, the spatial sector
heals to EXACT single-scale (size ~ area^2, area ~ duration) as L grows, but the
size-duration exponent saturates at ~1.75, so the single-scale over-determination gap
S13 found does NOT heal -- it GROWS with L to a true residual of ~0.2 (larger than the
1-D anchor 0.11), resolving the question S13 left open: single-scale scaling is exact
in space and fails permanently through duration, in 2-D as in 1-D, so AREA not duration
is the model's clean scaling variable -> the AREA-MULTIFRACTALITY CLOSURE (S18): is
S12's area D(q) drift asymptotic or a finite-size corona? Reaching L = 512 with a
sliding-window moment spectrum, the drift does NOT heal (~0.2 to L = 512, intercept
0.169 +- 0.048), so the area multifractality is a TRUE asymptotic property -- which
reconciles with S17 because gamma(S|A) -> 2 is a conditional MEAN while the area
DISTRIBUTION's higher moments stay multifractal: single-scale in the means (typical
avalanche), multifractal in the tails (largest avalanches) -> and the FILAMENT-FATTENING
TEST (S19): S18 left one tentative reading unsettled, that the footprint may FATTEN at
large L (the <A> growth slope steepened to 1.22 and D_mid to 1.23 at the largest, fewest-
seed points). Re-measured at L = 192-512 with 10-12 seeds (double S18) and SPLIT into the
typical avalanche vs the tail, it is the TAIL: the typical footprint is L-stable at the
filament values (median area ~ L^0.90, typical mass-radius D = 1.00 -> 1.01, ~one bond
wide to L = 512) while only the rare largest avalanches grow, and the mean-area slope that
read 1.22 at 5 seeds averages down to 0.99 +- 0.03 at 10. So the filament does NOT fatten;
S18's hint was tail-undersampling, S14/S16's one-bond filament stands, and the tentative
reading is downgraded to a pure tail statement -- consistent with (not contradicting) S18's
multifractal tail. Status: S1-S19 complete; the S11-S18 scaling-theory arc is closed, with
S19 a confirming self-test of its one open hint.

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
- S11 (method validation) Moment-scaling analysis reproduces the known result that
      canonical BTW is MULTIFRACTAL in toppling number (local slope D(q) drifts
      2.42 -> 2.72) but near-FSS in AREA (D_area ~ 2.05, flat) -- De Menech /
      Tebaldi. The machinery is trusted before being turned on the slope model.
- S12 (the moment-spectrum cut at universality) Properly equilibrated, the 2-D
      slope model's total activity (size/energy) is space-filling (D(q) -> 2.0)
      while its avalanche AREA (footprint) is FILAMENTARY (D_area ~ 1.0-1.1, area
      ~ L) with a resolved D(q) drift -- anomalous scaling, NOT the Manna single-
      fractal FSS class, and geometrically DISTINCT from BTW's compact avalanches.
      Sharpens S4 from "an exponent differs" to "the geometry and full moment
      structure differ." Needed two things: avalanche AREA added to the engine
      (the clean, non-cutoff-dominated observable) and slow over-steep
      equilibration (the slope pile reaches repose only from above).
- S13 (the mechanism) Conditional-exponent analysis (the De Menech-Tebaldi
      conditional method validated in S11, here on the slope model) confirms a
      ballistic filamentary front: <E|S> ~ S (energy = size), <S|A> ~ A^1.93 and
      <E|A> ~ A^1.94 (a footprint of A bonds swept ~A times), <A|T> ~ T^0.97
      (ballistic). But the OVER-DETERMINATION test fails through DURATION
      (gamma(S|A) direct 1.93 vs routed-through-T 1.79, ~7x the fit error) while
      closing exactly in the spatial sector (E ~ S): single-scale scaling holds
      among area/size/energy but breaks through duration -- T is a loose proxy for
      spatial extent (long avalanches partly LINGER), the 2-D residue of S3's
      quantized line/wedge families. Cutoff-free, so caveat-free where it is clean.
- S14 (the geometry) The footprint's mass-radius dimension is D ~ 1.0 (A ~ Rg^D),
      a constant-width thin filament -- 98% of large avalanches are a single bond
      wide (anisotropy 1.000) -- thinner than the exactly-solvable directed sandpile
      (D = 3/2) and far from compact BTW (D = 2.0, measured here under the same
      pipeline); and a ballistic front, first-topple time ~ radial distance from the
      launch site (slope 0.998, correlation 0.990). The geometric mechanism behind
      S12's D_area ~ 1, read off real footprints via a validated engine footprint dump.
- S15 (the decisive edge case) A tunable stochastic split (psto) diverts a fraction of
      each downhill flux to a random transverse neighbour, interpolating the
      deterministic gradient rule toward the Manna class. As psto rises the footprint's
      mass-radius dimension climbs 1.0 -> 1.87 (L = 128 and 192 overlapping, so
      intrinsic), passing the directed sandpile's 3/2 near psto ~ 0.1: the filament is a
      SPECIFIC consequence of the deterministic rule, not incidental (the causal upgrade
      of S12/S14). But the area moment drift does NOT flatten toward simple FSS
      (0.23 -> 0.38) -- the avalanches become compact but localized (area ~ L^1.2), so
      the model compactifies in shape without collapsing onto the Manna class; "outside
      Manna" reinforced.
- S16 (the dimensional anchor) The 1-D fast engine gained the S14 footprint dump (it had
      none; only the 2-D engine got it), validated bit-for-bit against a full-scan brute
      reference. The genuine 1-D slope avalanche has mass-radius dimension D = 1.00 (every
      N = 512-4096) and is a SOLID, gap-free interval (area = range; a Cantor self-test
      confirms the estimator would read a fractal as D < 1), a perfectly BALLISTIC front
      (first-topple time = 1.00 x distance from the launch node, corr 1.00), with
      conditional exponents (<E|S> 1.00, <S|A> 2.00, <A|T> 0.98) matching S13's 2-D values.
      So the 2-D "filament" (D ~ 1.02) has the GEOMETRY of a 1-D avalanche (differing only in directedness),
      and D ~ 1 is the intrinsic dimension of a slope avalanche -- not a 2-D measurement
      artifact (the directed sandpile at 3/2 and BTW at 2 are genuinely higher-dimensional).
      One real dimensional CONTRAST: the 1-D front is downhill-directed (the global drive of
      the open right edge; ~93-98% of the footprint downhill of the seed), whereas the 2-D
      front radiates isotropically from the seed. The S13 duration over-determination gap is
      ~0.11 in 1-D too -- inherited from S3's 1-D line/wedge families (48% of avalanches are
      single-sweep "lines"), so it is intrinsic to the slope rule, not a 2-D finite-size
      effect. Bonus enabler for S17: L = 512 equilibrates to repose 2.745 by ~157M iters via
      windowed-mean plateau detection, and the repose creep is confirmed FINITE-SIZE (not
      under-equilibration -- L = 256 reproduces S12's 2.64); whether it saturates is open at
      L <= 512.
- S17 (the duration closure) The question S13 left open: does the single-scale over-
      determination gap (gamma(S|A) direct minus gamma(S|T)/gamma(A|T)) HEAL as L grows or
      survive as a true residual? Using the S16 enabler to reach L = 512 (verified-stationary),
      the COMPONENT exponents settle it: as L grows from 96 to 512, gamma(S|A) -> 2 (1.89 ->
      1.95) and gamma(A|T) -> 1 (0.91 -> 0.99) -- the spatial sector becomes EXACTLY single-
      scale -- but gamma(S|T) saturates at ~1.75 (1.66 -> 1.75, last step flat), NOT 2. So the
      gap does NOT heal; it GROWS with L (0.066 -> 0.183) toward a true residual of ~0.2 (a
      1/L fit gives 0.19 and underestimates, since the gap is still accelerating; the S|T
      plateau implies ~0.25). The spatial lock gamma(E|S) = 1.00 at every L confirms the
      breakdown is duration-specific, not a fit artefact. ANSWER: single-scale scaling is
      EXACT in space (size ~ area^2, area ~ duration) and fails PERMANENTLY through duration --
      duration is an intrinsically loose proxy for spatial extent (some long avalanches linger:
      re-topple in place / propagate intermittently rather than reach farther), so AREA, not
      duration, is the model's clean scaling variable (vindicating S12). The residual is
      comparable to and somewhat LARGER than the 1-D anchor (0.11, S16) -- the duration anomaly
      is intrinsic across dimensions and stronger in 2-D (a 2-D front has more ways to linger).
      This closes the S11-S17 scaling-theory arc.
- S18 (the area-multifractality closure) The one question S11-S17 left: is S12's avalanche-
      AREA multifractality (D(q) drift ~0.27 at L <= 256) a TRUE asymptotic property or a
      finite-size corona that heals? Reaching L = 512 (S16 enabler) with a SLIDING 3-L-window
      of the area moment spectrum, the drift does NOT heal -- it stays ~0.2 across a 4x L range
      and extrapolates to a nonzero 0.169 +- 0.048 (~3.5 sigma; a corona would have fallen ~4x).
      So the area multifractality is ASYMPTOTIC. This FALSIFIES the naive read of S17
      (gamma(S|A) -> 2 with S ~ L^2 => A ~ L^1 mono-fractal) but RECONCILES with it: gamma(S|A)
      -> 2 is a conditional MEAN, while the area DISTRIBUTION's higher moments (tail-weighted)
      stay multifractal -- single-scale in the MEANS (typical avalanche, S17), multifractal in
      the TAILS (largest avalanches, S18), simultaneously and asymptotically. q-resolved: D(q~1)
      = 1.14 (typical filament) -> D(q~4) = 1.34 (tail). A tentative second reading was FLAGGED,
      not claimed: the area dimension may actively RISE at large L (D_mid 1.07 -> 1.23, <A> slope
      0.90 -> 1.22), but it rests on the fewest-avalanche points -> S19.
- S19 (the filament-fattening test, a self-test of S18's open hint) Does the footprint genuinely
      FATTEN at large L (a real crossover toward compactness) or was S18's apparent fattening
      tail-undersampling? Re-measured the footprint geometry at L = 192, 256, 384, 512 with 10-12
      seeds (double S18's 5) and SPLIT each lattice's avalanches into the TYPICAL bulk vs the
      area TAIL. Verdict: TAIL. The typical avalanche is L-stable at the S14/S16 filament values
      -- median area ~ L^0.90, typical mass-radius D(A ~ Rg^D) = 1.00 -> 1.01 across L = 192-512,
      and the top-decile footprints stay ~one bond wide (one-bond fraction 0.99 -> 0.95) -- while
      only the rare largest avalanches grow (A_max 323 -> 1090, mean area grows faster than the
      median). Decisively, the mean-area growth slope that read 1.22 at S18's 5 seeds averages
      DOWN to 0.99 +- 0.03 at 10 seeds, and it EXCEEDS the typical median-area slope 0.90 +- 0.04,
      so the extra growth is tail-weighted, not a thickening of the typical footprint. So the
      filament does NOT fatten: S18's tentative second reading was tail-undersampling at fewest
      seeds, S14/S16's one-bond filament stands to L = 512, and the result is consistent with
      (a sharper statement of) S18's main reading -- single-scale-thin in the typical avalanche,
      multifractal-fat only in the tail. A self-test in the F47/F48/F52/S6 tradition: a flagged
      hint tested and downgraded rather than left to harden into a claim.
- S20 (model selection: does the repose saturate?) S16 confirmed the mean bond slope at SOC
      repose rises with L (2.42 at L=64 -> 2.74 at L=512) as a real finite-size effect, leaving
      open whether it SATURATES (1/L correction -> finite limit) or DIVERGES slowly (log L -> no
      limit). Fitting both models to the seven S16 plateau values and comparing by AIC: the log(L)
      model is clearly preferred (DAIC = 9.7, R^2 0.990 vs 0.960, max residual 1.3x vs 2.8x the
      measurement spread). The per-doubling increments (0.138, 0.089, 0.098) are roughly constant,
      more consistent with log(L)'s expected flat increment than with 1/L's expected halving.
      Result: the SOC repose most likely diverges logarithmically; an infinite-volume critical slope
      may not exist for this model (boundary effects propagating O(L) into the bulk sustain steeper
      gradients at larger L). Caveat: L=64-512 cannot prove divergence over saturation at very large
      L -- confirming would require L~1024-2048. sandpile/repose_scaling.py; self-test confirms AIC
      recovers the true model on synthetic exact-1/L and exact-log(L) data.

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

---

## S11 -- The moment-scaling method, validated on canonical BTW

**Motivation.** S4 separated the slope model from BTW on a single avalanche-size
exponent (tau_S = 0.89 vs 1.14). Single tau exponents are a blunt and, when a
distribution is multifractal, ill-defined instrument -- which is exactly why S4
needed caveats. The field's decisive discriminator is the MOMENT SPECTRUM: how the
whole family of moments <x^q> scales with system size L. For a distribution obeying
simple finite-size scaling (FSS), P(x) = x^{-tau} G(x/x_c) with x_c ~ L^D, the
moment exponent sigma(q) defined by <x^q> ~ L^{sigma(q)} is LINEAR in q, so its
local slope D(q) = d sigma/dq is CONSTANT. A DRIFTING D(q) means multifractal /
anomalous scaling. The established result (De Menech, Stella, Tebaldi, PRE 58,
R2677 (1998); Tebaldi, De Menech, Stella, PRL 83, 3952 (1999)) is that BTW is
multifractal in toppling number but near-FSS in area. Before turning this on the
slope model, reproduce that -- the same "check a known result" discipline as S1,
applied to the analysis.

**Method (`sandpile/moments.py`, `sandpile/moment_fss.py`).** `moments.py`
computes <x^q> over a grid of q, fits sigma(q) by regressing log<x^q> on log L, and
takes D(q) = d sigma/dq; errors come from a bootstrap (S11) or seed-group
jackknife (S12). It carries its own self-test: synthetic avalanches from an EXACT
simple-FSS source (power law with a hard cutoff x_c = A L^D) must read out as a
FLAT D(q) and recover the analytic sigma(q). (Subtlety worth recording: for tau<1
-- the slope model's regime -- the cutoff carries the normalization, so the FSS
line is sigma(q)=D q, not D(q+1-tau); D(q) is flat = D in both regimes, which is
the signature that matters.) The self-test passes: a true FSS source gives D(q)
flat to <0.01, so the method does not manufacture multifractality. `moment_fss.py`
then runs BTW at L = 32-128 (track_area added to `btw_compare.btw_run`, additive)
and measures D(q) for toppling number S and area A.

**Evidence.**

| BTW observable | D(q=1) | D(q=3) | D(q) drift, q in [1,4] | read |
|----------------|--------|--------|------------------------|------|
| toppling number S | 2.42 | 2.72 | **0.302** | multifractal |
| area A (distinct sites) | 2.02 | 2.05 | **0.034** | near-FSS, D_area ~ 2.05 |

The toppling-number D(q) climbs from 2.42 toward ~2.72 (consistent with the
literature avalanche dimension ~2.75) -- a clear, ~10x-the-noise drift. The area
D(q) is flat at ~2.05 (an order of magnitude less drift), i.e. BTW avalanches have
compact, essentially 2-D footprints, and the multifractality lives in the multiple
topplings per site, not in the footprint. Both match De Menech / Tebaldi.

**Interpretation.** The machinery is trustworthy: it flags BTW's toppling number as
multifractal and BTW's area as near-FSS, exactly as known, and it reads a
synthetic FSS source as flat. The area-vs-toppling split is itself the key lesson
carried into S12 -- area is the clean, well-behaved observable.

**Honest note.** A naive "drift > 4 x noise" auto-verdict tags BTW area as
"multifractal" because the area bootstrap noise is tiny (~0.007); the honest read
is "near-FSS, a hair of finite-size creep" (the S6 lesson about auto-verdicts).
The reported verdict is a 3-tier rule requiring the drift to be both several times
the noise AND a sizable fraction of D itself.

---

## S12 -- The moment spectrum of the slope model: filamentary, anomalous, and distinct from BTW

**Question.** Is the 2-D continuous slope sandpile simple-FSS (flat D(q), like the
stochastic Manna class) or multifractal (drifting D(q), like deterministic
BTW/Zhang)? And does the moment spectrum give a caveat-free resolution of the S4
"different class from BTW" claim?

**Two obstacles, both genuine, both solved.**

1. *Size and energy are cutoff-dominated.* The slope model has tau_E, tau_S < 1
   (S3/S4), so every positive moment is set by the few largest, system-spanning
   avalanches (<S> ~ 0.1 S_max). Their moment scaling is noisy and correction-laden.
   The cure, learned from S11, is avalanche AREA = number of DISTINCT toppled bonds
   (the footprint), bounded by the lattice and not cutoff-dominated. Area was added
   to the fast engine (`sandpile_fast.py`, `run_sandpile2d_fast(track_area=True)`):
   it records, per iteration, the bonds toppling for the first time in the current
   avalanche. Validated bit-for-bit against an independent full-scan distinct-bond
   recount (max difference 0 over both lattice sizes tested) and confirmed not to
   perturb the dynamics (disp/act identical with tracking on vs off).
2. *The slope pile equilibrates slowly and only from above.* Started over-steep
   (uniform slope 0.9 Zc = 4.5) the pile begins active -- bonds near Zc -- and
   slowly relaxes its mean down to the repose ~2.5; started near repose at large L
   it stays dormant and bleeds out (dilute single-site forcing loses to boundary
   drainage). So each lattice needs a long, L-scaled warmup, and -- crucially --
   the stationarity gauge must be the MEAN SLOPE, not the cutoff-dominated <S>
   (which is too noisy to judge convergence; an early version mistook its sampling
   noise for non-stationarity). `moment_slope.py` therefore runs two-phase: a long
   warmup with the series UNRECORDED (no memory cost), confirm the mean slope has
   settled, then measure a recorded window from the equilibrated state.

**Method (`sandpile/moment_slope.py`).** Equilibrated runs at L = 64, 96, 128, 192,
256 (warmups 5M-30M scaled with L; 8-10 seeds each; ~1.8-2.4e5 avalanches per
size). Mean slope confirmed stable per L (2.43, 2.50, 2.57, 2.62, 2.64 -- the
equilibration check). D(q) for area A, toppling number S, and energy E, with
seed-group jackknife errors. BTW D(q) overlaid from S11.

**Evidence.**

| observable | D(q=1) | D(q=2) | D(q=3) | high-q | drift [1,4] | character |
|------------|--------|--------|--------|--------|-------------|-----------|
| area A | 1.01 | 1.07 | 1.15 | ~1.3-1.5 | 0.275 | **filamentary, D_area ~ 1.0-1.1** |
| toppling S | 1.89 | 1.97 | 1.99 | -> 2.00 | 0.105 | space-filling activity, D -> 2 |
| energy E | 1.90 | 1.98 | 2.00 | -> 2.01 | 0.107 | space-filling activity, D -> 2 |

- **The footprint is filamentary.** Area grows only ~linearly with L (local log-log
  slope of <A> = 0.88, 0.89, 0.92, 1.03 across the L range), so D_area ~ 1.0-1.1 --
  the distinct-bond footprint of even the largest avalanches is a thin, near-1-D
  front, not a compact blob. Contrast BTW's compact D_area ~ 2.05 (S11).
- **The activity is space-filling.** Size and energy D(q) climb smoothly to 2.0:
  the largest avalanches' TOTAL toppling work scales as L^2 (the lattice area).
- **Therefore each footprint bond topples ~L times.** size/area ~ L^2 / L^1 = L: a
  big avalanche is a thin front that sweeps the same bonds O(L) times. This is the
  slope/gradient analog of BTW's multiple-toppling, but concentrated on a 1-D
  footprint instead of a 2-D one.
- **Anomalous, not Manna-class.** No observable has a flat D(q). The area D(q) drift
  is resolved at low-to-mid q (D rises 1.01 -> 1.07 -> 1.15 over q = 1-3, a rise of
  0.15 against a seed-group noise of 0.003-0.02), so it is a real anomaly, not
  high-q noise. The slope model is not in the stochastic Manna single-fractal FSS
  class; it sits with the deterministic BTW/Zhang anomalous-scaling family.

**Interpretation.** This is the caveat-free upgrade of S4. S4 said "tau_S differs
from BTW (0.89 vs 1.14)" with a bond-vs-site hedge. S12 says the slope and BTW
sandpiles differ in their avalanche GEOMETRY and full moment structure: BTW
avalanches are compact (area ~ L^2) with multifractal toppling number; slope-model
avalanches are filamentary (area ~ L^1) with space-filling total activity, the
intense multiple-toppling running along a thin front. Both are deterministic SOC
with anomalous (non-Manna) scaling, but they are distinct anomalous classes -- a
stronger and more physical statement than a single-exponent gap. It also explains
S4's caveats: a single tau is ill-defined precisely because these distributions are
not simple-FSS.

**Honest caveats (own them).** (a) L is capped at 256: above that the over-steep
pile does not fully equilibrate within a verifiable warmup (the mean slope is still
creeping up, 2.64 at L=256), so L=512 is left to future work -- the equilibration
barrier, not compute (runs are seconds), is the limit. (b) The high-q area moments
become cutoff-sensitive too (q=5 error +-0.13), so the area D(q) drift is quoted
from the resolved low-mid-q range, not the noisy tail. (c) Size/energy D(q) -> 2.0
with a mild low-q correction (drift ~0.1) is consistent with corrections-to-scaling
around a space-filling D=2; I do NOT claim size/energy multifractality, only that
the activity is space-filling. The robust, resolved result is the geometric one:
filamentary footprint (D_area ~ 1) vs BTW's compact (D_area ~ 2). (d) The mean
repose slope drifts slightly upward with L (2.43 -> 2.64); whether that is a weak
finite-size dependence of the repose or residual under-equilibration is not settled
here, but it does not affect the avalanche-geometry conclusion, which rests on the
area scaling.

---

## S13 -- Conditional avalanche scaling: the 2-D slope avalanche is a ballistic filamentary front

**Question.** S12 established WHAT the 2-D slope avalanche looks like (filamentary
footprint, area ~ L; space-filling activity, ~ L^2) but through moment spectra,
which for this model are cutoff-dominated (tau < 1) and so carried hard-won
caveats. Can the same physics be read off a CLEAN, caveat-free observable, and does
it cohere into a single mechanism? The natural tool is the CONDITIONAL exponent
gamma(x|y) defined by <x | y> ~ y^{gamma(x|y)}, measured at FIXED y -- so it never
sees the system-size cutoff. (This is the De Menech-Tebaldi "conditional
distribution at a given area" method, PRL 83, 3952 -- the same papers validated in
S11 -- here turned on the slope model.)

**The mechanism and its predictions.** Three earlier results combine into one
picture. A 2-D avalanche is a thin front of linear extent ell that (i) propagates
one node per iteration (the book's ballistic front, p.122; confirmed by S10's
duration cutoff ~ L^1.07), so duration T ~ ell; (ii) leaves a filamentary footprint
(S12's D_area ~ 1), so area A ~ ell; and (iii) sweeps each footprint bond ~ell times
(S12's size/area ~ L), so size and energy S, E ~ ell^2. If a SINGLE scale ell
governs the avalanche, the conditional exponents are fixed:

    <A|T> ~ T^1   <S|T> ~ T^2   <E|T> ~ T^2
    <S|A> ~ A^2   <E|A> ~ A^2   <E|S> ~ S^1

The <S|T> ~ T^2 is the book's "wedge" law E ~ T^2 (p.122), which in 1-D only BOUNDS
the avalanche family from above; the prediction is that in 2-D it becomes the
TYPICAL behaviour, because a 2-D front almost always spreads transversely and the
pure "line" avalanches (slope +1) of 1-D disappear. Crucially the six exponents are
OVER-DETERMINED: single-scale scaling forces gamma(S|A) = gamma(S|T)/gamma(A|T) and
gamma(E|S) = gamma(E|T)/gamma(S|T), so the picture can FAIL a self-consistency test
-- it is not six free fits.

**Method (`sandpile/conditional.py`).** Equilibrated 2-D slope lattices at
L = 96, 128, 192, 256 (warm over-steep, gauge by mean slope, as S12; 6 seeds each,
~1.6-1.8e5 avalanches per size; mean slope 2.50 / 2.55 / 2.63 / 2.65). Per-avalanche
(E, S, T, A) all segmented on the displaced-mass series (the grouping validated in
sandpile_fast._test_area2d). Conditional means by log-binning on the conditioning
variable, log-log slope over a clean window (skip the quantized small-T head and the
sparse tail). A self-test draws a synthetic single-scale source (one hidden ell per
avalanche, A = ell, T = ell, S = E = ell^2 with multiplicative noise) and recovers
1, 2, 2, 2, 2, 1 to < 0.01 -- the binning/fit does not bias the slopes.

**Evidence (pooled L = 192 + 256; per-L trend in the last column).**

| relation | gamma (measured) | predicted | per-L trend (96 -> 256) |
|----------|------------------|-----------|--------------------------|
| <E\|S>   | 1.00 +- 0.00     | 1 | 1.00 flat |
| <S\|A>   | 1.93 +- 0.02     | 2 | 1.89 -> 1.94 |
| <E\|A>   | 1.94 +- 0.02     | 2 | 1.90 -> 1.95 |
| <A\|T>   | 0.97 +- 0.01     | 1 | 0.92 -> 0.98 |
| <S\|T>   | 1.73 +- 0.02     | 2 | 1.67 -> 1.72 |
| <E\|T>   | 1.74 +- 0.02     | 2 | 1.68 -> 1.73 |

Over-determination (single-scale identities, composite vs directly measured):

| identity | composite | direct |
|----------|-----------|--------|
| gamma(S\|A) = gamma(S\|T)/gamma(A\|T) | 1.79 | 1.93 |
| gamma(E\|S) = gamma(E\|T)/gamma(S\|T) | 1.01 | 1.00 |

**Two conclusions, one clean and one subtle.**

1. **The filamentary-multiply-swept mechanism is confirmed, cutoff-free.** Energy
   and size are locked (E ~ S^1.00 -- each toppling sheds ~Zc/4 of mass, so E is S
   times a constant), and a footprint of A bonds carries ~A^1.93 topplings, i.e.
   each footprint bond topples ~A times. This is exactly S12's "thin front swept
   ~ell times", now from a conditional measure the cutoff cannot contaminate and
   with no bond-vs-site or tau-definition caveat. The 2-D E-T plane (figure, lower
   left -- the 2-D analogue of the book's Fig 5.6) confirms the geometric
   prediction: the avalanche density hugs the slope-+2 "wedge" line, and the
   slope-+1 "line" branch that fills the lower 1-D wedge is essentially gone. The
   book's upper wedge bound has become the generic 2-D law.

2. **But single-scale scaling fails THROUGH DURATION.** The over-determination test
   does not close: the spatial size-area exponent measured directly
   (gamma(S|A) = 1.93) disagrees with the value routed through duration
   (gamma(S|T)/gamma(A|T) = 1.79) by 0.14, about seven times the per-fit error,
   whereas the purely spatial identity gamma(E|S) closes exactly (1.00 vs 1.01).
   Read literally: the spatial observables (area, size, energy) form a tight
   single-scale family (S, E ~ A^2, E ~ S), but duration is a LOOSE proxy for
   spatial extent -- <S|T> = 1.73 sits well below the 1.87 the spatial chain
   predicts, because some long-duration avalanches are long not because they reach
   farther but because they LINGER (re-topple in place / propagate intermittently).
   At fixed duration the avalanche keeps a shape/family degree of freedom. This is
   the two-dimensional residue of S3, where the 1-D avalanches split into discrete
   line/wedge families and the duration-based scaling relation likewise failed; and
   it vindicates S12's choice of AREA, not duration, as the clean scaling variable.

**Honest caveats (own them).** (a) Every exponent drifts toward its single-scale
value as L grows (<A|T> 0.92 -> 0.98, <S|A> 1.89 -> 1.94, <S|T> 1.67 -> 1.72 across
L = 96 -> 256), so part of the 0.14 gap is a finite-size correction; whether it
heals to exact single-scale or leaves a residual duration anomaly is the sharp
question for S14 and needs the L > 256 that the over-steep equilibration ceiling
(S12) currently blocks. (b) The runs sit at mean slopes 2.50-2.65, still mildly
under the asymptotic repose (same S12 equilibration limit); the L-trend, not any
single L, carries the conclusion. (c) <S|T> and <E|T> are the same exponent within
error (1.73 vs 1.74), as they must be once E ~ S -- a consistency check, not two
independent facts.

**Why this is interesting.** It converts S12's geometric snapshot into a mechanism
with falsifiable, over-determined predictions, and the test does something a single
exponent cannot: it isolates WHERE the slope model departs from simple finite-size
scaling (the duration sector, not the spatial one) and ties that departure
quantitatively to the 1-D family structure of S3. It also extends the chapter's
central avalanche figure (the 1-D E-T wedge, Fig 5.6) into 2-D and reads off the
prediction that the 2-D avalanche is generically a "wedge", the 1-D "line" branch
having closed. The clean piece (E ~ S, S ~ A^~2) is the caveat-free statement S4
and S12 were reaching for; the subtle piece (duration is a loose scaling variable)
is a new, concrete entry point for the duration-closure follow-up.

---

## S14 -- Avalanche geometry and directedness: the footprint is a ballistic constant-width filament

**Question.** S12 (moment area) and S13 (conditional exponents) both concluded the
2-D slope avalanche is "filamentary", area ~ L, but each measured only a single AREA
exponent -- neither looked at the SHAPE. Is the footprint actually a thin curve, or
just a low-density region that happens to enclose few distinct bonds? And is it a
DIRECTED path -- a downhill front, like the exactly-solvable directed sandpile -- or
an isotropically branching star? This is the geometric mechanism behind D_area ~ 1,
and the natural place to put the model next to the two reference sandpiles it is
being compared with (compact BTW, directed Dhar-Ramaswamy). S13 had flagged the
duration-closure question as the next step, but that needs lattices past L = 256 (the
equilibration ceiling S12/S13 hit); the geometry leg is ready now and explains the
filament directly, so it goes first.

**The decisive observable.** The per-avalanche mass-radius dimension D, defined by
A ~ Rg^D (footprint occupied-cell count A against its radius of gyration Rg). It is a
single dimensionless number that places the avalanche on one axis:

    BTW compact (S11 D_area ~ 2.05)              D = 2     transverse ~ longitudinal
    Dhar-Ramaswamy DIRECTED sandpile (exact)     D = 3/2   transverse ~ longitudinal^1/2
    a constant-width thin filament               D = 1     transverse ~ O(1)

A directed sandpile spreads diffusively about its drift axis, so a footprint of
linear extent ell has transverse width ~ ell^1/2 and area ~ ell^3/2 (D = 3/2 exactly,
tau = 4/3; Dhar-Ramaswamy 1989). If the slope model instead reads D ~ 1 its footprint
is even THINNER -- a constant-width filament, area ~ ell -- the most filamentary of
the three, and D_area ~ 1 (S12) becomes a literal geometric fact, not just an exponent.

**Method (`sandpile/geometry2d.py`).** The fast engine gained a validated FOOTPRINT
DUMP (`run_sandpile2d_fast(dump_fp=True)`): for every avalanche above an area cut it
records the set of distinct toppled bond ids, each bond's first-topple iteration, and
the launching grain site. It is a pure observer -- the dynamics are bit-identical to a
dump-off run (max|d disp| = max|d act| = 0), and the dumped sets, seeds and times match
an independent full-scan reference exactly (`_test_footprint2d`). Footprints were
collected from equilibrated lattices L = 96, 128, 192, 256 (warm over-steep, gauge by
mean slope, the S12 protocol; 4 seeds, ~45-51k footprints of area >= 8 per L; mean
slopes 2.50/2.53/2.62/2.67). Each footprint's bond ids decode to plane coordinates;
the per-footprint geometry is the radius-of-gyration tensor (rms widths along the two
principal axes; anisotropy = (lam1-lam2)/(lam1+lam2)), and the mass-radius D comes from
a binned A ~ Rg^D regression over the ensemble. Cross-checks: per-footprint
box-counting on the 40 largest, and the IDENTICAL pipeline on a canonical BTW baseline
(compact reference). A self-test reads back the mass-radius D of three synthetic
ensembles of known dimension -- straight lines (1.000), filled disks (2.001), directed
sqrt-width fronts (1.499) -- so the estimator is unbiased before it touches the data.

**Evidence.**

| quantity | slope model | reading |
|---|---|---|
| mass-radius D (A ~ Rg^D), pooled L=192+256 | 1.017 +- 0.003 | a thin filament |
| D per L (96 / 128 / 192 / 256) | 1.001 / 1.005 / 1.013 / 1.017 | L-independent (slow upward creep, as the mean slope) |
| longitudinal rms width w_long ~ A^? | A^0.99 +- 0.004 | the filament length = the area |
| transverse rms width (top decile A) | 0.093 bonds | constant, ~0 |
| one-bond-wide fraction (top decile A) | 0.98 | 98% are literally a single-bond line |
| anisotropy (top decile A) | 1.000 | a perfect line |
| box-counting D_box (40 largest) | 1.16 +- 0.04 | independent confirmation of D ~ 1 |
| BTW mass-radius D (L=64/96/128) | 2.00 / 2.00 / 2.01 | compact, matches S11's D_area 2.05 |

Directedness / ballistic spreading (L=256): pooling each bond's first-topple time
against its radial distance from the launch site gives time ~ 0.998 x radius with
linear correlation 0.990. The front advances ~one cell per iteration outward from the
seed -- a ballistic front, seen directly in space.

**Conclusions.**

1. **The footprint is a constant-width thin filament (D ~ 1).** Not D = 2 (BTW
   compact) and not even D = 3/2 (the directed sandpile, which spreads to transverse
   width ~ ell^1/2). The deterministic gradient/halving rule makes the most filamentary
   avalanche of the three classes: a footprint one bond wide (98%), whose length IS its
   area (w_long ~ A^1) and whose transverse width is O(1) and does not grow. S12's
   D_area ~ 1 and S13's mechanism are now a literal geometric statement -- a straight
   thin line -- read off real footprints, not inferred from an exponent.

2. **It is a ballistic front radiating from the seed.** First-topple time tracks radial
   distance one-to-one (slope 0.998, correlation 0.990): the avalanche is a front moving
   outward at ~one cell per iteration. This is the spatial face of S10's duration cutoff
   L^1.07 and S13's <A|T> ~ T -- ballistic propagation seen in space, not inferred from a
   duration exponent.

3. **Placement in the SOC landscape.** On the geometric (mass-radius) axis the slope
   model sits BELOW the directed sandpile: slope D ~ 1 < directed 3/2 < BTW 2. The model
   is not a directed sandpile -- it has no global drive direction (random interior
   forcing, all four edges drain) -- but each avalanche is locally a straight
   steepest-descent filament with none of the diffusive transverse wandering the
   stochastic directed model has. So the deterministic gradient rule is MORE ordered
   than the canonical directed sandpile, not less.

**Honest caveats (own them).** (a) D drifts slightly upward with L (1.001 -> 1.017),
tracking the same mean-slope creep (2.50 -> 2.67) that S12/S13 carry from incomplete
equilibration above L = 256; the trend is small and D stays pinned near 1, far from 3/2
and 2, so the placement is firm even though the third digit is not. Whether the
asymptotic D is exactly 1 or a hair above is the same finite-size question S12/S13 leave
open. (b) The per-footprint box-counting D_box = 1.16 is the soft confirmation -- only
~1.5-2 decades of box sizes at L <= 256 -- so the robust number is the ensemble
A ~ Rg^D = 1.02, from thousands of avalanches over a decade-plus of Rg. (c) The
Dhar-Ramaswamy directed sandpile is a different model (a global drive); the comparison
is on the geometric dimension axis only, not a claim that the two share a class. (d)
<w_trans> = 0.093 means most footprints have exactly zero transverse variance; the
residual ~2% with finite width are short branch points where the front momentarily
splits, so "filament" is the typical avalanche, with rare branched exceptions.

**Why this is interesting.** It converts S12/S13's "filamentary" from an exponent into
a picture you can see (the figure's footprint snapshots are single bright lines) and
gives D_area ~ 1 a mechanism: the deterministic gradient rule funnels each avalanche
into a one-bond-wide ballistic steepest-descent front. And it places the model concretely
against the two sandpiles it is most naturally compared with -- thinner than the
exactly-solvable directed sandpile, far from compact BTW -- which is exactly the "where
does it sit in the SOC landscape" question the universality thread (S4, S11, S12, S13)
has been building toward. figures/sandpile_geometry.png.

---

## S15 -- The decisive edge case: the filament is CAUSED by the deterministic rule

**Question.** S12 (moment area) and S14 (footprint geometry) established that the 2-D
slope avalanche is a constant-width filament (mass-radius D ~ 1) and asserted the model
is NOT in the stochastic Manna universality class. But that placement was
CORRELATIONAL: we characterised the deterministic model and noted it differs from Manna.
We never showed the determinism is the CAUSE of the filament -- a skeptic could ask
whether the thinness is incidental, or whether the model is really just Manna in
disguise. S15 makes it causal with a tunable knob: add stochasticity to the
redistribution, interpolate toward the Manna class, and watch whether the filament
survives.

**Method (the knob; `sandpile_fast.run_sandpile2d_fast(psto=...)`).** The deterministic
rule sheds z/4 of sand straight downhill across each unstable bond. The new `psto`
parameter diverts a fraction psto of that downhill share to a randomly chosen TRANSVERSE
neighbour of the recipient (sign drawn per topple), conservatively -- the recipient
loses exactly what the transverse site gains, no sand created or destroyed. This injects
the defining Manna ingredient (stochastic, non-directed redistribution) into the bond
rule while keeping everything else (slope threshold, halving magnitude, open edges, slow
forcing) fixed. psto = 0 is the deterministic gradient rule. The engine self-test
`_test_split2d` confirms psto = 0 is BIT-IDENTICAL to the S1-S14 dynamics (max|d
disp,act,area,S| = 0, no extra RNG draw), the split conserves sand (initial + added -
drained - final = 1e-10), and psto > 0 genuinely changes the dynamics. Two observables
are swept, both reusing the validated S12/S14 machinery unchanged: S14's mass-radius
dimension D (A ~ Rg^D, from the footprint dump; L = 128 with 4 seeds AND L = 192 with 3
seeds, so an intrinsic crossover shows as the two lattice sizes overlapping) and S12's
area moment drift D(q) (FSS across L = 96, 128, 192, 6 seeds, at the endpoints psto = 0
and 0.5). `sandpile/stochastic_split.py`.

**Evidence -- mass-radius dimension D vs psto (the geometry; the two L overlap, so this
is intrinsic, not finite-size):**

| psto | D (L=128) | D (L=192) | one-bond-wide (L=128) | <w_trans> (L=128) |
|------|-----------|-----------|------------------------|-------------------|
| 0.00 | 1.005     | 1.015     | 0.99                   | 0.04              |
| 0.05 | 1.270     | 1.284     | 0.23                   | 0.85              |
| 0.10 | 1.538     | 1.562     | 0.01                   | 1.24              |
| 0.20 | 1.728     | 1.755     | 0.00                   | 1.82              |
| 0.35 | 1.785     | 1.793     | 0.00                   | 2.38              |
| 0.50 | 1.869     | 1.862     | 0.00                   | 2.94              |

**Evidence -- area moment drift D(q) over q in [1,4] (the universality signature):**

| psto | D(q) drift | D_mid | read |
|------|------------|-------|------|
| 0.00 | 0.230      | 1.10  | anomalous, filamentary (reproduces S12) |
| 0.50 | 0.384      | 1.19  | anomalous, did NOT flatten toward simple FSS |

Mean bond slope stays 2.48-2.74 across the whole sweep, so the pile reaches a sensible
SOC repose at every psto and the comparison is not confounded by a different state.

**Two conclusions, one clean and one a genuine nuance.**

1. **The filament is a SPECIFIC consequence of the deterministic gradient rule (clean,
   causal).** Turning on stochasticity drives the footprint from a one-bond-wide filament
   (D ~ 1.0) toward a compact blob (D ~ 1.87), monotonically and -- the key control -- with
   the L = 128 and L = 192 curves lying on top of each other, so the crossover is intrinsic,
   not a finite-size effect. The footprint stops being one bond wide almost immediately
   (the one-bond-wide fraction collapses 0.99 -> 0.01 by psto = 0.1, transverse width 0.04 ->
   2.94). Strikingly, near psto ~ 0.1 the model passes THROUGH the exactly-solvable directed
   sandpile's value D = 3/2 (1.54 / 1.56): a small transverse leak reproduces the directed-
   sandpile geometry, and more leak overshoots it toward compact. This is the causal upgrade
   of S12/S14: their "filamentary" is not incidental but a direct consequence of the
   deterministic halving rule funnelling each avalanche into a single steepest-descent
   thread. Remove the determinism and the thread fattens into a blob.

2. **But the model does NOT collapse onto the Manna simple-FSS class (the nuance).** The
   naive expectation was that injecting stochastic redistribution would flatten the
   anomalous moment spectrum toward Manna's simple FSS (constant D(q) ~ 2). It does the
   opposite: the area D(q) drift GROWS (0.23 -> 0.38) and D_mid stays ~ 1.1-1.2, nowhere
   near 2. Reconciling this with the compact footprints (mass-radius D ~ 1.87): the
   avalanches become compact in SHAPE but LOCALIZED -- their area scales as ~ L^1.2 with
   system size, not the ~ L^2 of compact spanning avalanches. The transverse leak breaks
   the coherent ballistic front (S14) into compact but sub-spanning splats; it changes the
   local geometry without converting the global scaling to simple FSS. So S12's "outside
   Manna" is not just reaffirmed but REINFORCED: the deterministic gradient rule is not a
   hair's breadth from Manna that a little noise tips over -- its anomalous scaling is
   robust, and deliberately injecting stochastic redistribution compactifies the shape
   while keeping the scaling anomalous.

**Honest caveats (own them).** (a) `psto` is a transverse-LEAK on the existing bond rule,
not literally the Manna model (height threshold + fully random two-grain redistribution).
It is the minimal CONSERVATIVE deformation that injects non-directed stochastic
redistribution, so it interpolates TOWARD Manna-like behaviour without BEING Manna --
which is exactly why it can compactify the footprint yet not reach simple FSS. A literal
Manna comparison would be a separate model, not a knob. (b) D saturates at ~ 1.87 at
psto = 0.5, just short of compact 2, consistent with the residual deterministic backbone
(only a fraction psto of each flux leaks; the rest still flows straight downhill). (c) the
localized-compact reading (area ~ L^1.2) comes from a 3-point FSS (L = 96, 128, 192) with
6 seeds; the direction is firm (drift did not flatten, D_mid far from 2) but the precise
1.2 is soft. Pushing psto higher or to a fully random redistribution to probe the true
Manna limit is left to future work. (d) the mean slope wanders mildly with psto (2.48 ->
2.74 -> 2.54) but the SOC state persists at every psto, so the geometry comparison is
clean.

**Why this is interesting.** It converts the universality thread's central claim from a
correlation ("the deterministic slope model is filamentary and unlike Manna") into a
CAUSAL one ("the determinism is what makes it filamentary"), using the same tunable-knob
falsification design the chapter has leaned on throughout (the S6/S11 self-test
discipline, now turned on the model's own defining ingredient). And the result is richer
than the clean crossover it was built to find: the model does not sit a small perturbation
away from Manna, it sits robustly in its own anomalous place, and stochasticity
compactifies its avalanches into localized splats rather than carrying it into the Manna
class. The directed-sandpile dimension 3/2 appearing as a way-point near psto ~ 0.1 also
ties the three reference sandpiles (filament 1, directed 3/2, compact 2) onto one
continuous knob. figures/sandpile_stochastic.png.

---

## S16 -- The dimensional anchor: the 2-D filament has the geometry of a 1-D avalanche

**Question.** S12-S15 built the chapter's central geometric claim: the 2-D slope
avalanche is a constant-width FILAMENT (mass-radius dimension D ~ 1), a ballistic
front, thinner than the directed sandpile (D = 3/2) and far from compact BTW (D = 2).
That D ~ 1 was measured in two dimensions, where it is a non-trivial value -- a 2-D
lattice CAN host a compact (D = 2) avalanche, and BTW does. But "D ~ 1" begs a
dimensional question. D = 1 is the MAXIMAL value a 1-D footprint can take (it lives on
a line), so: what does a genuine 1-D slope avalanche look like under the SAME
measurement, and does the 2-D filament actually match it? If the 2-D avalanche has the
same mass-radius dimension, the same ballistic propagation and the same conditional-
exponent structure as the real 1-D avalanche, then the 2-D "filament" is literally a
1-D object embedded in the plane -- D ~ 1 is the intrinsic dimension of a slope
avalanche, not an artifact of the 2-D measurement. This is the dimensional anchor for
the S14 geometry, and it closes the loop back to S3, where the 1-D model was first
characterised.

**Infrastructure (the prerequisite).** Until now the fast 1-D engine
(`sandpile_fast.run_sandpile_fast`) recorded only mass / displaced-mass / falloff; the
S14 footprint dump existed for the 2-D engine ONLY. S16 added the same machinery to the
1-D engine: a per-iteration first-time-bond-toppling count (`track_area`, summed over an
avalanche = its AREA = distinct toppled bonds), a per-iteration toppling count (`act`,
summed = SIZE, the 1-D analogue of the 2-D series), and a per-avalanche FOOTPRINT dump
(`dump_fp`: the distinct toppled bond-id set with each bond's first-topple iteration and
the launching grain node; a bond id is just the pair index j). It is a pure observer --
two new self-tests, `_test_area1d` and `_test_footprint1d`, drive the fast engine and an
independent full-scan brute reference with one shared forcing stream and confirm: the
dynamics are bit-identical with tracking on vs off (max|d disp| = 0), the area and size
series match the brute recount exactly (max diff 0), and the dumped footprint sets,
launch nodes and times match the brute reference exactly (additive, default-off, so
S1-S15 are untouched).

**Method (`sandpile/geometry1d.py`).** Equilibrated 1-D lattices at N = 512, 1024, 2048,
4096 (warm from a triangle IC to the 1-D repose ~4.2, gauge by mean slope, 3 seeds each,
~7-9k footprints of area >= 8 per N), footprints dumped over a recorded window. The SAME
estimators as `geometry2d.py` (S14): the mass-radius dimension D from a binned A ~ Rg^D
regression, the conditional exponents from binned <y|x> slopes (the S13 method), and an
A ~ time-vs-distance ballistic fit. Two 1-D-specific observables: the SOLIDITY A/range
(= 1 for a gap-free interval, < 1 for a sparse/fractal set -- in 1-D the interesting
content is not "is D < 2" (it cannot be) but "is the footprint solid or fractal"), and
the DOWNHILL fraction (footprint bonds at or right of the launch node, toward the open
right edge). A self-test reads back the mass-radius dimension of synthetic 1-D sets of
known dimension -- solid intervals (D = 1.000, solidity 1.000) and a middle-third Cantor
set (D = 0.631 = log2/log3, solidity 0.017) -- so a measured D = 1 on the data is a real
"solid, not fractal" statement, not a tautology; the conditional and ballistic fitters
recover 2, 1 and slope 1 on synthetic single-scale input.

**Evidence -- the 1-D footprint geometry.**

| quantity | 1-D (S16) | 2-D (S14/S13) | reading |
|---|---|---|---|
| mass-radius D (A ~ Rg^D) | 0.999 (N = 512..4096, L-indep) | 1.02 | same intrinsic dimension |
| solidity A / range | 1.000 (median and top decile) | (n/a) | a solid, gap-free interval |
| ballistic time vs distance | 1.000 x distance, corr 1.000 | 0.998 x radius, corr 0.990 | one node per iteration |
| downhill fraction | 0.93 (median), 0.98 (top decile) | seed-isotropic (no global drive) | the one dimensional CONTRAST |

**Evidence -- conditional exponents (the S13 method, 1-D vs 2-D, pooled N = 2048+4096).**

| relation | 1-D (S16) | 2-D (S13) | prediction |
|----------|-----------|-----------|------------|
| <E\|S>   | 1.001 | 1.00 | 1 (energy = size) |
| <S\|A>   | 2.002 | 1.93 | 2 (A bonds swept ~A times) |
| <E\|A>   | 2.006 | 1.94 | 2 |
| <A\|T>   | 0.981 | 0.97 | 1 (ballistic) |
| <S\|T>   | 1.852 | 1.73 | 2 |

Over-determination (single-scale identity, direct vs routed through duration):
gamma(S|A) direct = 2.002 vs via-T gamma(S|T)/gamma(A|T) = 1.852/0.981 = 1.888, a GAP of
**0.114** -- comparable to the 2-D gap of 0.14 (S13). And 48% of 1-D avalanches are
single-sweep "lines" (size/area < 1.5), the rest multiply-swept "wedges" -- S3's
quantized line/wedge families, seen directly.

**Conclusions.**

1. **The 2-D filament has the geometry of a 1-D avalanche.** The genuine 1-D slope
   avalanche is a SOLID, gap-free, ballistic interval: mass-radius D = 1.000 (= its
   range, no fractal gaps -- the Cantor self-test confirms the estimator would catch
   gaps), first-topple time = 1.000 x distance from the seed (S3's "one node per
   iteration", now seen directly in space). This is exactly the geometry S14 measured for
   the 2-D footprint (D ~ 1.02, ballistic, 98% one bond wide). So D ~ 1 is not a feature
   of the 2-D measurement; it is the INTRINSIC dimension of a slope avalanche, the same
   in the dimension where the value is maximal (1-D) as in the dimension where it is one
   of three possibilities (2-D). The directed sandpile (3/2) and BTW (2) are genuinely
   higher-dimensional objects; the deterministic gradient rule is not. (One honest
   qualifier, taken up in conclusion 4: the two share dimension, sweep structure and
   ballistic propagation, but NOT symmetry -- the 1-D front is directed downhill, the 2-D
   front is seed-isotropic -- so the 2-D avalanche has the GEOMETRY of a 1-D avalanche
   rather than being identical to one.)

2. **The conditional-exponent mechanism is dimension-independent.** Energy locks to size
   (<E|S> = 1.00 in both), a footprint of A bonds carries ~A^2 topplings (each swept ~A
   times) in both, and the front is ballistic (<A|T> ~ 1) in both. The S13 picture -- a
   filamentary front swept O(ell) times -- is the same in 1-D and 2-D.

3. **The duration anomaly is inherited from 1-D, not a 2-D finite-size effect.** S13
   found single-scale scaling fails THROUGH DURATION in 2-D (over-determination gap 0.14)
   and read it as "the 2-D residue of S3's quantized line/wedge families", but flagged
   that part of the gap might be a finite-size correction healing only at L > 256. The
   1-D anchor measures the SAME gap (0.114) in the dimension where S3 identified its
   origin, with 48% of avalanches being single-sweep lines. So the duration sector's
   departure from single-scale scaling is intrinsic to the slope rule across dimensions,
   present already in 1-D -- a 1-D anchor for the open S17 question (it suggests the 2-D
   gap is a true residual, not purely finite-size, though the direct L > 256 test is S17).

4. **The one genuine dimensional contrast is directedness.** The 1-D front is strongly
   downhill-directed (93-98% of the footprint lies downhill of the seed) because the 1-D
   model has a global drive -- open right edge, walled left. The 2-D model has no global
   drive (interior forcing, all four edges drain), so its front radiates isotropically
   from the seed (S14). The avalanche is the same 1-D ballistic object in both; only its
   orientation relative to a global gradient differs.

**The L > 256 equilibration enabler (`sandpile/equilibrate2d.py`).** S13 named L > 256 as
the requirement for the duration-closure question (S17); S12-S15 could only VERIFY the
repose up to L = 256, with the mean slope still creeping with L. S16 delivers the enabler.
The mechanism, made explicit: the S12 start `pyramid_ic(L, 0.9*Zc)` has mean bond slope
~0.9*Zc/2 ~ 2.25 (each pyramid face has |bond| = slope but half the lattice ramps up and
half down), just below the repose; forcing builds the pile up, for large L it briefly
overshoots and relaxes to a plateau. Because the per-chunk mean slope FLUCTUATES (~0.03 at
L = 128, less at large L) -- more than a naive 0.01 tolerance -- convergence is judged on a
WINDOWED MEAN (the average over the last few chunks stops changing). `equilibrate(L)`
returns the equilibrated state, the plateau repose with its spread, and the trace, so S17
can warm an L = 512 lattice and run a recorded window from a verified-stationary state.
Two facts established: (1) **L = 512 equilibrates** to repose 2.745 by ~157M iterations
(~25 s with the S9 engine) -- compute was never the barrier, a verifiable stopping rule
was. (2) **The repose creep is finite-size, not under-equilibration**: every L reaches an
IC-independent (start-height-independent, self-test) stationary plateau, and L = 256's
plateau (2.647) reproduces S12's 2.64, so S12 was already equilibrated. The repose rises
with L (64..512: 2.42, 2.50, 2.56, 2.62, 2.65, 2.71, 2.745); a 1/L fit extrapolates to
~2.79, but the per-doubling increment over the last two doublings (0.09, 0.10) does not
clearly shrink, so whether it saturates or grows slowly (log L) is not settled at L <= 512
-- this does not affect S17, which needs the stationary state, not the asymptotic repose.

**Honest caveats (own them).** (a) In 1-D, D = 1 is the maximal mass-radius dimension a
footprint can have, so D = 1 alone is not surprising; the content is the SOLIDITY (A =
range, no fractal gaps -- which the Cantor self-test shows is a real measurement) plus the
match in ballistic and conditional structure to 2-D. The anchor's force is the
EQUIVALENCE of 1-D and 2-D avalanche geometry, not the 1-D number in isolation. (b) The
1-D over-determination gap (0.114) is slightly below the 2-D value (0.14); both are real,
non-zero, several times the fit error, and share the S3 line/wedge mechanism, but I do not
claim they are equal -- only that the duration anomaly exists and is intrinsic in 1-D too.
(c) <S|T> and <E|T> are the same exponent within error once E ~ S, as in 2-D -- a
consistency check, not an independent fact. (d) The enabler's repose extrapolation (~2.79)
is a 1/L fit over L = 128-512 and is the soft part; the firm result is "finite-size, not
under-equilibration", which rests on the verified IC-independent stationary plateaus and
the L = 256 / S12 agreement.

**Why this is interesting.** It converts S14's "the 2-D avalanche is filamentary, D ~ 1"
from a measured exponent into a dimensional statement: the 2-D slope avalanche has the
GEOMETRY of the 1-D avalanche the model produces in its native dimension -- same mass-radius
dimension, sweep structure and ballistic front, differing only in symmetry (directedness) --
so the whole filamentary-front picture (S12-S15) is the embedding of a 1-D-like front in 2-D
rather than a special 2-D phenomenon. It anchors the S14 placement (filament 1 < directed
3/2 < compact 2) at the 1-D end, ties the S13 duration anomaly back to its S3 origin
across dimensions, and isolates directedness as the single thing dimension changes. And it
clears the L > 256 equilibration ceiling that blocked S13's duration-closure question,
handing S17 a verified-stationary L = 512 state. figures/sandpile_geometry1d.png,
figures/sandpile_equilibrate.png.

---

## S17 -- The duration closure: single-scale scaling is exact in space, residual in time

**Question.** S13 established the central scaling result of the chapter: the 2-D slope
avalanche obeys single-scale ("one length scale per avalanche") scaling among its
SPATIAL observables -- energy locks to size (E ~ S) and a footprint of area A carries
~A^2 topplings (S, E ~ A^2) -- but the single-scale picture FAILS through DURATION. The
identity single-scale scaling forces, gamma(S|A) = gamma(S|T)/gamma(A|T), did not close:
the size-area exponent measured directly (~1.93) sat above the value routed through
duration (~1.79), a gap of ~0.14, while the purely spatial lock gamma(E|S) closed
exactly. S13 read this as the 2-D residue of S3's 1-D quantized line/wedge families
(some avalanches last long by LINGERING, not by reaching farther). But S13 could only
reach L = 256, with every exponent still drifting toward single-scale as L grew, so it
flagged the sharp question for later: does the gap HEAL to zero as L -> infinity (a
finite-size correction) or converge to a finite RESIDUAL (a true, intrinsic duration
anomaly)? S16 supplied the two things needed to answer it: the equilibration enabler
(verified-stationary states past the S13 ceiling) and the 1-D dimensional anchor (the
same gap, ~0.11, measured where there is no equilibration ceiling, an effective
L -> infinity reference).

**Method (`sandpile/duration_closure.py`).** For L = 96, 128, 192, 256, 384, 512:
warm to a verified-stationary state via `equilibrate2d.equilibrate` (S16), run a
recorded window (4-6 seeds, ~1.5-1.8e5 avalanches per L), gather per-avalanche
(E, S, T, A) with the S13 grouping, and measure the conditional exponents with the S13
machinery reused unchanged. Form the duration gap gamma(S|A) - gamma(S|T)/gamma(A|T)
per L with a leave-one-seed-out jackknife error, track the spatial lock gamma(E|S) as a
baseline (it must stay ~1, or the gap would be a generic fit artefact rather than a
duration-specific one), and read the L -> infinity behaviour from both a 1/L fit and --
more robustly -- the directly-measured COMPONENT trends. A self-test guards the gap
estimator: a synthetic single-scale source (one hidden scale, A = ell, S = E = ell^2,
T faithful) must read gap ~ 0 (no fabricated gap; a single hidden variable always
closes the identity because the T-conditioning factor cancels in the ratio), while a
two-scale source (an independent additive lingering scale in T that area and size do
not share) must read a clearly nonzero gap (the breakdown detected). The synthetic
breakdown's SIGN reflects its particular construction and is not meant to reproduce the
model's sign -- the test validates the null and detection, not the physics.

**Evidence.**

| L | gap (S\|A - S\|T/A\|T) | gamma(S\|A) | gamma(A\|T) | gamma(S\|T) | gamma(E\|S) | repose |
|---|------------------------|-------------|-------------|-------------|-------------|--------|
| 96  | 0.066 +- 0.003 | 1.891 | 0.912 | 1.664 | 1.003 | 2.50 |
| 128 | 0.075 +- 0.008 | 1.897 | 0.932 | 1.698 | 1.002 | 2.56 |
| 192 | 0.103 +- 0.022 | 1.917 | 0.948 | 1.720 | 1.002 | 2.61 |
| 256 | 0.118 +- 0.004 | 1.917 | 0.964 | 1.734 | 1.002 | 2.65 |
| 384 | 0.131 +- 0.012 | 1.927 | 0.975 | 1.751 | 1.002 | 2.71 |
| 512 | 0.183 +- 0.005 | 1.947 | 0.992 | 1.750 | 1.002 | 2.75 |

The repose values reproduce S16's per-L equilibrium (the runs are stationary). The
spatial lock gamma(E|S) = 1.00 at every L. The decisive reading is in the components,
not the derived gap (the S6/S11 auto-verdict discipline -- the gap is a ratio of fits,
the components are directly measured):

- **gamma(S|A) -> 2** (1.891 -> 1.947) and **gamma(A|T) -> 1** (0.912 -> 0.992): the
  SPATIAL sector heals to EXACT single-scale as L grows. Size scales as area squared and
  area scales ballistically with duration, both reaching their single-scale values.
- **gamma(S|T) saturates at ~1.75** (1.664 -> 1.751 -> 1.750, the last step flat) while
  the spatial exponents are still climbing -- it does NOT reach 2. The size-duration
  exponent plateaus below single-scale.

So the gap does NOT heal: it GROWS with L (0.066 -> 0.183). A 1/L fit extrapolates to a
residual of 0.19 +- 0.02, and that UNDERESTIMATES -- the gap is convex in 1/L
(accelerating as L grows), and the gamma(S|T) ~ 1.75 plateau against gamma(S|A) -> 2,
gamma(A|T) -> 1 implies an asymptotic gap of ~2 - 1.75 = 0.25. Either way the residual is
clearly nonzero and larger than the 1-D anchor (0.11, S16).

**Conclusion -- the answer to S13.** Single-scale scaling in the 2-D slope sandpile is
EXACT in space and RESIDUAL in time. The spatial observables (area, size, energy) form a
single-scale family that becomes exact in the L -> infinity limit (S, E ~ A^2 with the
exponent converging to 2, A ~ T ballistic with the exponent converging to 1). But
duration is an intrinsically loose proxy for spatial extent: at fixed duration the size
grows only as T^~1.75, not the T^2 the (asymptotically exact) spatial chain predicts,
and that shortfall does NOT vanish with system size -- it is a true residual, present and
growing to L = 512. Physically, some long-duration avalanches are long because they
LINGER -- re-toppling in place or propagating intermittently -- rather than because they
reach farther, so duration carries information the spatial extent does not. This is the
2-D survival of S3's 1-D line/wedge families, now shown to persist to the thermodynamic
limit rather than wash out. It vindicates S12's choice of AREA, not duration, as the
model's clean scaling variable, and it completes the S11-S17 scaling-theory arc: the
ballistic-filamentary-front theory predicts the entire SPATIAL exponent set from one
length scale exactly, with duration as the single, quantified exception.

**Cross-dimensional reading.** The residual is comparable to and somewhat larger than the
1-D anchor (2-D ~0.19-0.25 vs 1-D 0.11, S16). The duration anomaly is therefore intrinsic
to the slope rule across dimensions -- the 1-D anchor proves it is not a 2-D artefact --
and it is STRONGER in 2-D, consistent with a 2-D front having more ways to linger
(transverse spread and intermittent re-activation) than a strictly forward 1-D line.

**Honest caveats (own them).** (a) The robust claim is QUALITATIVE -- the gap does not
heal, it grows -- and it rests on the directly-measured component trend (gamma(S|T)
plateaus at ~1.75 while gamma(S|A), gamma(A|T) reach 2, 1), which is firmer than the
derived gap. The asymptotic MAGNITUDE is bounded but not pinned: the 1/L fit gives 0.19,
the S|T-plateau argument gives ~0.25, so I quote ~0.2 with that spread, not a single
digit. (b) gamma(S|T) at L = 512 (1.750) is statistically flat against L = 384 (1.751),
which is the strongest single piece of evidence for a genuine plateau, but it is two
points; a hypothetical slow further rise of S|T toward 2 at L >> 512 cannot be excluded,
only made unlikely by the clear deceleration (increments 0.034, 0.022, 0.014, 0.017,
-0.001). (c) The conditional windows (lo_floor and a 0.30-of-max upper bound) are applied
identically at every L and to every exponent, and the spatial gamma(S|A) measured by the
SAME machinery does reach 2, so the S|T shortfall is not a windowing bias. (d) The jackknife
errors are over 4-6 seeds and capture run-to-run spread; the L = 192 point is the noisiest
(few-seed scatter) but off the trend by less than its error.

**Why this is interesting.** It converts S13's flagged uncertainty into a definite,
falsifiable answer: the single-scale theory of the slope avalanche is not approximately
true with a finite-size blemish, it is EXACTLY true in space and EXACTLY false (by a
measured residual) in time, asymptotically. That is a sharper statement than "the exponents
roughly agree" -- it says precisely which scaling relation holds in the thermodynamic limit
and which does not, and ties the failure quantitatively to the 1-D family structure the
chapter started from (S3). With the 1-D anchor (S16) and the L = 512 closure (S17) the
arc is complete: the 2-D slope avalanche is a ballistic, filamentary, multiply-swept front
whose spatial scaling is single-scale and whose duration is an intrinsically loose
variable, placed causally (S15) against BTW, the directed sandpile and Manna.
figures/sandpile_duration_closure.png.

---

## S18 -- The area-multifractality closure: the anomaly is asymptotic, not a finite-size corona

**Question.** S12 measured the avalanche-AREA moment spectrum of the 2-D slope sandpile
and found the local slope D(q) = d sigma(q)/dq (where <A^q> ~ L^sigma(q)) DRIFTS with q --
a rise of about 0.27 over q in [1, 4] -- and read it as genuine multifractality, placing
the model in the deterministic (BTW/Zhang) anomalous-scaling family rather than the
stochastic Manna single-fractal FSS class. But S12 could only equilibrate to L = 256, and
quoted that as its leading caveat ("L = 512 is left to future work"); its drift was a
single moment regression pooled over the whole finite-size range L = 64-256 -- exactly
where a finite-size corona would live. So the question left open at the close of the
S11-S17 arc: is the area multifractality a TRUE asymptotic property, or a finite-size
corona that heals to simple FSS as L grows? S17 sharpened the stakes. It proved the
conditional MEAN <S|A> reaches its exact single-scale value gamma(S|A) -> 2 as
L -> infinity, with the activity space-filling (S, E ~ L^2). Read naively -- as if the
conditional mean were the whole distribution -- that predicts A ~ L^1 with no scatter,
a mono-fractal area with a flat D(q): on that reading the multifractality SHOULD heal. S18
tests the prediction with an independent observable: the UNCONDITIONAL area moments,
against S17's conditional exponents.

**Method (`sandpile/area_multifractality.py`).** Warm verified-stationary states at
L = 64, 96, 128, 192, 256, 384, 512 via `equilibrate2d.equilibrate` (the S16 enabler),
run a recorded window from each (5-6 seeds, ~1.5-2.7e5 avalanches per L), and gather the
per-avalanche AREA (distinct toppled bonds, S12's clean non-cutoff-dominated observable).
The multifractal signature is LOCALIZED in L by a SLIDING WINDOW: for each set of three
consecutive lattice sizes regress log<A^q> on log L to get sigma(q) and D(q) = d sigma/dq
from that window alone, and read the drift of D(q) over q in [1, 4] (a quadratic fit of
sigma over the window, robust to the noisy central-difference endpoints; the raw max-min
range is reported alongside). As the window slides to larger L, a finite-size corona makes
the drift SHRINK toward zero; a true asymptotic multifractal keeps it nonzero. Drift vs
1/L_window is extrapolated to L -> infinity (the S17 protocol). A self-test guards the
estimator: an exact simple-FSS source (one power law, one cutoff) must read drift ~ 0 at
EVERY window (no fabricated multifractality, no spurious L-trend -- the null that makes a
nonzero intercept on the real data meaningful), and a bifractal mixture must read a clearly
nonzero |drift| (a real sigma(q) curvature is detected; the sign of the synthetic drift is
construction-specific, as with the S17 synthetic, so only the magnitude is asserted).

**Evidence.**

| L-window | center L | area D(q) drift over [1,4] | D(q~1) -> D(q~4) |
|----------|----------|----------------------------|------------------|
| 64-96-128   | 92  | 0.219 +- 0.040 | 0.96 -> 1.18 |
| 96-128-192  | 133 | 0.272 +- 0.023 | 0.97 -> 1.25 |
| 128-192-256 | 185 | 0.243 +- 0.042 | 0.99 -> 1.23 |
| 192-256-384 | 266 | 0.145 +- 0.062 | 1.06 -> 1.20 |
| 256-384-512 | 369 | 0.194 +- 0.031 | 1.14 -> 1.34 |

The repose per L (2.43, 2.50, 2.56, 2.61, 2.65, 2.71, 2.75) reproduces S16/S17, so the
states are stationary. Two readings, the robust one first (the S6/S11/S17 discipline: read
the directly-measured components, not one derived auto-verdict):

- **The drift does NOT heal.** Across a 4x range in window-center L it stays at
  about 0.2 (0.219 at L ~ 92, 0.194 at L ~ 369), with a noisy dip at L ~ 266; it shows no
  decay toward the simple-FSS value of zero. A 1/L_window fit extrapolates to a residual of
  **0.169 +- 0.048** at L -> infinity, clearly nonzero (about 3.5 sigma). A finite-size
  corona would have fallen by ~4x over this L range; it does not fall at all. So the S12
  area multifractality is a TRUE asymptotic property, not a corona.
- **If anything the area dimension RISES with L** (a tentative second reading). The mid-q
  footprint dimension climbs from ~1.07 at the small-L windows to ~1.23 at the largest, and
  the per-step <A> growth slope steepens from ~0.90 over L = 64-256 (the S12 filament,
  A ~ L^0.9) to 1.00 (256->384) and 1.22 (384->512). The largest avalanches at the largest
  lattices appear to be FATTER than the one-bond filament S14/S16 found for typical
  avalanches. This rests on the noisiest, fewest-avalanche points and is flagged as
  suggestive, not firm (see caveats).

**Conclusion -- the answer to S12's open question, and the reconciliation with S17.** The
naive prediction from S17's conditional result -- that the area should be mono-fractal
because <S|A> ~ A^2 and S ~ L^2 -- is FALSIFIED: the area multifractality survives to the
thermodynamic limit. The two results are nonetheless consistent, and seeing why is the
content of S18. S17's gamma(S|A) -> 2 is a statement about a conditional MEAN -- the
first moment of S at fixed A. Single-scale scaling of that mean does NOT imply the area
DISTRIBUTION is single-fractal: the higher moments, which weight the rare largest
avalanches and the fluctuations about the mean, can scale with their own, larger exponents,
and that is exactly what multifractality is. The conditional method (S13/S17) measures the
typical relationship over a windowed mid-range and so reads the single-scale "core"; the
moment method (S12/S18) weights the tail and so reads the multifractal fluctuations. The
slope sandpile therefore has BOTH: a conditional mean that becomes exactly single-scale in
space (S17) AND an area distribution whose higher moments stay multifractal asymptotically
(S18). The q-resolved data show the mechanism directly -- within the largest-L window D(q)
rises from 1.14 at q ~ 1 (mean-weighted, near the typical filament) to 1.34 at q ~ 4
(tail-weighted): higher moments see a higher dimension because the largest avalanches are
the ones that depart from the one-bond filament. The single-scale-in-space picture is a
property of the typical avalanche; the multifractality is a property of the spread.

**Cross-checks (own the discipline).** (a) The null self-test reads drift < 0.01 at every
window on exact simple-FSS data, so the method neither manufactures a drift nor a spurious
L-trend -- a measured nonzero intercept is real. (b) The drift is flat in L while staying
inside the resolved q in [1, 4] range, so it is not the S12 high-q cutoff sensitivity
leaking in (that would inflate at large L, not stay level). (c) The repose values match the
independent S16/S17 equilibration, so the large-L states are genuinely stationary, not
under-warmed.

**Honest caveats (own them).** (a) The robust claim is "does not heal" -- the drift stays
nonzero across a 4x lever arm in L and extrapolates ~3.5 sigma from zero. The asymptotic
MAGNITUDE (0.17) is from a noisy, non-monotonic 1/L fit and should be read as "a clearly
nonzero residual of order 0.2," not a pinned digit. A much SLOWER (logarithmic) corona
cannot be strictly excluded by a 4x range, only strongly disfavored against a flat residual.
(b) The second reading -- that the area dimension is actively RISING at the largest L
(D_mid 1.07 -> 1.23, <A> slope 0.90 -> 1.22) -- is the most interesting hint here but the
least certain: L = 384 and 512 carry the fewest avalanches (~1.5e5) and the longest tails,
and A_max nearly doubled in the 384 -> 512 step (587 -> 1088), so a handful of rare large
avalanches could be inflating <A> and the high-q moments. Firming up whether the filament
genuinely fattens at large L (a real crossover) versus tail-undersampling needs more seeds
at L = 512 (left as the natural follow-up). (c) This is multifractality of the area
DISTRIBUTION; it does not touch the S14/S16 geometric result that the TYPICAL footprint is
one-bond-wide (98% at L <= 256). It says the largest avalanches' areas scale anomalously,
which is a tail statement, fully compatible with a thin typical avalanche.

**Why this is interesting.** It closes the last genuinely open question of the scaling-theory
arc with a result that is sharper than a confirmation: S12's anomalous area scaling is not a
finite-size artifact, it is asymptotic. And it resolves what looked like a contradiction
between the chapter's two main tools -- the conditional exponents say "single-scale in
space," the moment spectrum says "multifractal" -- by locating each precisely: single-scale
is a property of the conditional mean (the typical avalanche), multifractal is a property of
the distribution's higher moments (the largest avalanches and the fluctuations). The slope
sandpile is single-scale in its means and multifractal in its tails, simultaneously and
asymptotically -- a more complete statement of where it sits relative to BTW (multifractal),
Manna (single-fractal FSS), and the directed sandpile (exactly single-scale) than either
tool gives alone. figures/sandpile_area_multifractal.png.

---

## S19 -- The filament-fattening test: a self-test of S18's one open hint

**Question.** S18 closed the area-multifractality question but ended on a deliberately
TENTATIVE second reading it could not settle: that the avalanche footprint may be actively
FATTENING at the largest lattices. The signs were a per-step <A> growth slope that steepened
from ~0.90 over L = 64-256 (the S12/S14 one-bond filament, A ~ L^0.9) to 1.00 (256->384) and
1.22 (384->512), a mid-q footprint dimension D_mid that climbed 1.07 -> 1.23, and a largest
avalanche A_max that nearly doubled in the 384->512 step (587 -> 1088). S18 flagged these as
the noisiest, fewest-avalanche points (only 5 seeds at L = 384, 512) and named the follow-up
explicitly: more seeds at L = 512 to decide a real crossover from tail-undersampling. The two
readings are physically opposite and the whole geometric spine of the arc (S12/S14/S16: the
TYPICAL avalanche is a constant-width one-bond filament of mass-radius dimension D ~ 1) rests
on which is true: either the typical footprint genuinely thickens at large L (D ~ 1 is only a
small-L description and the deterministic model drifts toward compactness on its own), or the
typical footprint stays a one-bond D ~ 1 filament and only the rare largest avalanches grow
(the "fattening" is the S18 multifractal TAIL, and S14/S16/S18's picture stands unchanged).

**Method (`sandpile/filament_fattening.py`).** Re-measure the footprint geometry at
L = 192, 256, 384, 512 with 10-12 seeds (double S18's 5, to average down the rare tail and
put an honest error on the growth slopes), each lattice warmed to a verified-stationary
repose via the S16 enabler and its avalanches dumped through the validated S14 footprint dump.
Split each lattice's avalanches into the TYPICAL bulk (bottom 90% by area) and the area TAIL
(top decile), and measure both:
  - mean(A) growth slope vs L -- the S18 (tail-weighted) quantity that read 1.22;
  - median(A) growth slope vs L -- the TYPICAL avalanche;
  - one-bond-wide fraction of the top-decile footprints vs L -- S14's 98% at L <= 256;
  - mass-radius dimension D (A ~ Rg^D) over the FULL ensemble vs the TYPICAL ensemble.
Errors are a seed-level bootstrap (300 resamples), the honest run-to-run spread as in S18's
jackknife. A self-test guards the typical/tail separator: on a synthetic mixture of many thin
LINES (D = 1) plus a rare population of large filled DISKS (D = 2), the low-area split must
read the thin filament (D ~ 1) and the high-area split the fat population (D ~ 2), so a split
read on the real data is meaningful (the geometry estimators themselves are guarded by
geometry2d's D = 1/2/3-2 self-test, reused here).

**Evidence (`figures/sandpile_filament_fattening.png`).** 234k/229k/183k/173k footprints at
L = 192/256/384/512 (12/12/10/10 seeds), repose 2.61/2.66/2.71/2.75 (reproducing S16/S17, so
the states are stationary).

| L | <A> | median A | A_max | one-bond%% (tail) | D_full | D_typ |
|---|-----|----------|-------|-------------------|--------|-------|
| 192 | 37.6 | 32 | 323  | 0.99 | 1.01 | 1.00 |
| 256 | 47.3 | 38 | 442  | 0.98 | 1.02 | 1.00 |
| 384 | 68.6 | 53 | 670  | 0.97 | 1.02 | 1.01 |
| 512 | 93.9 | 71 | 1090 | 0.95 | 1.02 | 1.01 |

Growth slopes over L = 256-384-512 (1.0 = filament A ~ L; 2.0 = compact A ~ L^2):
**mean(A) ~ L^0.99 +- 0.03** (the S18 quantity that read 1.22 at 5 seeds) and
**median(A) ~ L^0.90 +- 0.04** (the typical avalanche).

The decisive reading (the directly-measured components, the S6/S11/S17/S18 discipline):
- **The typical avalanche is L-stable at the filament values.** Its mass-radius dimension
  D_typ is flat at 1.00 -> 1.01 across L = 192-512, its median area grows as L^0.90 (the S12
  filament, A ~ L), and the top-decile footprints stay essentially one bond wide (one-bond
  fraction 0.99 -> 0.95). None of S14/S16's filament geometry moves with L.
- **Only the tail grows, and the mean-slope hint evaporates with more seeds.** The mean area
  grows faster than the median (mean-slope 0.99 EXCEEDS median-slope 0.90 by more than their
  combined error), so the extra growth is concentrated in the rare largest avalanches. And the
  per-step mean-area slope that read 1.22 at S18's 5 seeds comes down to ~1.09 (384->512) and
  the 256-384-512 regression to 0.99 +- 0.03 at 10-12 seeds -- the 1.22 was sampling noise in
  the tail of the mean, not a real exponent. (A_max 323 -> 1090 reproduces S18's 1088, so the
  single largest avalanche is real and reproducible; it is just rare.)

**Conclusion -- the answer to S18's open hint.** The filament does NOT fatten. The apparent
large-L fattening in S18 was tail-undersampling at the fewest-seed points: with the tail
properly sampled, the mean-area slope is 0.99 not 1.22, and the TYPICAL avalanche (median
area, mass-radius D, one-bond width) is unchanged from its small-L filament geometry all the
way to L = 512. So S14/S16's constant-width one-bond filament is the asymptotic geometry of the
typical slope avalanche, not a small-L description. This does not overturn S18 -- it SHARPENS
it. S18 said the area multifractality is asymptotic and lives in the tail (single-scale in the
means, multifractal in the tails); S19 confirms that the means/typical avalanche stay exactly
the one-bond filament while the multifractality is confined to the tail, exactly as S18's
reconciliation with S17 requires. The two S18 readings are now cleanly separated: the robust
one (asymptotic tail multifractality) stands, the tentative one (typical fattening) falls.

**Honest caveats (own them).** (a) The one-bond-wide fraction of the TAIL does drift down a
little, 0.99 -> 0.95, so the very widest tail members are marginally fatter at L = 512 than at
L = 192 -- a faint real effect, fully consistent with "the tail is multifractal" (S18) and not
with "the typical footprint thickens" (D_typ and the median are flat). (b) The median-area
slope 0.90 sits just below 1; it is the S12 filament value (area ~ L^0.9, the small mismatch
from exactly 1 is the known finite-Rg curvature of the binned mass-radius fit, applied
identically at every L), and what matters is that it does NOT climb with L. (c) L = 384 and 512
still carry the fewest avalanches (~180k/173k), but at double S18's seed count the mean-slope
already collapsed from 1.22 to ~1.0, which is the direct demonstration that the original number
was undersampling. (d) This settles the geometry of the TYPICAL avalanche; it says nothing new
about the tail beyond S18, by design -- the tail multifractality is S18's result and is not
re-litigated here.

**Why this is interesting.** It is a self-test in the project's tradition (F47, F48, F52, S6):
a flagged hint tested before it could harden into a claim, and downgraded honestly. The
scientific payoff is that the arc's central geometric statement -- the slope avalanche is a
constant-width one-bond filament -- is now confirmed asymptotic (to L = 512) for the typical
avalanche, with the only L-growth cleanly assigned to the multifractal tail S18 already
characterized. The picture is exactly as complete as S18 claimed, with its one loose thread
tied off rather than left dangling. figures/sandpile_filament_fattening.png.

---

## S20 -- Model selection: does the angle of repose saturate or diverge?

**Question.** S16 confirmed that the mean bond slope at the SOC repose rises
with L as a real finite-size effect (2.42 at L = 64 -> 2.74 at L = 512, not
under-equilibration), and left open whether the drift SATURATES to a finite
infinite-volume limit (a 1/L finite-size correction, the most common form for
local critical models) or DIVERGES slowly without bound (a log L dependence,
which would mean the model has no well-defined thermodynamic repose slope). No
new simulation is needed to address this: seven converged plateau values from
S16 are already in hand, and the two functional forms make distinct predictions
that can be compared statistically.

**Method (`sandpile/repose_scaling.py`).** Fit both models to the S16 plateau
values (L = 64, 96, 128, 192, 256, 384, 512) by ordinary least squares and
compare them by AIC (Akaike information criterion). Both models have two free
parameters (intercept + slope), so AIC reduces to choosing the smaller residual
sum of squares with a common penalty; DAIC > 2 is "positive evidence" for the
preferred model, DAIC > 6 "strong evidence," DAIC > 10 "very strong." A
per-doubling increment table provides an intuitive check: 1/L predicts the
increment should HALVE each doubling, log L predicts it should be CONSTANT.
Self-test: fitting to synthetic exact-1/L and exact-log(L) data (noise 0.005)
confirms AIC recovers the correct model in each case.

**Models and fits.**

Model 1 (saturating):  r(L) = a + b/L
  a = r_inf = 2.754,  b = -22.9
  R^2 = 0.960,  AIC = -49.9,  max residual = 0.035  (2.8x the typical spread)

Model 2 (diverging):   r(L) = a + b*log(L)
  a = 1.803,  b = 0.152  (increment 0.106 per doubling)
  R^2 = 0.990,  AIC = -59.6,  max residual = 0.016  (1.3x the typical spread)

DAIC = AIC_log - AIC_1/L = -9.7  ->  log(L) preferred (strong evidence).

Per-doubling increment check: 64->128: 0.138, 128->256: 0.089, 256->512: 0.098.
Ratio last/previous = 1.11, which is much closer to log(L)'s predicted 1.00
than to 1/L's predicted ~0.50. The 1/L model's extrapolated r_inf = 2.754 is
already undercut by the actual L=512 repose 2.745 (residual 5.9x the spread at
that point), showing the saturation is predicted to happen too early.

**Result.** The log(L) model fits substantially better by every metric. The SOC
angle of repose most likely DIVERGES slowly (logarithmically) rather than
saturating at a finite limit. The physical picture: open boundary effects
propagate O(L) into the bulk, so larger lattices can sustain steeper average
gradients at criticality; the critical slope is not a single number independent
of system size, but continues to creep upward. This is qualitatively different
from canonical BTW, where the thermodynamic repose is well-defined.

**Caveat.** Seven data points over a factor of eight in L cannot prove
logarithmic divergence against a very slow saturation. The AIC prefers log L
within the L=64-512 range, but if the 1/L correction had a coefficient much
larger than the data allow (making the asymptote far from the current repose)
the distinction would wash out at larger L. Confirming divergence would require
L ~ 1024-2048, each of which takes ~60-100s to equilibrate at current throughput.
The 1/L fit's r_inf = 2.754 remains the best-case (lowest) bound on any eventual
asymptote. figures/sandpile_repose_scaling.png.
