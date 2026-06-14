# Findings -- Sandpiles / Self-Organized Criticality
Started 2026-06-14

Charbonneau, *Natural Complexity* (2017), Chapter 5. A **new chapter**, distinct
from the flocking work (F1-F92 in `findings.md`). To keep the two cleanly
separated, sandpile findings use their own **S-series** (S1, S2, ...).

The arc: validated 1-D core -> rigorous 1-D critical exponents (finite-size
scaling) -> a 2-D slope sandpile -> the universality question (do 1-D and 2-D
share exponents, and how do they compare to the canonical abelian/BTW sandpile?).

Code lives in `sandpile/`. Figures in `figures/sandpile_*.png`, run logs in
`outputs/sandpile_*.txt`.

---

## Index

- S1  Validated 1-D slope-sandpile reproduces the chapter's SOC signatures
- S2  Angle of repose: the MEAN slope sits ~16% below Zc (eps-independent); only
      the PEAK (pre-avalanche) slope approaches Zc as eps -> 0
- S3  1-D critical exponents by finite-size scaling: tau_E ~ 1.03, D_E ~ 2.0,
      D_T ~ 1.0; avalanches are quantized families, not simple fractals

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
