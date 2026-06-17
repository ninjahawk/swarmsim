# Findings -- Earthquakes / the Olami-Feder-Christensen model
Started 2026-06-16

Charbonneau, *Natural Complexity* (2017), Chapter 8. A **new chapter**, distinct from
the flocking work (F-series in `findings.md`) and the sandpile work (S-series in
`findings_sandpile.md`). Earthquake findings use their own **E-series** (E1, E2, ...).

This chapter is the direct continuation of the sandpile conservation thread. The
sandpile findings S5/S7 established that **bulk conservation is necessary for true
self-organized criticality** in the slope sandpile: with conservation the avalanche
cutoff grows with system size (no characteristic scale), but any bulk dissipation
truncates the distribution at a fixed, dissipation-set size. The Olami-Feder-
Christensen (OFC) earthquake model is the canonical system in which that exact
question is "perennially debated", because its bulk conservation is a continuous
knob: the parameter `alpha` (0 <= alpha <= 0.25) is the fraction of a toppling node's
force passed to *each* of its four neighbours, so 4*alpha is the conserved fraction
and (1 - 4*alpha) is dissipated per topple. At alpha = 0.25 redistribution is
conservative (dissipation only at the open boundary, exactly like the sandpile); for
alpha < 0.25 the bulk dissipates.

**Model (book eqs. 8.10-8.14).** An L x L lattice of real "force" values F[i,j],
random in [0, Fc] initially (Fc = 1). Uniform driving adds delta_f to every node
until some node reaches Fc; that node topples (F -> 0, each neighbour += alpha*F),
which can push neighbours over threshold, cascading into an avalanche. Open
boundaries (the ghost border held at 0) leak force off the lattice. Stop-and-go
driving (forcing pauses during an avalanche), exactly as in the Ch. 5 sandpile. This
is the earthquake analogue of the slope sandpile, but with the stability criterion on
the nodal VALUE rather than its slope -- and that single change, plus tunable
conservation, produces markedly different behaviour.

**Headline.** OFC reproduces the Gutenberg-Richter power law, and `alpha` tunes the
exponent continuously -- but, unlike the sandpile, the nonconservative model is
deterministic and develops spatially synchronized domains that drive *quasi-periodic*
recurrent avalanching, a temporal structure the sandpile entirely lacks. The
conservation knob lets us watch true criticality (system-spanning avalanche cutoff)
give way to a dissipation-set characteristic scale -- the S5 result made quantitative
on the canonical earthquake model.

Code lives in `earthquake/` (self-contained; imports nothing from the flocking or
sandpile code). Core model + validation: `ofc.py`. Figures in `figures/ofc_*.png`,
run logs in `outputs/ofc_*.txt`.

---

## Index

- E1  Validated OFC reproduces Gutenberg-Richter (fig. 8.7): conservative alpha=0.25
      gives a clean power law (PDF slope -1.19); lowering alpha steepens the slope and
      shrinks the cutoff
- E2  Finite-size scaling and the conservation test (cf. S5/S7): the avalanche cutoff
      scales with L when conservative and saturates to a dissipation-set size when not
- E3  Recurrent (quasi-periodic) avalanching and spatial synchronization -- the
      qualitative contrast with the temporally uncorrelated sandpile
- E4  Avalanche size-duration relation E ~ T^gamma (Exercise 5)
- E5  Forcing skip-ahead speedup (Exercise 3): bit-identical, delta_f-dependent
- E6  (self-test) Does a stochastic alpha break the quasi-periodicity? (Exercise 4)
- E7  Grand Challenge: earthquake prediction from the recurrence rhythm (Exercise 6)

Model parameters unless noted: Fc = 1, delta_f = 1e-4, open boundaries.

---

## Finding E1: A validated OFC implementation reproduces the Gutenberg-Richter law (fig. 8.7) -- conservative alpha=0.25 gives a clean power law (PDF slope -1.19), and lowering alpha steepens the slope and shrinks the cutoff

**Setup.** The chapter's central result: the OFC avalanche-size PDF is a power law (the
Gutenberg-Richter law, eq. 8.1), and the conservation parameter alpha tunes its slope.
We reproduce fig. 8.7 on a 128x128 lattice, delta_f = 1e-4, open boundaries, for
alpha = 0.10, 0.20, 0.25, comparing the fitted PDF slopes to the book's reported values
(-3.34, -1.92, -1.19). The model and its engine are validated first by six self-tests
(ofc.py): single interior/corner topples, exact dissipation bookkeeping (a bulk topple
loses exactly (1-4*alpha)*F), conservation under periodic boundaries at alpha=0.25,
post-avalanche stability, and -- importantly -- that the Exercise-3 forcing skip-ahead
is bit-identical to one-step forcing.

**Result.** All three slopes match the book. Conservative alpha=0.25 gives a clean
power law spanning five decades with slope -1.19 (book -1.19; the cumulative
Gutenberg-Richter exponent b ~ 1). alpha=0.20 gives -1.87 (book -1.92), and alpha=0.10
gives -3.52 (book -3.34). The slope steepens and the cutoff shrinks monotonically as
alpha decreases (maxE ~ 4x10^5, 7600, 380 at alpha = 0.25, 0.20, 0.10), exactly the
fig.-8.7 trend.

**Implication.** The model is validated, and the qualitative physics is clear: a
statistically stationary state must dissipate force at the same average rate the
forcing injects it. At alpha=0.25 the only sink is the open boundary, so avalanches
must occasionally span the whole lattice to discharge there, giving large events and a
shallow (-1.19) slope. As alpha falls, the bulk dissipates (1-4*alpha) per topple, so a
stationary balance is reached with more frequent small avalanches and a steeper slope
with a smaller cutoff. A practical lesson recorded for later findings: the strongly
dissipative cases need very long warmup to reach the synchronized stationary state --
alpha=0.10 reached only maxE=61 with a (too steep) slope -3.5...-4.5 after 4x10^5
avalanches, and required ~3x10^6 avalanches to mature to maxE=380 and slope -3.52. The
synchronization domains (E3) that produce the larger recurrent avalanches coarsen
slowly, the more so the lower alpha. earthquake/ofc_gr.py

## Finding E2: Finite-size scaling makes the S5 conservation result quantitative on OFC -- the avalanche cutoff GROWS with system size when (near-)conservative and SATURATES to a dissipation-set scale when strongly dissipative

**Setup.** The deep payoff of the chapter and the direct test of sandpile finding S5
(bulk conservation is necessary for true SOC). For alpha = 0.10, 0.20, 0.25 we sweep
the lattice side L in {16, 24, 32, 48, 64, 96} and measure the avalanche-size cutoff
via the moment ratio <E^2>/<E> (which tracks the upper cutoff for a power law with
exponent < 2). The question: does the cutoff GROW with L (a true critical system,
avalanches limited only by the box) or SATURATE (a subcritical system with a finite,
dissipation-set characteristic size below L)? Nonconservative warmup scaled as
max(20000, 25*L^2) events to reach the synchronized stationary state. Exercise 2.

**Result.** A clean conservation-tuned transition. Fitting cutoff ~ L^D:
- alpha=0.25 (conservative): D = 3.19 -- the cutoff climbs steeply with L (from 102
  at L=16 to 31700 at L=96), and the mean avalanche size <E> itself grows ~L^2.5
  (23 -> 2017). Avalanches are limited only by the system: true criticality.
- alpha=0.20: D = 1.95 -- the cutoff grows close to L^2, still critical-like over this
  range (the literature places OFC's critical regime at alpha above ~0.18-0.20).
- alpha=0.10 (strongly dissipative, 60% lost per topple): D = -0.15 -- the cutoff is
  FLAT across the whole L range (2.5-4.0) and maxE stays ~50 regardless of L. The mean
  avalanche size is also L-independent (~1.5). The avalanches are capped by a
  dissipation-set characteristic size, not by the system: subcritical.

**Implication.** This is exactly the S5 signature, now demonstrated on the canonical
earthquake model with conservation as a continuous knob. In the slope sandpile S5
found a binary contrast (conserve -> cutoff ~ N^2; dissipate -> cutoff truncated); OFC
shows the same physics as a TUNED crossover, from a subcritical, dissipation-limited
state at low alpha (a characteristic earthquake size, set by how far a disturbance
propagates before the lost force starves it) to genuine scale-free criticality as
alpha -> 0.25 (no characteristic size, avalanches span the system). The mean-avalanche
panel sharpens the mechanism: at alpha=0.25 the only sink is the open boundary, so <E>
must grow with area to evacuate the forcing input, whereas at alpha=0.10 the bulk
dissipates locally and <E> is L-independent. Caveat: alpha=0.10's ABSOLUTE cutoff is
still slowly growing with run length (E1), but its L-INDEPENDENCE -- the criticality
diagnostic -- is robust across the sweep; alpha=0.20 is critical-like over L<=96 but
its status at much larger L is the genuinely open question OFC is debated over.
earthquake/ofc_fss.py

## Finding E3: The nonconservative OFC model is deterministic and develops synchronized spatial domains that drive quasi-periodic recurrent avalanching -- the qualitative contrast with the temporally uncorrelated sandpile -- with a recurrence period that shrinks with alpha and vanishes at conservation

**Setup.** The book's headline qualitative result (secs. 8.3-8.4): unlike the
stochastically-forced slope sandpile, OFC is fully deterministic (the only randomness is
the initial condition), and at nonconservative alpha it organizes into spatial DOMAINS
of locked, near-equal nodal values that collapse and rebuild near-periodically, giving
recurrent avalanching. We measure the recurrence period vs alpha from the
autocorrelation of the iteration-indexed toppling-activity series (N=128, delta_f=1e-4),
and record the lattice as it synchronizes from a random start.

**Result.** Clean reproduction of figs. 8.4-8.6. The autocorrelation of the avalanche
activity has a strong damped-oscillation peak for nonconservative alpha and decays
immediately (no peak) at the conservative alpha=0.25. The raw recurrence period (all
iterations) shrinks with alpha -- 12800, 10600, 7000 iterations at alpha = 0.10, 0.15,
0.20 -- and vanishes at 0.25 (the alpha=0.15 raw period matches the book's fig.-8.4
~10960). Converting to the book's forcing-corrected period (subtracting the avalanching
fraction f_av of iterations, which rises with alpha: 0.53, 0.62, 0.69, 0.98) gives 6025,
3982, 2145 iterations -- matching the book's tabulated 6435, 4002, 2165 almost exactly,
and the alpha=0.15 avalanching fraction 0.62 matches the book's "63% of iterations spent
avalanching". The lattice snapshots reproduce fig. 8.6: a random salt-and-pepper start
coarsens over ~10^6 iterations into large synchronized domains with characteristic
fault-line textures.

**Implication.** This is the deepest qualitative difference from the chapter-5 sandpile.
The sandpile's avalanches are temporally UNCORRELATED -- each is an independent response
to the slow random forcing, and the S-series never found inter-avalanche structure. OFC,
being deterministic and nonconservative, instead synchronizes: a contiguous domain of
equal nodal values stays locked under both the uniform forcing and the redistribution
(equal neighbours map to equal values), so it collapses as a unit on a near-fixed period
set by how long forcing takes to refill it (~Fc/delta_f, reduced by the jumps an average
node receives from avalanching neighbours, which is why the period shrinks as alpha
raises those jumps). The recurrence weakens and vanishes as alpha -> 0.25 because near
conservation almost every iteration is spent avalanching (f_av -> 0.98), so there is no
quiescent forcing phase to set a clock. The same domain structure is what makes the
conservative case critical (E2) and what a forecaster tries to exploit (E7) -- and what a
tiny disorder destroys (E6). earthquake/ofc_quasiperiodic.py

## Finding E4: The avalanche size-duration relation E ~ T^gamma steepens toward T^2 as conservation increases -- avalanches become more compact and space-filling near the conservative limit

**Setup.** For alpha = 0.15, 0.20, 0.25 we record both avalanche size E (total
topplings) and duration T (number of synchronous relaxation sweeps) and fit the mean
relation E ~ T^gamma (Exercise 5). Both quantities are defined identically across all
alpha, so the comparison across the conservation parameter is clean -- unlike the
sandpile size measure, which counted bond topplings differently from the canonical
model (the S4/S6 caveat).

**Result.** E ~ T^gamma with gamma rising monotonically toward 2 as alpha increases:
gamma = 1.47 (alpha=0.15), 1.64 (alpha=0.20), 1.89 (alpha=0.25), with the duration
range growing from maxT=57 to maxT=1642 over the same sequence.

**Implication.** gamma > 1 means the toppled area accumulates faster than the
avalanche's lifetime: the active front is a compact, roughly 2-D region that grows as
it spreads, so each successive sweep topples more nodes. As alpha -> 0.25 the front
becomes near space-filling (gamma -> ~2, the value for a compact front whose area
grows as the square of its radius and hence of its lifetime), consistent with the
conservative avalanches being system-spanning (E2). At lower alpha the dissipation
fragments and truncates the front, so the same duration yields fewer topplings (lower
gamma). This mirrors the sandpile dimensional result that E ~ T^2 for a 1-D wedge
avalanche (S3); here the exponent interpolates with the conservation parameter.
earthquake/ofc_et.py

## Finding E5: The forcing skip-ahead (Exercise 3) is bit-identical and gives a delta_f-dependent speedup that is modest here because OFC avalanching, not forcing, dominates the iteration count

**Setup.** Because the OFC forcing is deterministic and uniform, the number of forcing
iterations before the next avalanche is exactly (Fc - max F)/delta_f, so one can jump
straight to the next instability instead of adding delta_f one step at a time
(Exercise 3). The `ofc.run` engine does this by default; this measures the speedup
against naive one-step forcing on N=48, alpha=0.20, for delta_f = 1e-3, 1e-4, 1e-5.

**Result.** The avalanche-size series are bit-identical for all delta_f (the skip
changes only run time). The speedup grows as delta_f shrinks -- 1.1x at delta_f=1e-3,
1.1x at 1e-4, 2.0x at 1e-5 -- because the skip collapses the (Fc - max F)/delta_f idle
forcing steps between avalanches into a single array add, and that idle count grows as
delta_f falls.

**Implication.** The speedup is real but modest in this regime, and the reason is
itself informative: in the OFC model a large fraction of iterations are spent
avalanching rather than forcing (the book notes ~63% avalanching in its alpha=0.15
run), so collapsing the forcing iterations can only buy so much. This is unlike a
model where slow forcing dominates the iteration budget. The skip is still what makes
the long stationary runs and finite-size scaling of this chapter tractable, and it is
the OFC counterpart of the sandpile fast engine (S9), but its ceiling is set by the
avalanche workload, not the forcing. earthquake/ofc_speedup.py

## Finding E6: (self-test) A mildly stochastic conservation parameter breaks the OFC quasi-periodicity -- CONFIRMED -- because synchronization needs the redistribution to map equal neighbours to equal values exactly

**Setup.** The recurrent avalanching of the nonconservative OFC model (E3) rests on
spatial domains of EXACTLY equal nodal values that stay locked because the
deterministic forcing and the fixed-alpha redistribution preserve equality (book eq.
8.16: two equal neighbours stay equal under a topple). Exercise 4 asks whether drawing
a fresh alpha uniformly in [0.14, 0.16] at every toppling node -- so equal neighbours no
longer map to equal values -- destroys the synchronization and washes out the
periodicity. Pre-registered prediction (self-test tradition of the flocking F47/F81 and
sandpile S6 checks): the recurrence peak should collapse.

**Result.** CONFIRMED. The autocorrelation recurrence peak of the toppling-activity
series drops from strength 0.439 (fixed alpha=0.15) to 0.193 under alpha ~ U[0.14,0.16]
-- 44% of the fixed-alpha peak -- and the largest avalanche shrinks from 433 to 252. A
+/-0.01 per-topple jitter strongly suppresses the periodic component.

**Implication.** The quasi-periodicity is a fragile, fine-tuned consequence of EXACT
determinism, not a robust dynamical attractor. Synchronization needs equality of
neighbouring nodal values to be an exact fixed point of the update; per-topple alpha
noise removes that fixed point, so the domains cannot stay locked and the recurrent
rhythm erodes. This sharpens the contrast with the chapter-5 sandpile, whose SOC is
robust to its stochastic forcing precisely because it relies on no such synchronization,
and it tempers the prediction prospect (E7): the very regularity one would exploit to
forecast is the first casualty of any disorder in the redistribution physics.
earthquake/ofc_stochastic_alpha.py

## Finding E7: Grand Challenge -- the OFC recurrence rhythm gives real but modest earthquake-prediction skill (1.5-1.7x above chance), yet fails on exactly the largest events one most wants to predict

**Setup.** The chapter's Grand Challenge (Exercise 6): from the first half of an
alpha=0.15 avalanche time series, learn the recurrence rhythm and characteristic
large-event size, then forecast the timing and size of the large events (E > 20% of the
training maximum) in the second half. A "good" forecast gets timing within +/-100
iterations and amplitude within +/-25% (book criteria). The forecaster is deliberately
simple and uses the time series only (no peeking at the lattice state): it phase-locks
to the recurrence cycle P (the raw autocorrelation period) and predicts one recurrent
largest event per cycle, re-anchoring each cycle on the biggest event actually observed
so far (online forecasting). A 1.16-million-iteration stationary run gave ~290 cycles,
split into train/test halves.

**Result.** The learned recurrence cycle P = 10400 iterations matches the book's raw
fig.-8.4 period (~10960); its forcing-corrected value (~3900, subtracting the
avalanching fraction) matches the book's tabulated ~4002 at alpha=0.15. The phase-locked
forecaster beats chance but only modestly: at the book's tight +/-100-iteration window
it reaches precision 0.25 against a chance level of 0.15 (skill 1.7x, 14 of 56
predictions hitting a large event); at +/-300 iterations precision is 0.68 vs chance
0.44 (skill 1.5x). Amplitude is correct (within +/-25%) for about half the hits. But the
genuinely LARGEST events -- the top 10% by size, n=43 -- are caught only 4 times within
+/-300 iterations (recall 0.09): the forecaster essentially misses the big ones.

**Implication.** Quasi-periodicity does confer some predictability -- a forecaster that
knows nothing but the recurrence rhythm beats chance by ~1.5-1.7x -- but the skill is
weak and, crucially, concentrated on the ordinary recurrent events, not the rare large
ones. The largest avalanches are the irregular, drifting domain collapses (E3) that do
NOT fall on the simple cycle, so they are the hardest to forecast precisely when they
matter most. This reproduces, in a model whose dynamics are fully known and noise-free,
the central difficulty the book raises in "Predicting Real Earthquakes": large events
are rare and sit in the unpredictable tail, so even perfect knowledge of the mechanism
buys little skill on them (the L'Aquila cautionary tale). It also dovetails with E6 --
the regularity that supplies what skill there is rests on a fine-tuned synchronization
that any disorder destroys. Caveat: one alpha, one realization, a single deliberately
simple forecaster; a lattice-state-aware predictor (using max F, eq. of Exercise 3, to
time the next avalanche) would do better on timing but not on which event is large.
earthquake/ofc_predict.py

---

## Synthesis (E1-E7, 2026-06-16)

The earthquake chapter is the conservation thread's natural conclusion. Where the
sandpile (S5/S7) showed in a binary way that bulk conservation is *necessary* for true
SOC, OFC makes conservation a continuous knob and shows the full crossover: as alpha
falls from the conservative 0.25, the avalanche-size distribution steepens (E1: PDF
slope -1.19 -> -1.87 -> -3.52) and its cutoff stops scaling with the system and
saturates to a dissipation-set characteristic size (E2: cutoff exponent D ~ 3.2 ->
~2.0 -> ~0). True criticality is the conservative limit; dissipation buys a
characteristic earthquake size.

The second theme is that the *same* dissipation that kills criticality buys temporal
order. Because OFC is deterministic and nonconservative, it synchronizes into domains
that collapse near-periodically (E3: recurrence periods matching the book to within a
few percent once forcing-corrected), the avalanche fronts grow more compact toward the
conservative limit (E4: E ~ T^gamma, gamma 1.47 -> 1.89), and that quasi-periodicity is
fragile (E6: a +/-0.01 alpha jitter halves it). The Grand Challenge ties it together:
the recurrence rhythm gives a forecaster modest skill (E7: 1.5-1.7x over chance) but
fails on the largest, irregular events -- the honest earthquake-prediction lesson, in a
model whose physics is fully known. The skip-ahead engine (E5) made the long stationary
runs feasible.

Cross-chapter: OFC sits to the sandpile as a tunable, deterministic cousin -- same
stop-and-go SOC scaffolding, but stability on the nodal value (not the slope) plus a
conservation parameter, which together convert robust-but-featureless scale invariance
into a conservation-tuned crossover with rich, fragile temporal structure. Status:
E1-E7 complete.
