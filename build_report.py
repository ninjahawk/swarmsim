# build_report.py -- Generate report_draft.pdf using reportlab
#
# Run: python build_report.py
# Output: report_draft.pdf

import os
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.units import cm

OUTPUT = 'report_draft.pdf'
FIGURES = 'figures'

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
base = getSampleStyleSheet()

title_style = ParagraphStyle('Title2', parent=base['Title'],
    fontSize=18, leading=22, spaceAfter=6, alignment=TA_CENTER)

subtitle_style = ParagraphStyle('Subtitle', parent=base['Normal'],
    fontSize=11, leading=14, spaceAfter=4, alignment=TA_CENTER, textColor=colors.HexColor('#444444'))

author_style = ParagraphStyle('Author', parent=base['Normal'],
    fontSize=11, leading=14, spaceAfter=2, alignment=TA_CENTER)

section_style = ParagraphStyle('Section', parent=base['Heading1'],
    fontSize=13, leading=16, spaceBefore=18, spaceAfter=6,
    textColor=colors.HexColor('#1a1a1a'), borderPad=0)

subsection_style = ParagraphStyle('Subsection', parent=base['Heading2'],
    fontSize=11, leading=14, spaceBefore=12, spaceAfter=4,
    textColor=colors.HexColor('#2a2a2a'))

body_style = ParagraphStyle('Body', parent=base['Normal'],
    fontSize=10, leading=15, spaceAfter=8, alignment=TA_JUSTIFY)

code_style = ParagraphStyle('Code', parent=base['Code'],
    fontSize=9, leading=12, spaceAfter=6, leftIndent=24,
    fontName='Courier', textColor=colors.HexColor('#333333'))

caption_style = ParagraphStyle('Caption', parent=base['Normal'],
    fontSize=9, leading=12, spaceAfter=10, alignment=TA_CENTER,
    textColor=colors.HexColor('#555555'), italics=True)

ref_style = ParagraphStyle('Ref', parent=base['Normal'],
    fontSize=9.5, leading=14, spaceAfter=4, leftIndent=18, firstLineIndent=-18)

bold_body = ParagraphStyle('BoldBody', parent=body_style, fontName='Helvetica-Bold')


def fig(filename, width=5.5*inch, caption_text=None):
    path = os.path.join(FIGURES, filename)
    if not os.path.exists(path):
        return [Paragraph(f'[Figure not found: {filename}]', caption_style)]
    items = [Image(path, width=width, height=width*0.6)]
    if caption_text:
        items.append(Paragraph(caption_text, caption_style))
    return items


def section(num, title):
    return Paragraph(f'{num}. {title}', section_style)


def subsection(num, title):
    return Paragraph(f'{num} {title}', subsection_style)


def p(text):
    return Paragraph(text, body_style)


def eq(text):
    return Paragraph(text, code_style)


def sp(n=6):
    return Spacer(1, n)


def hr():
    return HRFlowable(width='100%', thickness=0.5, color=colors.HexColor('#cccccc'), spaceAfter=4)


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------
doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=LETTER,
    leftMargin=1.1*inch, rightMargin=1.1*inch,
    topMargin=1.1*inch, bottomMargin=1.1*inch,
    title='Emergent Flocking and Collective Evasion in a Force-Based Agent Model',
    author='Nathan Langley'
)

story = []

# ---------------------------------------------------------------------------
# Title page block
# ---------------------------------------------------------------------------
story += [
    sp(30),
    Paragraph('Emergent Flocking and Collective Evasion<br/>in a Force-Based Agent Model',
              title_style),
    sp(10),
    hr(),
    sp(8),
    Paragraph('PHY 351 — Independent Summer Research', subtitle_style),
    Paragraph('Nathan Langley', author_style),
    Paragraph('May 2026', author_style),
    Paragraph('Advisor: Prof. Ian Beatty', author_style),
    sp(28),
    hr(),
]

# ---------------------------------------------------------------------------
# Abstract
# ---------------------------------------------------------------------------
story += [
    sp(14),
    Paragraph('<b>Abstract</b>', section_style),
    p('I present computational simulations of a force-based flocking model (Charbonneau, 2017) '
      'in which <i>N</i> agents on a periodic two-dimensional domain interact through repulsion, '
      'velocity alignment, self-propulsion, and random noise. After validating the implementation '
      'against three analytically tractable limiting cases, I characterize the parameter space '
      'through systematic sweeps and identify an exact analytical result: the equilibrium cruise '
      'speed of an aligned flock is v<sub>eq</sub> = v<sub>0</sub> + α/μ, not the '
      'nominal target speed v<sub>0</sub>. Finite-size scaling of the repulsion-only system shows '
      'no diverging susceptibility, indicating a smooth crossover rather than a true phase '
      'transition. I then extend the model with a predator agent and find that flocking prey '
      'maintain near-perfect velocity alignment (Φ ~ 1.0) under sustained predator pressure '
      'while non-flocking prey scatter completely (Φ ~ 0.1). With multiple predators, flock '
      'coherence remains intact while flock elongation increases substantially, suggesting shape '
      'adaptation as a collective evasion strategy. These results demonstrate that the primary '
      'function of flocking under predation is coherence maintenance, not distance maximization.'),
]

# ---------------------------------------------------------------------------
# 1. Introduction
# ---------------------------------------------------------------------------
story += [
    section(1, 'Introduction'),
    p('One of the central puzzles in complex systems is how large-scale ordered behavior '
      'emerges from purely local interactions. Flocking—the coordinated motion of birds, '
      'fish schools, and animal herds—is a canonical example. Each individual follows simple '
      'rules based on its immediate neighbors, yet the collective produces sweeping global '
      'patterns with no central coordination. Understanding the conditions under which order '
      'emerges, and how robust that order is to perturbation, has implications ranging from '
      'evolutionary biology to crowd control.'),
    p('The model studied here is based on Chapter 10 of Charbonneau (2017), which was '
      'originally developed by Silverberg et al. (2013) to describe crowd dynamics in mosh pits '
      'at heavy metal concerts. Each agent in the model is subject to four forces: short-range '
      'repulsion, velocity-aligning flocking force, self-propulsion toward a target speed, and '
      'random noise. The interplay of these four forces produces a rich behavioral phase space, '
      'including crystalline order, disordered fluid motion, and coherent streaming flocks.'),
    p('This report covers four main investigations. First, I validate the implementation and '
      'establish baseline behavior through limiting cases. Second, I sweep the noise and '
      'alignment parameters to characterize the transition to flocking. Third, I examine whether '
      'the repulsion-only transition constitutes a true phase transition using finite-size '
      'scaling. Finally, I extend the model with a predator agent and characterize collective '
      'evasion behavior, including the effect of multiple simultaneous predators on flock '
      'geometry and coherence.'),
]

# ---------------------------------------------------------------------------
# 2. Model
# ---------------------------------------------------------------------------
story += [
    section(2, 'Model'),
    subsection('2.1', 'Setup'),
    p('<i>N</i> agents move on a periodic unit square (<i>x</i>, <i>y</i> ∈ [0, 1]), '
      'implemented as a torus so that agents exiting one edge reappear on the opposite side. '
      'Agent positions and velocities are updated at each timestep using the forward Euler '
      'method at d<i>t</i> = 0.01. The state of agent <i>j</i> at time <i>t</i> is fully '
      'described by its position (<i>x<sub>j</sub></i>, <i>y<sub>j</sub></i>) and velocity '
      '(<i>v<sub>xj</sub></i>, <i>v<sub>yj</sub></i>).'),

    subsection('2.2', 'Forces'),
    p('The total force on agent <i>j</i> is a sum of four contributions '
      '(Charbonneau Eqs. 10.1–10.8):'),

    p('<b>Repulsion.</b> A short-range force prevents agents from overlapping. It acts on pairs '
      'within distance 2<i>r</i><sub>0</sub> and grows in intensity as agents approach:'),
    eq('F_rep,j  =  eps * SUM_k [ (1 - r_jk / 2r0)^(3/2) * r_hat_jk ]    for r_jk <= 2r0'),
    p('where <i>r<sub>jk</sub></i> is the distance between agents <i>j</i> and <i>k</i>, '
      'and <i>ŕ&#770;<sub>jk</sub></i> is a unit vector pointing from <i>k</i> toward <i>j</i>.'),

    p('<b>Flocking.</b> An alignment force drives the velocity of agent <i>j</i> toward the '
      'mean velocity of its neighbors within a flocking radius <i>r<sub>f</sub></i>:'),
    eq('F_flock,j  =  alpha * V_bar / |V_bar|,    V_bar = SUM_{k: r_jk <= rf} v_k'),
    p('The normalized form ensures the flocking force has constant magnitude α regardless '
      'of how many neighbors are present.'),

    p('<b>Self-propulsion.</b> A speed-correcting force drives agent <i>j</i> toward a target '
      'speed <i>v</i><sub>0</sub> along its current direction of motion:'),
    eq('F_prop,j  =  mu * (v0 - |v_j|) * v_hat_j'),
    p('where <i>v&#770;<sub>j</sub></i> is the unit vector along <b>v</b><sub>j</sub>. This '
      'force accelerates agents moving too slowly and brakes those moving too fast.'),

    p('<b>Random noise.</b> Each component of the random force is drawn independently from a '
      'uniform distribution on [−<i>ramp</i>, <i>ramp</i>] at each timestep.'),

    p('With unit mass for all agents, Newton\'s second law gives '
      '<b>a</b><sub>j</sub> = <b>F</b><sub>j</sub>, and the equations of motion are integrated as:'),
    eq('x_j(t + dt)  =  x_j(t) + v_j(t) * dt\n'
       'v_j(t + dt)  =  v_j(t) + F_j(t) * dt'),

    subsection('2.3', 'Periodic Boundary Implementation'),
    p('Force calculations near domain boundaries require special handling. Agents within range '
      '<i>r<sub>f</sub></i> of any boundary are replicated as ghost copies on the opposite side, '
      'so that the flocking and repulsion forces computed for a real agent account correctly for '
      'neighbors across the periodic boundary. This buffer zone approach follows Charbonneau '
      'Fig. 10.2.'),

    subsection('2.4', 'Metrics'),
    p('The primary measure of collective order is the <b>order parameter</b>:'),
    eq('Phi  =  | (1/N) SUM_j v_hat_j |'),
    p('Φ = 1 corresponds to perfect velocity alignment; Φ = 0 to randomly oriented '
      'motion. I also track total kinetic energy KE = (1/2)Σ<i><sub>j</sub></i> '
      '|<b>v</b><i><sub>j</sub></i>|<sup>2</sup> and, for flock geometry, the '
      '<b>radius of gyration</b> R<sub>g</sub> (root-mean-square distance from center of mass) '
      'and the <b>aspect ratio</b> AR (ratio of the major to minor eigenvalue of the spatial '
      'covariance matrix, measuring elongation).'),

    subsection('2.5', 'Default Parameters'),
    p('Unless otherwise noted, simulations use the parameters from Charbonneau Table 10.1:'),
    eq('N = 350,  r0 = 0.005,  eps = 0.1,  rf = 0.1,  alpha = 1.0,\n'
       'v0 = 1.0,  mu = 10.0,  ramp = 0.5,  dt = 0.01'),
]

# ---------------------------------------------------------------------------
# 3. Validation
# ---------------------------------------------------------------------------
story += [
    section(3, 'Validation'),
    p('Before drawing any conclusions from the simulations, I verified the implementation '
      'against three limiting cases with known expected behavior.'),
    p('<b>Case 1: Pure random walk.</b> With all physical forces disabled (ε = α = '
      'μ = v<sub>0</sub> = 0, ramp = 1), agents should perform a pure random walk with no '
      'preferred direction. The measured order parameter was Φ = 0.04 (expected ∼0) '
      'and agent positions spread uniformly across the domain (standard deviation ∼0.29, '
      'consistent with a uniform distribution). This confirms the integration and boundary '
      'conditions are working.'),
    p('<b>Case 2: Repulsion and noise only.</b> With α = 0 and v<sub>0</sub> = 0 '
      '(self-propulsion acts as a brake), the model reproduces Fig. 10.5 from Charbonneau: '
      'at low noise (η = 1) agents pack into a close-packed quasi-hexagonal structure, '
      'while at high noise (η = 30) the arrangement disorders into a fluid. Φ remains '
      'near zero throughout (no alignment force), as expected.'),
    p('<b>Case 3: Flocking only.</b> With ε = 0 and v<sub>0</sub> = 0, the alignment '
      'force alone should drive agents into a coherent stream. The final order parameter was '
      'Φ = 0.998 after <i>t</i> = 30, confirming that a single coherent flock forms from '
      'random initial conditions, consistent with Fig. 10.6 from Charbonneau (Fig. 1).'),
]
story += fig('validate_3_flocking_only.png', width=5.0*inch,
             caption_text='Figure 1. Flocking-only limiting case (ε = 0, v₀ = 0). '
                          'Starting from random initial conditions, all 350 agents align into a '
                          'single coherent stream within t = 30 (Φ = 0.998).')

# ---------------------------------------------------------------------------
# 4. Results
# ---------------------------------------------------------------------------
story += [
    section(4, 'Results'),
    subsection('4.1', 'Equilibrium Cruise Speed'),
    p('An exact result follows directly from the force equations. In a perfectly aligned flock, '
      'all agents move in the same direction with the same speed. The flocking force then acts '
      'purely in the forward direction with magnitude α. The self-propulsion force '
      'balances this when:'),
    eq('alpha + mu * (v0 - v_eq)  =  0    =>    v_eq  =  v0 + alpha/mu'),
    p('With the default parameters (α = 1, μ = 10, v<sub>0</sub> = 1), this predicts '
      'v<sub>eq</sub> = 1.10. I verified this prediction by measuring steady-state mean speed '
      'across four values of α with v<sub>0</sub> = 1, μ = 10 fixed. Measured speeds '
      'agreed with the prediction to within 0.002 in all cases (Fig. 2). The implication is '
      'that v<sub>0</sub> and α are not independent knobs for cruise speed: to achieve a '
      'target cruising speed v<sub>c</sub>, one must set v<sub>0</sub> = v<sub>c</sub> − '
      'α/μ.'),
]
story += fig('phase4_sweeps.png', width=5.8*inch,
             caption_text='Figure 2. Parameter sweeps from analysis.py. Top-right panel: '
                          'steady-state mean speed vs. α (blue, measured) vs. '
                          'v₀ + α/μ (dashed, predicted). Agreement is within 0.002 '
                          'across all tested values.')

story += [
    subsection('4.2', 'Flock Formation'),
    p('Sweeping the flocking amplitude α with noise fixed at ramp = 0.1 (5 seeds per '
      'point, error bars represent standard deviation) shows a sharp onset of flocking near '
      'α ∼ 0.05. At α = 0, Φ = 0.09 ± 0.01. By α = 0.05, '
      'Φ = 0.40 ± 0.12, and by α = 0.20, Φ = 0.89 ± 0.03. The large '
      'run-to-run variance near the threshold (std ∼ 0.1–0.2 for '
      '0.05 ≤ α ≤ 0.15) indicates sensitivity to initial conditions near the '
      'onset. Above α ∼ 0.2, flocks form reliably (Fig. 3).'),
    p('With all forces active and the default α = 1, flock coherence is robust: Φ '
      'exceeds 0.99 up to noise amplitude ramp = 3, exceeds 0.97 at ramp = 5, and drops below '
      '0.5 only at ramp ∼ 20. The alignment force makes the system dramatically more '
      'resistant to noise disruption than the repulsion-only case.'),
]
story += fig('phase4_sweeps.png', width=5.8*inch,
             caption_text='Figure 3. Flock formation sweep. Left panel: Φ vs. α '
                          '(5 seeds, error bars = 1σ) showing sharp onset near α ∼ 0.05 '
                          'with large variance in the transition region. Right panel: Φ vs. '
                          'noise amplitude with full model active.')

story += [
    subsection('4.3', 'Nature of the Solid-to-Fluid Transition'),
    p('In the repulsion-only system (α = 0, v<sub>0</sub> = 0), kinetic energy rises with '
      'noise amplitude, suggesting a transition from a solid-like crystalline state to a '
      'fluid-like disordered state. To test whether this constitutes a true phase transition, '
      'I performed finite-size scaling across N = 25, 50, 100, and 200 (Fig. 4).'),
    p('A true phase transition would produce KE/<i>N</i> curves that depend on <i>N</i>, with '
      'a critical point (susceptibility peak) that converges to a finite η<sub>c</sub> as '
      '<i>N</i> increases. Instead, the KE/<i>N</i> curves are essentially identical for all '
      'four system sizes, and the susceptibility χ = N · var(KE/<i>N</i>) increases '
      'monotonically with η with no peak.'),
    p('This indicates a smooth crossover rather than a true critical phenomenon. The physical '
      'picture is consistent with the high compactness of the system '
      '(C = πNr<sub>0</sub>² ∼ 0.78 for the parameters used): each agent is '
      'effectively caged by its neighbors and oscillates harmonically around a fixed lattice '
      'site. This produces KE proportional to η², independent of <i>N</i>—'
      'behavior characteristic of uncoupled harmonic oscillators, not a correlated system '
      'approaching criticality.'),
]
story += fig('phase_transition_scaling.png', width=5.8*inch,
             caption_text='Figure 4. Finite-size scaling of the solid-to-fluid transition '
                          '(N = 25, 50, 100, 200). Left: KE/N vs. noise amplitude η—all '
                          'curves collapse onto a single line, indicating no N-dependence. '
                          'Right: susceptibility χ = N·var(KE/N) rises monotonically '
                          'with no peak, ruling out a true critical point.')

story += [
    subsection('4.4', 'Predator-Prey Dynamics'),
    p('I extended the model with a predator agent that chases the prey center of mass via a '
      'strong alignment force (α<sub>pred</sub> = 5) and generates a long-range repulsive '
      'force on nearby prey (r<sub>0,pred</sub> = 0.1). Prey parameters were set to the '
      'slow-walking regime (v<sub>0</sub> = 0.02, α = 1.0, ramp = 0.1) to match the '
      'concert crowd context from Silverberg et al.'),
    p('<b>Flock coherence under pressure.</b> Comparing flocking prey (α = 1) versus '
      'non-flocking prey (α = 0) across 10 random initializations shows a striking '
      'divergence. Flocking prey maintain Φ ∼ 0.998 throughout the simulation '
      'despite continuous predator pressure. Non-flocking prey scatter almost immediately, '
      'reaching Φ ∼ 0.096 in steady state (Fig. 5). Non-flocking agents maintain '
      'marginally more individual distance from the predator (0.127 vs. 0.112), but they lose '
      'all collective structure. The flock absorbs the disturbance while remaining coherent.'),
    p('<b>Evasion distance saturates.</b> Sweeping predator aggression α<sub>pred</sub> '
      '(which sets effective predator speed as v<sub>eq,pred</sub> = v<sub>0,pred</sub> + '
      'α<sub>pred</sub>/μ<sub>pred</sub>) reveals that the mean predator-to-nearest-prey '
      'distance drops from 0.24 with a passive predator to ∼0.10 for '
      'α<sub>pred</sub> ≥ 1 and then saturates—the collective repulsion response '
      'establishes a minimum buffer distance that persists regardless of predator aggression.'),
]
story += fig('predator_2_coherence.png', width=5.6*inch,
             caption_text='Figure 5. Flock coherence under predator pressure (10 seeds). '
                          'Flocking prey (blue) maintain Φ ∼ 0.998 throughout. '
                          'Non-flocking prey (red) scatter to Φ ∼ 0.096. '
                          'Non-flocking agents are marginally farther from the predator '
                          '(0.127 vs. 0.112) but lose all collective structure.')

story += [
    p('<b>Flock geometry.</b> Without a predator, the steady-state aspect ratio is AR = 2.61 '
      'and radius of gyration R<sub>g</sub> = 0.215. With a predator, these shift modestly to '
      'AR = 2.76 and R<sub>g</sub> = 0.274. Stronger flocking (larger α) produces '
      'substantially more elongated flocks: AR increases from 2.09 at α = 0.2 to 7.27 '
      'at α = 2.0. These highly elongated configurations resemble the arched, thinning '
      'flocks predicted by Charbonneau Exercise 6 (Fig. 6).'),
]
story += fig('geometry_2_alpha_sweep.png', width=5.8*inch,
             caption_text='Figure 6. Flock geometry vs. flocking amplitude α under predator '
                          'pressure (8 seeds). Aspect ratio rises sharply with α, reaching '
                          'AR = 7.27 at α = 2.0. Coherence Φ remains near 1.0 in all '
                          'cases.')

story += [
    p('<b>Multiple predators.</b> With 1 to 4 simultaneous predators, flock order parameter '
      'stays near 0.975–0.991—coherence is maintained across the entire range '
      '(Fig. 7). Aspect ratio rises substantially with predator count (AR = 2.83 with one '
      'predator, AR = 7.91 with three), while R<sub>g</sub> increases modestly. '
      'Counterintuitively, the minimum predator-to-prey distance increases slightly as the '
      'number of predators grows (0.093 for one predator, 0.106 for three). The flock under '
      'multiple predators is more elongated but no more accessible to any individual predator.'),
]
story += fig('multi_pred_3_summary.png', width=5.8*inch,
             caption_text='Figure 7. Steady-state flock metrics vs. number of predators (N = 100, '
                          '8 seeds). Coherence (Φ, left) stays near 1.0 regardless of '
                          'predator count. Aspect ratio (center) rises from 2.83 to 7.91. '
                          'Evasion distance (right) counterintuitively increases with more predators.')

# ---------------------------------------------------------------------------
# 5. Discussion
# ---------------------------------------------------------------------------
story += [
    section(5, 'Discussion'),
    p('The most striking result of the predator simulations is that flocking is not primarily '
      'a strategy for maximizing distance from a predator. Non-flocking agents individually '
      'maintain slightly greater separation from the predator, yet the flocking agents clearly '
      'have a more robust collective response: they remain coordinated, move in concert, and '
      'hold a consistent buffer distance. This distinction—coherence versus distance—'
      'may reflect something real about the function of biological flocking. A coherent flock '
      'can mount a coordinated escape response; scattered individuals cannot.'),
    p('The increasing aspect ratio under multiple predators is interesting. When pressure '
      'arrives from multiple directions, the flock elongates rather than fragmenting. This '
      'could represent a geometric optimum: a thin stream is a harder target to surround than '
      'a compact cluster, and threading between predators may minimize total exposure. That '
      'this behavior emerges spontaneously from the force equations, without any deliberate '
      'strategy encoded in the agents, is a good example of emergence.'),
    p('The equilibrium speed result (v<sub>eq</sub> = v<sub>0</sub> + α/μ) is an '
      'exact consequence of the force equations that Charbonneau does not explicitly note. It '
      'means a researcher using this model who sets v<sub>0</sub> = 1 and α = 1 expecting '
      'agents to cruise at speed 1 will find them consistently at speed 1.1. For simulations '
      'where absolute speed matters (e.g., comparing flocking and non-flocking agents under '
      'identical predator pressure), this correction is necessary.'),
    p('The phase transition result is perhaps the most physically interesting negative result. '
      'The expectation from analogies with equilibrium statistical mechanics—that a system '
      'with a solid-to-fluid transition should show a sharp critical point—does not hold '
      'here. The <i>N</i>-independence of KE/<i>N</i> and the absence of a susceptibility peak '
      'indicate that the agents in this high-compactness regime are not behaving cooperatively '
      'at the transition. Each agent vibrates independently. Whether a true phase transition '
      'could be found in a lower-compactness regime (where agents have more room to move) '
      'is an open question.'),
]

# ---------------------------------------------------------------------------
# 6. Conclusions
# ---------------------------------------------------------------------------
story += [
    section(6, 'Conclusions'),
    p('This study produced four main results:'),
    p('<b>1. Equilibrium speed:</b> The cruise speed of an aligned flock is '
      'v<sub>eq</sub> = v<sub>0</sub> + α/μ, exactly. This is a direct consequence '
      'of the force equations and must be accounted for when comparing simulations at different '
      'parameter values.'),
    p('<b>2. Phase transition:</b> The solid-to-fluid transition in the repulsion-only system '
      'is a smooth crossover, not a true phase transition. Finite-size scaling shows no '
      'N-dependent critical point, consistent with independent harmonic oscillator behavior '
      'in a high-compactness system.'),
    p('<b>3. Flock coherence under predation:</b> Flocking prey maintain near-perfect velocity '
      'alignment under predator pressure while non-flocking prey scatter. Evasion distance '
      'saturates at a minimum buffer value regardless of predator aggression.'),
    p('<b>4. Collective geometry:</b> Flocks become more elongated under both stronger internal '
      'alignment and greater predator pressure. Multiple predators elongate the flock without '
      'degrading coherence; evasion distance counterintuitively improves.'),
    p('Taken together, these results suggest that the primary function of the alignment force '
      'in this model—and possibly in biological flocking—is not to keep individuals '
      'far from threats, but to maintain coordinated collective response.'),
]

# ---------------------------------------------------------------------------
# References
# ---------------------------------------------------------------------------
story += [
    section('', 'References'),
    Paragraph('Charbonneau, P. (2017). <i>Natural Complexity: A Modeling Handbook</i>. '
              'Princeton University Press.', ref_style),
    sp(4),
    Paragraph('Silverberg, J. L., Bierbaum, M., Sethna, J. P., and Cohen, I. (2013). '
              'Collective motion of humans in mosh and circle pits at heavy metal concerts. '
              '<i>Physical Review Letters</i>, 110, 228701.', ref_style),
]

# ---------------------------------------------------------------------------
# Appendix: Code
# ---------------------------------------------------------------------------
story += [
    section('', 'Appendix: Code'),
    p('All simulation code is available at '
      '<a href="https://github.com/ninjahawk/Summer_Research" color="blue">'
      'https://github.com/ninjahawk/Summer_Research</a>.'),
    sp(6),
]

table_data = [
    ['File', 'Description'],
    ['flocking.py', 'Core model: buffer zone, vectorized force function, run loop, metrics'],
    ['analysis.py', 'Validation limiting cases and parameter sweeps'],
    ['predator.py', 'Single-predator extension with 4 experiments'],
    ['phase_transition.py', 'Finite-size scaling of solid-to-fluid transition'],
    ['geometry.py', 'Radius of gyration and aspect ratio analysis'],
    ['multi_predator.py', 'Multi-predator experiments'],
    ['build_report.py', 'Generates this PDF using reportlab'],
]

col_widths = [1.8*inch, 4.5*inch]
tbl = Table(table_data, colWidths=col_widths)
tbl.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e8eaf6')),
    ('FONTNAME',   (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE',   (0, 0), (-1, -1), 9),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f5f5')]),
    ('GRID',       (0, 0), (-1, -1), 0.4, colors.HexColor('#cccccc')),
    ('LEFTPADDING',  (0, 0), (-1, -1), 8),
    ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ('TOPPADDING',   (0, 0), (-1, -1), 5),
    ('BOTTOMPADDING',(0, 0), (-1, -1), 5),
    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
]))
story.append(tbl)

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
doc.build(story)
print(f'PDF written to {OUTPUT}')
