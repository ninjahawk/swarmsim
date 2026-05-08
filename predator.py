# predator.py -- Predator-prey extension of the flocking model (Exercise 6)
#
# One predator hunts a flock of N prey agents. The predator is fast, strongly
# self-propelled toward the prey, and generates a long-range repulsive force
# on nearby prey. Prey interact with each other via the standard flocking model
# but feel extra repulsion from the predator.
#
# Research questions:
#   Q1: Does a flock evade the predator better than lone agents?
#   Q2: What predator speed is needed to catch the flock?
#   Q3: Do the predicted arched/splitting flock shapes emerge?
#   Q4: How does flock size N affect evasion?

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from flocking import params, buffer, force, order_parameter

os.makedirs('figures', exist_ok=True)

# =============================================================================
# PREDATOR PARAMETERS
# =============================================================================
PREY_DEFAULT = dict(
    N      = 100,    # number of prey
    n_iter = 3000,
    dt     = 0.01,
    r0     = 0.005,  # prey-prey repulsion radius
    eps    = 0.1,    # prey-prey repulsion amplitude
    rf     = 0.1,    # prey flocking radius
    alpha  = 1.0,    # prey flocking amplitude
    v0     = 0.02,   # prey target speed (slow walk)
    mu     = 10.0,   # prey self-propulsion amplitude
    ramp   = 0.1,    # prey random force
)

PRED_DEFAULT = dict(
    v0_pred  = 0.05,   # predator target speed
    mu_pred  = 10.0,   # predator self-propulsion amplitude
    alpha_pred = 5.0,  # predator flocking force toward prey
    rf_pred  = 0.15,   # predator perception radius (can see prey within this)
    r0_pred  = 0.1,    # predator long-range repulsion radius on prey
    eps_pred = 2.0,    # amplitude of predator repulsion felt by prey
    ramp_pred = 1.0,   # predator random force (erratic motion)
)


# =============================================================================
# SIMULATION
# =============================================================================
def run_predator(prey_overrides=None, pred_overrides=None,
                 n_frames=200, seed=None):
    """
    Run the predator-prey simulation.
    State layout: x[0:N]=prey x-pos, x[N:2N]=prey y-pos
    Predator state held separately: pred_x, pred_y, pred_vx, pred_vy.
    Returns list of (prey_px, prey_py, prey_vx, prey_vy, pred_x, pred_y) frames.
    """
    if seed is not None:
        np.random.seed(seed)

    pp = PREY_DEFAULT.copy()
    if prey_overrides:
        pp.update(prey_overrides)
    pd = PRED_DEFAULT.copy()
    if pred_overrides:
        pd.update(pred_overrides)

    p = params(pp)   # prey use the standard flocking params dict
    N = p['N']
    dt = p['dt']
    frame_every = max(1, p['n_iter'] // n_frames)

    # --- initialise prey ---
    x  = np.zeros(2*N)
    x[:N]  = np.random.uniform(0., 1., N)
    x[N:]  = np.random.uniform(0., 1., N)
    vx = np.random.uniform(-1., 1., N) * p['v0']
    vy = np.random.uniform(-1., 1., N) * p['v0']

    # --- initialise predator at a random edge position ---
    pred_x  = np.random.uniform(0., 1.)
    pred_y  = np.random.uniform(0., 1.)
    pred_vx = np.random.uniform(-1., 1.) * pd['v0_pred']
    pred_vy = np.random.uniform(-1., 1.) * pd['v0_pred']

    frames = []

    for i in range(p['n_iter']):
        # ----------------------------------------------------------------
        # 1. PREY FORCES (standard flocking + extra repulsion from predator)
        # ----------------------------------------------------------------
        rb = max(p['r0'], p['rf'])
        nb, xb, yb, vxb, vyb = buffer(rb, x, vx, vy, N)
        fx, fy = force(nb, xb, yb, vxb, vyb, x, vx, vy, p)

        # predator exerts long-range repulsion on nearby prey
        for j in range(N):
            # periodic distance to predator
            ddx = pred_x - x[j];  ddx -= round(ddx)
            ddy = pred_y - x[N+j]; ddy -= round(ddy)
            d = np.sqrt(ddx**2 + ddy**2)
            if 0 < d <= pd['r0_pred']:
                strength = pd['eps_pred'] * (1. - d/pd['r0_pred'])**1.5 / d
                fx[j] -= strength * ddx   # push prey away from predator
                fy[j] -= strength * ddy

        # ----------------------------------------------------------------
        # 2. PREDATOR FORCES (self-propulsion toward prey centre of mass)
        # ----------------------------------------------------------------
        # find prey centre of mass (periodic-aware)
        # use mean of angles trick for periodic wrapping
        cx = np.arctan2(np.sin(2*np.pi*x[:N]).mean(),
                        np.cos(2*np.pi*x[:N]).mean()) / (2*np.pi) % 1.
        cy = np.arctan2(np.sin(2*np.pi*x[N:]).mean(),
                        np.cos(2*np.pi*x[N:]).mean()) / (2*np.pi) % 1.

        # direction from predator to prey CoM (shortest path on torus)
        tx = cx - pred_x;  tx -= round(tx)
        ty = cy - pred_y;  ty -= round(ty)
        dist_to_prey = np.sqrt(tx**2 + ty**2)
        if dist_to_prey > 0:
            tx /= dist_to_prey;  ty /= dist_to_prey

        # predator self-propulsion: accelerate toward prey CoM
        pred_speed = np.sqrt(pred_vx**2 + pred_vy**2)
        if pred_speed > 0:
            # align toward prey CoM with flocking-like force
            pfx = pd['alpha_pred'] * tx
            pfy = pd['alpha_pred'] * ty
            # self-propulsion toward v0_pred
            pfx += pd['mu_pred'] * (pd['v0_pred'] - pred_speed) * (pred_vx/pred_speed)
            pfy += pd['mu_pred'] * (pd['v0_pred'] - pred_speed) * (pred_vy/pred_speed)
        else:
            pfx = pd['alpha_pred'] * tx
            pfy = pd['alpha_pred'] * ty

        # random component (erratic predator)
        pfx += pd['ramp_pred'] * np.random.uniform(-1., 1.)
        pfy += pd['ramp_pred'] * np.random.uniform(-1., 1.)

        # ----------------------------------------------------------------
        # 3. INTEGRATE
        # ----------------------------------------------------------------
        vx += fx * dt;  vy += fy * dt
        x[:N] = (x[:N] + vx*dt) % 1.
        x[N:] = (x[N:] + vy*dt) % 1.

        pred_vx += pfx * dt;  pred_vy += pfy * dt
        pred_x = (pred_x + pred_vx*dt) % 1.
        pred_y = (pred_y + pred_vy*dt) % 1.

        if i % frame_every == 0:
            frames.append((x[:N].copy(), x[N:].copy(),
                           vx.copy(), vy.copy(),
                           float(pred_x), float(pred_y),
                           float(pred_vx), float(pred_vy)))

    return frames


# =============================================================================
# CATCH DETECTION
# =============================================================================
def catch_radius():
    """Predator catches prey within this distance (hard-sphere contact: sum of agent radii)."""
    return PREY_DEFAULT['r0'] * 2   # = 0.01, covers ~pi*N*0.01^2 = 3% of domain

def count_caught(frames, p):
    """Return array of cumulative catch counts per frame."""
    r_catch = catch_radius()
    N = p['N']
    dt = p['dt']
    caught = set()
    counts = []
    for px, py, vx, vy, prd_x, prd_y, _, _ in frames:
        for j in range(N):
            ddx = prd_x - px[j];  ddx -= round(ddx)
            ddy = prd_y - py[j];  ddy -= round(ddy)
            if np.sqrt(ddx**2+ddy**2) < r_catch:
                caught.add(j)
        counts.append(len(caught))
    return np.array(counts)


# =============================================================================
# VISUALISATION
# =============================================================================
def animate_predator(frames, p=None, title='Predator-Prey Flocking'):
    if p is None:
        p = params(PREY_DEFAULT)
    dt = p['dt']
    n_iter = p['n_iter']
    frame_step = max(1, n_iter // len(frames))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_title(title)
    time_text = ax.text(0.02, 0.97, '', transform=ax.transAxes, fontsize=9, va='top')

    px0, py0, vx0, vy0, prdx0, prdy0, _, _ = frames[0]
    sp0 = np.sqrt(vx0**2+vy0**2); sp0[sp0==0]=1.
    prey_scat = ax.scatter(px0, py0, s=6, color='steelblue', zorder=3, label='Prey')
    prey_quiv = ax.quiver(px0, py0, vx0/sp0, vy0/sp0,
                          scale=40, width=0.002, color='steelblue', alpha=0.5, zorder=3)
    pred_scat = ax.scatter([prdx0], [prdy0], s=120, color='crimson',
                           marker='*', zorder=5, label='Predator')
    ax.legend(loc='upper right', fontsize=8)

    def update(fi):
        px, py, vx, vy, prdx, prdy, pvx, pvy = frames[fi]
        prey_scat.set_offsets(np.column_stack([px, py]))
        sp = np.sqrt(vx**2+vy**2); sp[sp==0]=1.
        prey_quiv.set_offsets(np.column_stack([px, py]))
        prey_quiv.set_UVC(vx/sp, vy/sp)
        pred_scat.set_offsets([[prdx, prdy]])
        time_text.set_text(f't = {fi * frame_step * dt:.2f}')
        return prey_scat, prey_quiv, pred_scat, time_text

    ani = animation.FuncAnimation(fig, update, frames=len(frames),
                                  interval=50, blit=True)
    plt.tight_layout()
    plt.show()
    return ani


def plot_predator_snapshot(frames, p, idx=-1, ax=None, title=None, save=None):
    fi = len(frames)-1 if idx == -1 else idx
    px, py, vx, vy, prdx, prdy, _, _ = frames[fi]
    sp = np.sqrt(vx**2+vy**2); sp[sp==0]=1.
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(px, py, s=4, color='steelblue', zorder=3)
    ax.quiver(px, py, vx/sp, vy/sp, scale=60, width=0.003, color='steelblue', alpha=0.5)
    ax.scatter([prdx], [prdy], s=200, color='crimson', marker='*', zorder=5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    dt = p['dt']; n_iter = p['n_iter']
    fs = max(1, n_iter // len(frames))
    ax.set_title(title or f't = {fi * fs * dt:.1f}', fontsize=9)
    if own:
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=120)
        plt.show()


# =============================================================================
# RESEARCH EXPERIMENTS
# =============================================================================
if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

    SEED = 7
    print('Running predator-prey experiments...')

    # ---- Experiment 1: baseline run, snapshot sequence ----
    print('\nExp 1: Baseline predator-prey run (N=100 prey)')
    p_prey = PREY_DEFAULT.copy()
    p_prey['n_iter'] = 5000
    frames = run_predator(prey_overrides=p_prey, seed=SEED, n_frames=250)
    p = params(p_prey)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.suptitle('Predator-prey: baseline run (N=100 prey)', fontsize=11)
    times = [0., 5., 10., 15., 20., 49.]
    for ax, t in zip(axes.flat, times):
        fs = max(1, p['n_iter'] // len(frames))
        fi = min(int(t / (fs * p['dt'])), len(frames)-1)
        plot_predator_snapshot(frames, p, idx=fi, ax=ax)
    plt.tight_layout()
    plt.savefig('figures/predator_1_snapshots.png', dpi=120)
    plt.close()
    print('  --> figures/predator_1_snapshots.png')

    # ---- Experiment 2: flock coherence under predator pressure ----
    # The predator model is a disturbance agent -- it repels prey and shapes the flock.
    # Key question: does flocking (alpha>0) help prey maintain collective distance from
    # the predator vs non-flocking prey who scatter individually?
    # Metric: mean predator-to-nearest-prey distance and flock order parameter Phi over time.
    print('\nExp 2: Flock coherence and predator distance  (10 seeds each)')
    N_seeds = 10
    n_it = 5000; n_fr = 200

    def mean_pred_dist(frames):
        """Mean distance from predator to nearest 10% of prey, per frame (periodic)."""
        dists = []
        N_prey = len(frames[0][0])
        k = max(1, N_prey // 10)
        for px, py, _, _, prdx, prdy, _, _ in frames:
            ddx = prdx - px; ddx -= np.round(ddx)
            ddy = prdy - py; ddy -= np.round(ddy)
            d = np.sqrt(ddx**2 + ddy**2)
            dists.append(np.sort(d)[:k].mean())
        return np.array(dists)

    flock_dist_all = []; flock_phi_all = []
    loner_dist_all = []; loner_phi_all = []

    for s in range(N_seeds):
        # flocking prey
        pf = PREY_DEFAULT.copy(); pf['n_iter'] = n_it; pf['alpha'] = 1.0
        ff = run_predator(prey_overrides=pf, seed=s, n_frames=n_fr)
        flock_dist_all.append(mean_pred_dist(ff))
        flock_phi_all.append([order_parameter(vx, vy) for _,_,vx,vy,_,_,_,_ in ff])

        # non-flocking prey
        pl = PREY_DEFAULT.copy(); pl['n_iter'] = n_it; pl['alpha'] = 0.0
        fl = run_predator(prey_overrides=pl, seed=s, n_frames=n_fr)
        loner_dist_all.append(mean_pred_dist(fl))
        loner_phi_all.append([order_parameter(vx, vy) for _,_,vx,vy,_,_,_,_ in fl])

    fd = np.array(flock_dist_all); ld = np.array(loner_dist_all)
    fp = np.array(flock_phi_all);  lp = np.array(loner_phi_all)
    fs_base = max(1, n_it // n_fr)
    t2 = np.arange(n_fr) * fs_base * PREY_DEFAULT['dt']

    print(f'  Steady-state mean predator-nearest distance:')
    print(f'    Flocking: {fd[:,-20:].mean():.3f}  Non-flocking: {ld[:,-20:].mean():.3f}')
    print(f'  Steady-state Phi:')
    print(f'    Flocking: {fp[:,-20:].mean():.3f}  Non-flocking: {lp[:,-20:].mean():.3f}')

    fig, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Exp 2: Flocking vs non-flocking prey under predator pressure\n(N=100, 10 seeds)', fontsize=10)
    ax2a.plot(t2, fd.mean(0), color='steelblue', lw=2, label='Flocking (alpha=1)')
    ax2a.fill_between(t2, fd.mean(0)-fd.std(0), fd.mean(0)+fd.std(0), color='steelblue', alpha=0.2)
    ax2a.plot(t2, ld.mean(0), color='firebrick', lw=2, label='Non-flocking (alpha=0)')
    ax2a.fill_between(t2, ld.mean(0)-ld.std(0), ld.mean(0)+ld.std(0), color='firebrick', alpha=0.2)
    ax2a.set_xlabel('Time'); ax2a.set_ylabel('Mean dist: predator to nearest 10% of prey')
    ax2a.set_title('Predator proximity (higher = safer)'); ax2a.legend()

    ax2b.plot(t2, fp.mean(0), color='steelblue', lw=2, label='Flocking')
    ax2b.fill_between(t2, fp.mean(0)-fp.std(0), fp.mean(0)+fp.std(0), color='steelblue', alpha=0.2)
    ax2b.plot(t2, lp.mean(0), color='firebrick', lw=2, label='Non-flocking')
    ax2b.fill_between(t2, lp.mean(0)-lp.std(0), lp.mean(0)+lp.std(0), color='firebrick', alpha=0.2)
    ax2b.set_xlabel('Time'); ax2b.set_ylabel('Order parameter Phi')
    ax2b.set_title('Flock coherence under predator pressure'); ax2b.legend(); ax2b.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig('figures/predator_2_coherence.png', dpi=120)
    plt.close()
    print('  --> figures/predator_2_coherence.png')

    # ---- Experiment 3: predator aggression sweep ----
    # Vary alpha_pred (how hard the predator chases the flock).
    # Effective predator speed = v0_pred + alpha_pred/mu_pred.
    # Metric: steady-state Phi of prey (does stronger predator break flock coherence?)
    # and mean predator distance to flock.
    print('\nExp 3: Flock response vs predator aggression alpha_pred  (8 seeds)')
    N_seeds3 = 8
    alpha_pred_vals = [0., 0.5, 1., 2., 3., 5., 8., 15.]
    phi_pred3 = []; phi_std3 = []
    dist_pred3 = []; dist_std3 = []
    eff_speeds3 = []
    n_it3 = 3000; n_fr3 = 150

    for ap in alpha_pred_vals:
        v_eff = PRED_DEFAULT['v0_pred'] + ap / PRED_DEFAULT['mu_pred']
        eff_speeds3.append(v_eff)
        phis_here = []; dists_here = []
        for s in range(N_seeds3):
            pe = PREY_DEFAULT.copy(); pe['n_iter'] = n_it3
            fe = run_predator(prey_overrides=pe,
                              pred_overrides={'alpha_pred': ap},
                              seed=s, n_frames=n_fr3)
            tail = fe[int(0.8*n_fr3):]
            phis_here.append(np.mean([order_parameter(vx,vy) for _,_,vx,vy,_,_,_,_ in tail]))
            dists_here.append(mean_pred_dist(tail).mean())
        phi_pred3.append(np.mean(phis_here)); phi_std3.append(np.std(phis_here))
        dist_pred3.append(np.mean(dists_here)); dist_std3.append(np.std(dists_here))
        print(f'  alpha_pred={ap:5.1f}  v_eff={v_eff:.3f}  Phi={np.mean(phis_here):.3f}+/-{np.std(phis_here):.3f}  dist={np.mean(dists_here):.3f}')

    fig, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Exp 3: Flock response to predator aggression\n(N=100, 8 seeds)', fontsize=11)
    ax3a.errorbar(alpha_pred_vals, phi_pred3, yerr=phi_std3,
                  fmt='o-', color='steelblue', capsize=4)
    ax3a.set_xlabel('Predator aggression alpha_pred')
    ax3a.set_ylabel('Steady-state flock Phi')
    ax3a.set_title('Flock coherence vs predator aggression'); ax3a.set_ylim(0, 1)

    ax3b.errorbar(eff_speeds3, dist_pred3, yerr=dist_std3,
                  fmt='s-', color='crimson', capsize=4)
    ax3b.set_xlabel('Predator effective speed (v0+alpha/mu)')
    ax3b.set_ylabel('Mean predator-to-nearest-prey dist')
    ax3b.set_title('Flock evasion vs predator effective speed')
    plt.tight_layout()
    plt.savefig('figures/predator_3_aggression_sweep.png', dpi=120)
    plt.close()
    print('  --> figures/predator_3_aggression_sweep.png')

    # ---- Experiment 4: flock size sweep ----
    # Does a larger flock evade better?
    print('\nExp 4: Evasion vs flock size N')
    N_vals = [10, 25, 50, 100, 200]
    mean_frac = []; std_frac = []   # fraction of flock caught
    n_it4 = 3000

    for N_val in N_vals:
        fracs = []
        for s in range(N_seeds):
            pn = PREY_DEFAULT.copy(); pn['n_iter'] = n_it4; pn['N'] = N_val
            fn = run_predator(prey_overrides=pn, seed=s, n_frames=100)
            p_pn = params(pn)
            c = count_caught(fn, p_pn)
            fracs.append(c[-1] / N_val)
        mean_frac.append(np.mean(fracs))
        std_frac.append(np.std(fracs))
        print(f'  N={N_val:4d}  fraction caught={np.mean(fracs):.3f} +/- {np.std(fracs):.3f}')

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(N_vals, mean_frac, yerr=std_frac,
                fmt='s-', color='steelblue', capsize=4)
    ax.set_xlabel('Flock size N')
    ax.set_ylabel('Fraction of flock caught by t=30')
    ax.set_title('Exp 4: Does flock size affect evasion?\n(5 seeds, shaded = std)')
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig('figures/predator_4_size_sweep.png', dpi=120)
    plt.close()
    print('  --> figures/predator_4_size_sweep.png')

    print('\nAll predator experiments complete. Figures in figures/')
