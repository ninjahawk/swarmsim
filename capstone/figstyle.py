"""Shared figure style for the capstone site (CAPSTONE_PLAN.md sections 4-5).

Usage: import figstyle; figstyle.apply()  BEFORE any pyplot figure exists.
The re-render runner (_shim_run.py) does this and then executes an untouched
research script, so every site figure comes off the same style sheet without
editing any research code. Three mechanisms:

  1. rcParams: white background, Inter (falls back to DejaVu Sans), base 11pt,
     no top/right spines, faint grid, frameless-look legends, design-token
     palette as the default color cycle.
  2. Named-color remap: research scripts hardcode 'tab:blue', 'C3', 'red', ...
     Those names are re-pointed at the site palette via matplotlib's named-color
     registry, so explicit colors land on the same tokens as the cycle.
  3. savefig interception: forces 200 dpi (retina at the page's 680-900 px
     figure width), white face, tight bbox, and REDIRECTS any write aimed at
     <repo>/figures/ into <repo>/docs/figures/ -- the repo figures/ archive is
     never overwritten by a site re-render.

ASCII-only (Windows cp1252 safe).
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
from matplotlib import font_manager
from matplotlib.figure import Figure
from cycler import cycler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
SITE_FIGDIR = os.path.join(ROOT, "docs", "figures")

DPI = 200          # forced on every savefig (originals ask 120-140)

# Optional font scale for figures whose native width lands the base 11pt below
# ~12 screen px at the page's display width. Set FIGSTYLE_FONT_SCALE in the
# caller's environment; unset (1.0) reproduces the session-B style exactly.
FONT_SCALE = float(os.environ.get("FIGSTYLE_FONT_SCALE", "1.0"))
# Titles can be scaled separately: long multi-panel titles collide before the
# small text does. Defaults to FONT_SCALE when unset.
TITLE_SCALE = float(os.environ.get("FIGSTYLE_TITLE_SCALE", "0") or 0) or FONT_SCALE

# ---- design tokens (plan section 5) ----
PAPER = "#ffffff"   # figures ship on white; the page paper is #fcfcfa
INK = "#1a1a1e"
MUTED = "#5f6368"
ACCENT = "#2b4c7e"  # flock-chapter hue doubles as the site accent
SAND = "#a07840"    # sandpile-chapter hue
RUST = "#8c3b34"    # earthquake-chapter hue
MOSS = "#5e7d5a"    # reserve categorical (5th series and 'green' requests)
SLATE = "#7d93b8"   # light companion to the accent (6th series / 'purple')
GRID = "#dfdfda"

CYCLE = [ACCENT, SAND, RUST, MUTED, MOSS, SLATE]

# What the scripts' hardcoded color names should mean under the site palette.
_REMAP = {
    "tab:blue": ACCENT, "b": ACCENT, "blue": ACCENT,
    "tab:orange": SAND, "orange": SAND, "tab:brown": SAND,
    "tab:red": RUST, "r": RUST, "red": RUST,
    "tab:green": MOSS, "g": MOSS, "green": MOSS, "tab:olive": MOSS,
    "tab:purple": SLATE, "purple": SLATE, "m": SLATE, "magenta": SLATE,
    "tab:pink": SLATE, "tab:cyan": SLATE,
    "tab:gray": MUTED, "tab:grey": MUTED, "gray": MUTED, "grey": MUTED,
    # CSS names hardcoded by the collective/predator scripts
    "teal": ACCENT, "steelblue": ACCENT,
    "crimson": RUST, "darkred": RUST,
    "seagreen": MOSS, "forestgreen": MOSS,
    "darkorange": SAND,
    "k": INK, "black": INK,
    # 'cyan' is only used for launch-site stars drawn over dark colormaps
    "cyan": "#ffffff", "c": "#ffffff",
}

_applied = False


def _register_fonts():
    """Register the bundled Inter statics if present; report the family used."""
    have = False
    if os.path.isdir(FONTDIR):
        for f in sorted(os.listdir(FONTDIR)):
            if f.lower().endswith(".ttf"):
                try:
                    font_manager.fontManager.addfont(os.path.join(FONTDIR, f))
                    have = True
                except Exception:
                    pass
    return "Inter" if have else "DejaVu Sans"


def apply():
    """Install the capstone figure style process-wide (idempotent)."""
    global _applied
    if _applied:
        return
    _applied = True

    family = _register_fonts()

    mcolors.get_named_colors_mapping().update(_REMAP)

    matplotlib.rcParams.update({
        # canvas
        "figure.facecolor": PAPER,
        "axes.facecolor": PAPER,
        "savefig.facecolor": PAPER,
        # type
        "font.family": "sans-serif",
        "font.sans-serif": [family, "DejaVu Sans"],
        "font.size": 11.0 * FONT_SCALE,
        "axes.titlesize": 12.0 * TITLE_SCALE,
        "axes.titleweight": "bold",
        "axes.labelsize": 11.0 * FONT_SCALE,
        "xtick.labelsize": 9.5 * FONT_SCALE,
        "ytick.labelsize": 9.5 * FONT_SCALE,
        "legend.fontsize": 9.5 * FONT_SCALE,
        "figure.titlesize": 13.0 * TITLE_SCALE,
        "mathtext.fontset": "dejavusans",
        # geometry of the frame
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.edgecolor": INK,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        # grid: faint, under the data
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.alpha": 1.0,
        "axes.axisbelow": True,
        # series
        "axes.prop_cycle": cycler(color=CYCLE),
        "lines.linewidth": 1.6,
        "lines.markersize": 5.5,
        # legends: quiet, near-frameless
        "legend.frameon": True,
        "legend.framealpha": 0.85,
        "legend.edgecolor": "none",
        "legend.fancybox": False,
        # save
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    })

    _install_savefig_hook()


def _install_savefig_hook():
    orig = Figure.savefig
    if getattr(Figure.savefig, "_figstyle_hook", False):
        return

    def savefig(self, fname, *args, **kwargs):
        kwargs["dpi"] = DPI
        kwargs.setdefault("bbox_inches", "tight")
        kwargs.setdefault("facecolor", PAPER)
        out = fname
        if isinstance(fname, (str, os.PathLike)):
            p = os.path.abspath(str(fname))
            site = os.path.abspath(SITE_FIGDIR)
            if (os.path.basename(os.path.dirname(p)) == "figures"
                    and not p.startswith(site)):
                os.makedirs(site, exist_ok=True)
                out = os.path.join(site, os.path.basename(p))
        r = orig(self, out, *args, **kwargs)
        if out is not fname:
            print("[figstyle] redirected save -> %s"
                  % os.path.relpath(out, ROOT))
        return r

    savefig._figstyle_hook = True
    Figure.savefig = savefig


def panel_label(ax, letter, dx=-0.02, dy=1.02):
    """Uniform panel label ('a', 'b', ...) top-left, outside the axes."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=13 * FONT_SCALE,
            fontweight="bold", va="bottom", ha="right", color=INK)


if __name__ == "__main__":
    # smoke test: style applies, palette remap live, save redirects
    apply()
    import numpy as np
    import matplotlib.pyplot as plt
    x = np.linspace(0, 4, 60)
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for k, c in enumerate(["tab:blue", "tab:red", "C1", "gray"]):
        ax.plot(x, np.sin(x + 0.6 * k) + 0.2 * k, color=c,
                label="series %d (%s)" % (k, c))
    ax.set_xlabel("x label")
    ax.set_ylabel("y label")
    ax.set_title("figstyle smoke test")
    ax.legend()
    panel_label(ax, "a")
    fig.savefig("figures/_figstyle_smoketest.png", dpi=120)
    print("family ->", matplotlib.rcParams["font.sans-serif"][0])
    print("OK: check docs/figures/_figstyle_smoketest.png")
