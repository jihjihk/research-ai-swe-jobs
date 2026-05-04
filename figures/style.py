"""Project-wide matplotlib style for AAAI 2026 paper figures.

Read figures/style.md before writing any figure script.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# AAAI 2026 geometry: textwidth 7.0 in, columnsep 0.375 in -> 3.3125 in per column.
COL_WIDTH = 3.3125
TEXT_WIDTH = 7.0

FIGSIZE_SINGLE = (COL_WIDTH, COL_WIDTH * 0.72)
FIGSIZE_DOUBLE = (TEXT_WIDTH, TEXT_WIDTH * 0.40)

OUTPUT_DIR = Path(__file__).parent / "output"

_INK = "#1f1f1f"
_GRID = "#dddddd"
_PALETTE = [
    "#4477AA", "#EE6677", "#228833", "#CCBB44",
    "#66CCEE", "#AA3377", "#BBBBBB",
]


def setup() -> None:
    """Apply the project's standard plotting style. Call once at the top of each figure script."""
    try:
        import scienceplots  # noqa: F401
        plt.style.use(["science", "no-latex"])
    except ImportError:
        plt.style.use("default")

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times", "Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 10,

        "text.color": _INK,
        "axes.labelcolor": _INK,
        "axes.edgecolor": _INK,
        "xtick.color": _INK,
        "ytick.color": _INK,
        "axes.prop_cycle": mpl.cycler(color=_PALETTE),

        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,

        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": _GRID,
        "grid.linewidth": 0.5,

        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,

        "legend.frameon": False,
        "legend.handlelength": 1.6,

        "lines.linewidth": 1.2,
        "lines.markersize": 4,

        "figure.figsize": FIGSIZE_SINGLE,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # AAAI forbids Type 3 fonts. 42 = TrueType embedding, which is allowed.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig, name: str, *, also_png: bool = True) -> Path:
    """Save `fig` as figures/output/<name>.pdf (canonical) and optionally a .png preview."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_DIR / f"{name}.pdf"
    fig.savefig(pdf_path)
    if also_png:
        fig.savefig(OUTPUT_DIR / f"{name}.png")
    return pdf_path
