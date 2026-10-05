"""Nature-journal-style matplotlib defaults for pdb_align figures."""
from contextlib import contextmanager
from typing import Optional

# Okabe-Ito colorblind-safe categorical palette.
PALETTE = [
    "#0072B2", "#E69F00", "#009E73", "#D55E00",
    "#CC79A7", "#56B4E9", "#F0E442", "#000000",
]

# Nature column widths in millimetres.
_WIDTH_MM = {"single": 89.0, "double": 183.0}
_MM_PER_INCH = 25.4

_RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "axes.linewidth": 0.75,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.75,
    "ytick.major.width": 0.75,
    "axes.grid": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
}


@contextmanager
def apply_nature_style():
    """Context manager applying Nature-style rcParams, restoring on exit."""
    import matplotlib.pyplot as plt
    with plt.rc_context(_RCPARAMS):
        yield


def nature_figure(width: str = "single", height: Optional[float] = None):
    """Return (fig, ax) sized to a Nature column width at 300 dpi."""
    import matplotlib.pyplot as plt
    w_in = _WIDTH_MM.get(width, _WIDTH_MM["single"]) / _MM_PER_INCH
    h_in = height if height is not None else w_in * 0.62
    with apply_nature_style():
        fig, ax = plt.subplots(figsize=(w_in, h_in))
    return fig, ax


def panel_label(ax, letter: str):
    """Place a bold panel label at the axis top-left (outside the frame)."""
    ax.text(-0.12, 1.05, letter, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="right")
