from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np


def apply_journal_style(*, use_tex: bool = False) -> None:
    """Apply a consistent, publication-quality Matplotlib style.

    This is intentionally lightweight (no seaborn dependency) and safe to call
    multiple times.
    """

    import matplotlib as mpl

    mpl.rcParams.update(
        {
            # Typography
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "STIXGeneral",
                "DejaVu Serif",
            ],
            "mathtext.fontset": "stix",
            "text.usetex": bool(use_tex),
            # Figure geometry
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            # Axes & ticks
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            # Legend
            "legend.fontsize": 9,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            # Lines
            "lines.linewidth": 2.0,
            "lines.markersize": 5.0,
            # Grid
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": ":",
            "grid.linewidth": 0.6,
            # Vector output quality (font embedding)
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig, path: Union[str, Path], *, transparent: bool = False) -> Path:
    """Save figure with consistent settings and return the resolved path."""

    from matplotlib.figure import Figure

    if not isinstance(fig, Figure):
        raise TypeError("save_figure expects a matplotlib.figure.Figure")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Explicit format avoids backend guessing.
    suffix = out_path.suffix.lower()
    if suffix in {".pdf", ".png", ".svg"}:
        fmt = suffix.lstrip(".")
    else:
        fmt = "pdf"
        out_path = out_path.with_suffix(".pdf")

    fig.savefig(out_path, format=fmt, transparent=transparent)
    return out_path


def plot_with_shadow(
    ax,
    x: Sequence,
    y: Sequence,
    *,
    color: str,
    label: Optional[str] = None,
    alpha_main: float = 0.8,
    alpha_shadow: float = 0.2,
    linewidth_main: float = 2.5,
    linewidth_shadow: float = 5.0,
    shadow_offset: Tuple[float, float] = (0.0, 0.0),
    marker: Optional[str] = None,
    linestyle: str = "-",
):
    """Plot a line with a subtle shadow for visibility."""

    x_arr = np.asarray(x)
    y_arr = np.asarray(y)

    # Shadow
    ax.plot(
        x_arr + shadow_offset[0],
        y_arr + shadow_offset[1],
        color=color,
        alpha=alpha_shadow,
        linewidth=linewidth_shadow,
        linestyle=linestyle,
        marker=marker,
        label=None,
    )

    # Main
    (main_line,) = ax.plot(
        x_arr,
        y_arr,
        color=color,
        alpha=alpha_main,
        linewidth=linewidth_main,
        linestyle=linestyle,
        marker=marker,
        label=label,
    )

    return main_line
