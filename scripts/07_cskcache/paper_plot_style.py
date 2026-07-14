"""Shared publication plotting style for CSKCache system figures."""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt


FONT_SIZE = 10
COLORS = {
    "Scheduler lookup": "#4C78A8",
    "Gap prefill": "#F2CF5B",
    "Probe roundtrip": "#E45756",
    "Anchor prefill": "#B279A2",
    "Load dispatch": "#72B7B2",
    "Tail KV load": "#54A24B",
    "Disk deserialize": "#79706E",
    "Storage lookup": "#9D755D",
    "Key H2D": "#4C78A8",
    "Value H2D": "#72B7B2",
    "RoPE": "#F58518",
    "Probe gather": "#ECA82C",
    "Residual": "#B279A2",
    "Scatter": "#54A24B",
    "Other/control": "#BAB0AC",
}
HATCHES = {
    "Scheduler lookup": "///",
    "Gap prefill": "...",
    "Probe roundtrip": "\\\\",
    "Anchor prefill": "xx",
    "Load dispatch": "++",
    "Tail KV load": "--",
    "Disk deserialize": "///",
    "Storage lookup": "...",
    "Key H2D": "\\\\",
    "Value H2D": "--",
    "RoPE": "xx",
    "Probe gather": "++",
    "Residual": "oo",
    "Scatter": "||",
    "Other/control": "",
}


def apply_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "axes.labelsize": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE - 1,
            "ytick.labelsize": FONT_SIZE - 1,
            "legend.fontsize": FONT_SIZE - 2,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "text.usetex": False,
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "hatch.linewidth": 0.45,
        }
    )


def save_figure(fig: plt.Figure, output_stem: str) -> None:
    fig.savefig(f"{output_stem}.pdf")
    fig.savefig(f"{output_stem}.png")
