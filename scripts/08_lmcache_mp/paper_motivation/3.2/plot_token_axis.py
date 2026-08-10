#!/usr/bin/env python3
"""Plot token-axis KV fidelity and same-layer K residual commonality."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


COLORS = ("#0072B2", "#E69F00", "#009E73", "#D55E00")
LINESTYLES = ("--", "-.", (0, (2, 1)), (0, (5, 1)))
MARKERS = ("o", "s", "^", "D")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def layer_values(
    rows: list[dict[str, str]], skill: str, component: str, metric: str
) -> tuple[list[int], list[float], list[float]]:
    selected = sorted(
        (
            row
            for row in rows
            if row["skill"] == skill and row["component"] == component
        ),
        key=lambda row: int(row["layer"]),
    )
    layers = [int(row["layer"]) + 1 for row in selected]
    if metric == "cosine":
        direct = [float(row["direct_to_recompute_cosine_mean"]) for row in selected]
        corrected = [
            float(row["corrected_to_recompute_cosine_mean"]) for row in selected
        ]
    elif metric == "l2":
        direct = [
            math.sqrt(
                float(row["direct_to_recompute_sse"])
                / float(row["recompute_sq_norm"])
            )
            for row in selected
        ]
        corrected = [
            math.sqrt(
                float(row["corrected_to_recompute_sse"])
                / float(row["recompute_sq_norm"])
            )
            for row in selected
        ]
    else:
        raise ValueError(metric)
    if len(layers) != 40:
        raise ValueError(f"expected 40 layers for {skill}/{component}")
    return layers, direct, corrected


def fixed_alpha(rows: list[dict[str, str]]) -> float:
    values = {float(row["alpha"]) for row in rows}
    if len(values) != 1:
        raise ValueError(f"expected exactly one fixed alpha, found {sorted(values)}")
    alpha = values.pop()
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError(f"invalid alpha {alpha}")
    return alpha


def metric_limits(
    rows: list[dict[str, str]], component: str, metric: str
) -> tuple[float, float]:
    values: list[float] = []
    for skill in dict.fromkeys(row["skill"] for row in rows):
        _, direct, corrected = layer_values(rows, skill, component, metric)
        values.extend(direct)
        values.extend(corrected)
    value_min, value_max = min(values), max(values)
    padding = max((value_max - value_min) * 0.07, 0.001)
    lower = max(0.0, value_min - padding)
    upper = min(1.002, value_max + padding) if metric == "cosine" else value_max + padding
    return lower, upper


def plot_fidelity(
    rows: list[dict[str, str]], output_dir: Path, metric: str, variant: str
) -> None:
    fixed_alpha(rows)
    if variant not in {"direct", "corrected"}:
        raise ValueError(f"unknown fidelity variant {variant}")
    skills = list(dict.fromkeys(row["skill"] for row in rows))
    if len(skills) > len(COLORS):
        raise ValueError("add colors before plotting more Skill cases")
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.0), constrained_layout=True)
    for axis, component in zip(axes, ("K", "V"), strict=True):
        for index, skill in enumerate(skills):
            layers, direct, corrected = layer_values(rows, skill, component, metric)
            values = direct if variant == "direct" else corrected
            axis.plot(
                layers,
                values,
                color=COLORS[index],
                linestyle=LINESTYLES[index],
                linewidth=1.35,
                marker=MARKERS[index],
                markersize=2.8,
                markevery=4,
                markeredgewidth=0.5,
            )
        lower, upper = metric_limits(rows, component, metric)
        axis.set_ylim(lower, upper)
        axis.set_xlim(1, 40)
        axis.set_xticks([1, 10, 20, 30, 40])
        axis.set_xlabel("Model layer")
        axis.set_ylabel(
            f"{component} cosine to Recompute"
            if metric == "cosine"
            else f"{component} normalized L2"
        )
        axis.grid(axis="y", color="#D0D0D0", linewidth=0.5, alpha=0.65)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.0)

    handles: list[Line2D] = [
        Line2D(
            [0],
            [0],
            color=COLORS[index],
            linestyle=LINESTYLES[index],
            linewidth=1.35,
            marker=MARKERS[index],
            markersize=3.2,
            label=skill,
        )
        for index, skill in enumerate(skills)
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=4,
        fontsize=9,
        columnspacing=0.8,
        handlelength=2.0,
        handletextpad=0.5,
    )
    metric_stem = (
        "layerwise_cosine"
        if metric == "cosine"
        else "layerwise_normalized_l2"
    )
    stem = f"{variant}_recompute_{metric_stem}"
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def plot_commonality(rows: list[dict[str, str]], output_dir: Path) -> None:
    k_rows = [row for row in rows if row["component"] == "K"]
    skills = list(dict.fromkeys(row["skill"] for row in k_rows))
    fig, axes = plt.subplots(
        1, len(skills), figsize=(7.6, 2.15), sharex=True, sharey=True,
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes)
    image = None
    for axis, skill in zip(axes_array, skills, strict=True):
        matrix = np.full((40, 8), np.nan, dtype=np.float64)
        for row in k_rows:
            if row["skill"] != skill:
                continue
            matrix[int(row["layer"]), int(row["head"])] = float(
                row["prefix_suffix_direction_cosine"]
            )
        if np.isnan(matrix).all():
            raise ValueError(f"all commonality cells are undefined for {skill}")
        color_map = plt.get_cmap("YlGn").copy()
        color_map.set_bad("#D9D9D9")
        image = axis.imshow(
            matrix,
            origin="upper",
            aspect="auto",
            cmap=color_map,
            vmin=-1.0,
            vmax=1.0,
            interpolation="nearest",
        )
        axis.set_title(skill, fontsize=8)
        axis.set_xlabel("KV head")
        axis.set_xticks([0, 3, 7], [1, 4, 8])
        axis.set_yticks([0, 9, 19, 29, 39], [1, 10, 20, 30, 40])
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(0.9)
    axes_array[0].set_ylabel("Model layer")
    if image is None:
        raise ValueError("no K commonality rows")
    colorbar = fig.colorbar(image, ax=axes_array.tolist(), fraction=0.025, pad=0.02)
    colorbar.set_label("Prefix-to-suffix K residual cosine")
    fig.savefig(
        output_dir / "token_residual_commonality.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    fig.savefig(
        output_dir / "token_residual_commonality.pdf",
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fidelity-csv", type=Path, required=True)
    parser.add_argument("--commonality-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    fidelity_rows = read_rows(args.fidelity_csv)
    commonality_rows = read_rows(args.commonality_csv)
    for metric in ("cosine", "l2"):
        for variant in ("direct", "corrected"):
            plot_fidelity(fidelity_rows, args.output_dir, metric, variant)
    plot_commonality(commonality_rows, args.output_dir)
    print(f"[token-axis-plots] output={args.output_dir}")


if __name__ == "__main__":
    main()
