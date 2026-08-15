#!/usr/bin/env python3
"""Plot self-only corrected-to-Recompute cosine by model layer."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from analyze_context_free_residual import write_csv
from analyze_layer_axis import NUM_LAYERS
from plot_direct_full_cosine import COLORS, LINE_STYLES


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def corrected_rows(
    fidelity: list[dict[str, str]], skills: list[str], cutoff: int
) -> list[dict[str, Any]]:
    lookup = {
        (row["component"], row["skill"], int(row["target_layer"])): row
        for row in fidelity
        if int(row["cutoff"]) == cutoff
    }
    rows: list[dict[str, Any]] = []
    for component in ("K", "V"):
        for skill in skills:
            for layer in range(NUM_LAYERS):
                if layer < cutoff:
                    cosine = 1.0
                    source = "recomputed_shallow_layer"
                else:
                    key = (component, skill, layer)
                    if key not in lookup:
                        raise ValueError(f"missing corrected fidelity row: {key}")
                    cosine = float(
                        lookup[key]["corrected_to_recompute_cosine_mean"]
                    )
                    source = "self_shallow_offset_deep_layer"
                rows.append(
                    {
                        "skill": skill,
                        "component": component,
                        "layer": layer,
                        "shallow_layer_cutoff": cutoff,
                        "corrected_to_recompute_cosine_mean": cosine,
                        "value_source": source,
                    }
                )
    return rows


def component_limits(
    direct_rows: list[dict[str, str]], component: str
) -> tuple[float, float]:
    values = [
        float(row["direct_to_full_cosine_mean"])
        for row in direct_rows
        if row["component"] == component
    ]
    if not values:
        raise ValueError(f"no Direct baseline values for component {component}")
    padding = max((max(values) - min(values)) * 0.08, 0.002)
    return min(values) - padding, min(1.002, max(values) + padding)


def plot(
    rows: list[dict[str, Any]],
    direct_rows: list[dict[str, str]],
    output_dir: Path,
    output_stem: str,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    skills = list(dict.fromkeys(str(row["skill"]) for row in rows))
    fig, axes = plt.subplots(1, 2, figsize=(7, 2), constrained_layout=True)
    for axis, component, ylabel in zip(
        axes,
        ("K", "V"),
        (
            "Key cosine to Recompute",
            "Value cosine to Recompute",
        ),
        strict=True,
    ):
        for index, skill in enumerate(skills):
            skill_rows = sorted(
                (
                    row
                    for row in rows
                    if row["component"] == component and row["skill"] == skill
                ),
                key=lambda row: int(row["layer"]),
            )
            axis.plot(
                [int(row["layer"]) + 1 for row in skill_rows],
                [float(row["corrected_to_recompute_cosine_mean"]) for row in skill_rows],
                color=COLORS[index],
                linestyle=LINE_STYLES[index],
                linewidth=1.6,
                label=skill,
            )
        axis.set_ylim(*component_limits(direct_rows, component))
        axis.set_xlim(1, NUM_LAYERS)
        axis.set_xticks([1, 10, 20, 30, 40])
        axis.set_xlabel("Model layer")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#D0D0D0", linewidth=0.5, alpha=0.65)
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.0)
    axes[1].legend(frameon=False, loc="best")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / f"{output_stem}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.1,
    )
    fig.savefig(
        output_dir / f"{output_stem}.pdf",
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fidelity-csv", type=Path, required=True)
    parser.add_argument("--direct-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cutoff", type=int, default=4)
    parser.add_argument(
        "--output-stem", default="corrected_recompute_layerwise_cosine"
    )
    args = parser.parse_args()
    if not 0 < args.cutoff < NUM_LAYERS:
        raise ValueError("cutoff must be inside [1, 39]")
    fidelity = read_rows(args.fidelity_csv)
    direct = read_rows(args.direct_csv)
    skills = list(dict.fromkeys(row["skill"] for row in direct))
    rows = corrected_rows(fidelity, skills, args.cutoff)
    write_csv(args.output_csv, rows)
    plot(rows, direct, args.output_dir, args.output_stem)
    print(
        f"[corrected-recompute-cosine] cutoff={args.cutoff} "
        f"skills={len(skills)} rows={len(rows)} output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
