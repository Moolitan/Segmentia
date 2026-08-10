#!/usr/bin/env python3
"""Plot layer-wise normalized L2 to Recompute for Direct or corrected KV."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from analyze_context_free_residual import write_csv
from analyze_layer_axis import NUM_LAYERS
from plot_direct_full_cosine import COLORS, LINE_STYLES


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_rows(
    direct: list[dict[str, str]],
    fidelity: list[dict[str, str]] | None,
    cutoff: int | None,
) -> list[dict[str, Any]]:
    skills = list(dict.fromkeys(row["skill"] for row in direct))
    direct_lookup = {
        (row["component"], row["skill"], int(row["layer"])): row
        for row in direct
    }
    corrected_lookup = (
        {
            (row["component"], row["skill"], int(row["target_layer"])): row
            for row in fidelity or []
            if int(row["cutoff"]) == cutoff
        }
        if cutoff is not None
        else {}
    )
    rows: list[dict[str, Any]] = []
    for component in ("K", "V"):
        for skill in skills:
            for layer in range(NUM_LAYERS):
                key = (component, skill, layer)
                if cutoff is None:
                    if key not in direct_lookup:
                        raise ValueError(f"missing Direct row: {key}")
                    value = float(
                        direct_lookup[key]["direct_to_full_normalized_l2"]
                    )
                    source = "direct_measured_layer"
                elif layer < cutoff:
                    value = 0.0
                    source = "recomputed_shallow_layer"
                else:
                    if key not in corrected_lookup:
                        raise ValueError(f"missing corrected fidelity row: {key}")
                    row = corrected_lookup[key]
                    recompute_sq_norm = float(row["recompute_sq_norm"])
                    if recompute_sq_norm <= 0:
                        raise ValueError(
                            f"non-positive Recompute squared norm: {key}"
                        )
                    value = math.sqrt(
                        float(row["corrected_to_recompute_sse"])
                        / recompute_sq_norm
                    )
                    source = "self_shallow_offset_deep_layer"
                rows.append(
                    {
                        "skill": skill,
                        "component": component,
                        "layer": layer,
                        "shallow_layer_cutoff": "" if cutoff is None else cutoff,
                        "normalized_l2_to_recompute": value,
                        "value_source": source,
                    }
                )
    return rows


def component_limits(
    direct: list[dict[str, str]],
    fidelity: list[dict[str, str]],
    component: str,
) -> tuple[float, float]:
    values = [
        float(row["direct_to_full_normalized_l2"])
        for row in direct
        if row["component"] == component
    ]
    values.extend(
        math.sqrt(
            float(row["corrected_to_recompute_sse"])
            / float(row["recompute_sq_norm"])
        )
        for row in fidelity
        if row["component"] == component and int(row["cutoff"]) in (4, 8)
    )
    if not values:
        raise ValueError(f"no Direct values for component {component}")
    padding = max((max(values) - min(values)) * 0.08, 0.002)
    return max(0.0, min(values) - padding), max(values) + padding


def plot(
    rows: list[dict[str, Any]],
    direct: list[dict[str, str]],
    fidelity: list[dict[str, str]],
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

    # Do not use constrained_layout together with bbox_inches="tight"
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7, 2),
    )

    # Explicit layout control to avoid ylabel clipping
    fig.subplots_adjust(
        left=0.18,
        right=0.98,
        bottom=0.25,
        top=0.95,
        wspace=0.30,
    )

    for axis, component, ylabel in zip(
        axes,
        ("K", "V"),
        (
            "Key normalized L2 error",
            "Value normalized L2 error",
        ),
        strict=True,
    ):
        for index, skill in enumerate(skills):
            skill_rows = sorted(
                (
                    row
                    for row in rows
                    if row["component"] == component
                    and row["skill"] == skill
                ),
                key=lambda row: int(row["layer"]),
            )

            axis.plot(
                [int(row["layer"]) + 1 for row in skill_rows],
                [
                    float(row["normalized_l2_to_recompute"])
                    for row in skill_rows
                ],
                color=COLORS[index],
                linestyle=LINE_STYLES[index],
                linewidth=1.6,
                label=skill,
            )

        axis.set_ylim(
            *component_limits(
                direct,
                fidelity,
                component,
            )
        )

        axis.set_xlim(
            1,
            NUM_LAYERS,
        )

        axis.set_xticks(
            [1, 10, 20, 30, 40]
        )

        axis.set_xlabel(
            "Model layer"
        )

        axis.set_ylabel(
            ylabel,
            labelpad=8,
        )

        axis.grid(
            axis="y",
            color="#D0D0D0",
            linewidth=0.5,
            alpha=0.65,
        )

        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.0)

    axes[1].legend(
        frameon=False,
        loc="best",
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_dir / f"{output_stem}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.15,
    )

    fig.savefig(
        output_dir / f"{output_stem}.pdf",
        bbox_inches="tight",
        pad_inches=0.15,
    )

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-csv", type=Path, required=True)
    parser.add_argument("--fidelity-csv", type=Path, required=True)
    parser.add_argument("--cutoff", type=int)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-stem", required=True)
    args = parser.parse_args()
    if args.cutoff is not None and not 0 < args.cutoff < NUM_LAYERS:
        raise ValueError("cutoff must be inside [1, 39]")
    direct = read_rows(args.direct_csv)
    required = {"direct_to_full_normalized_l2"}
    missing = required - set(direct[0])
    if missing:
        raise ValueError(f"Direct CSV is missing fields: {sorted(missing)}")
    fidelity = read_rows(args.fidelity_csv)
    rows = build_rows(direct, fidelity, args.cutoff)
    write_csv(args.output_csv, rows)
    plot(rows, direct, fidelity, args.output_dir, args.output_stem)
    mode = "direct" if args.cutoff is None else f"corrected-{args.cutoff}layers"
    print(f"[layerwise-normalized-l2] mode={mode} rows={len(rows)}")


if __name__ == "__main__":
    main()
