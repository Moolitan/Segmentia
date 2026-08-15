#!/usr/bin/env python3
"""Plot self-only shallow-to-deep residual direction by layer and KV head."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze_layer_axis import NUM_KV_HEADS


FIELD = "self_shallow_to_deep_cosine"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def macro_matrix(
    rows: list[dict[str, str]], component: str
) -> tuple[list[int], np.ndarray]:
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row in rows:
        if row["component"] == component:
            grouped[(int(row["target_layer"]), int(row["head"]))].append(
                float(row[FIELD])
            )
    layers = sorted({key[0] for key in grouped})
    if not layers:
        raise ValueError(f"no direction rows for component {component}")
    matrix = np.empty((len(layers), NUM_KV_HEADS), dtype=np.float64)
    for layer_index, layer in enumerate(layers):
        for head in range(NUM_KV_HEADS):
            values = grouped[(layer, head)]
            if not values:
                raise ValueError(
                    f"missing direction values layer={layer} head={head}"
                )
            matrix[layer_index, head] = np.mean(values)
    return layers, matrix


def plot(rows: list[dict[str, str]], output_dir: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    matrices = {
        component: macro_matrix(rows, component) for component in ("K", "V")
    }
    color_limit = max(
        0.05,
        max(float(np.abs(matrix).max()) for _, matrix in matrices.values()),
    )
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.35), constrained_layout=True)
    image = None
    for axis, component in zip(axes, ("K", "V"), strict=True):
        layers, matrix = matrices[component]
        image = axis.imshow(
            matrix,
            origin="upper",
            aspect="auto",
            cmap="RdBu_r",
            vmin=-color_limit,
            vmax=color_limit,
            interpolation="nearest",
        )
        axis.set_xticks(range(NUM_KV_HEADS), range(1, NUM_KV_HEADS + 1))
        axis.set_xlabel("KV head")
        tick_layers = [layer for layer in (4, 9, 19, 29, 39) if layer in layers]
        axis.set_yticks(
            [layers.index(layer) for layer in tick_layers],
            [layer + 1 for layer in tick_layers],
        )
        axis.set_ylabel(f"{component} target layer")
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.0)
    if image is None:
        raise ValueError("no direction data to plot")
    colorbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.92, pad=0.02)
    colorbar.set_label("Residual-direction cosine")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / "shallow_deep_residual_direction.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    fig.savefig(
        output_dir / "shallow_deep_residual_direction.pdf",
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = read_rows(args.input_csv)
    plot(rows, args.output_dir)
    print(f"[self-shallow-deep-direction-plotted] output={args.output_dir}")


if __name__ == "__main__":
    main()
