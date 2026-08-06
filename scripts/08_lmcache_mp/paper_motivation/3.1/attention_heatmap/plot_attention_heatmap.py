#!/usr/bin/env python3
"""Plot four CacheBlend Figure-4-style panels summed over all layers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm


GAMMA = 0.35


def load_layer(root: Path, mode: str, layer: int) -> dict[str, np.ndarray]:
    path = root / mode / f"{mode}_layer_{layer:02d}.npz"
    with np.load(path) as loaded:
        return {name: loaded[name] for name in loaded.files}


def sum_layers(root: Path, mode: str, name: str) -> np.ndarray:
    matrices = [load_layer(root, mode, layer)[name] for layer in range(40)]
    shapes = {matrix.shape for matrix in matrices}
    if len(shapes) != 1:
        raise RuntimeError(f"inconsistent {mode} {name} shapes: {shapes}")
    return np.sum(np.stack(matrices), axis=0, dtype=np.float64)


def sum_direct_cross(root: Path, metadata: dict[str, Any]) -> np.ndarray:
    skill_start = int(metadata["skill_start"])
    skill_end = int(metadata["skill_end"])
    total = np.zeros((skill_end - skill_start, 48), dtype=np.float64)
    expected_positions: np.ndarray | None = None
    layer_rows: list[np.ndarray] = []
    for layer in range(40):
        values = load_layer(root, "direct", layer)
        positions = values["cross_positions"].astype(np.int64)
        if expected_positions is None:
            expected_positions = positions
        elif not np.array_equal(positions, expected_positions):
            raise RuntimeError("direct cross query positions differ across layers")
        layer_rows.append(values["cross"])
    assert expected_positions is not None
    if len(expected_positions):
        row_indices = expected_positions - skill_start
        if np.any(row_indices < 0) or np.any(row_indices >= len(total)):
            raise RuntimeError("direct cross query position lies outside the Skill")
        total[row_indices] = np.sum(np.stack(layer_rows), axis=0, dtype=np.float64)
    return total


def draw(
    ax: Any,
    matrix: np.ndarray,
    norm: PowerNorm,
    *,
    colorbar: bool = False,
) -> None:
    image = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        norm=norm,
    )
    ax.tick_params(axis="both", labelsize=14)
    if colorbar:
        bar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        bar.ax.tick_params(labelsize=14)
        vmax = float(norm.vmax)
        ticks = [value for value in (0, 0.5, 1, 2, 5, 10, 20) if value < vmax]
        ticks.append(vmax)
        bar.set_ticks(ticks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    recompute_meta = json.loads((args.output_dir / "recompute" / "prompt_metadata.json").read_text())
    direct_meta = json.loads((args.output_dir / "direct" / "prompt_metadata.json").read_text())
    if recompute_meta["prompt_token_ids"] != direct_meta["prompt_token_ids"]:
        raise RuntimeError("recompute and direct prompts are not token-identical")

    recompute_cross = sum_layers(args.output_dir, "recompute", "cross")
    direct_cross = sum_direct_cross(args.output_dir, direct_meta)
    recompute_forward = sum_layers(args.output_dir, "recompute", "forward")
    direct_forward = sum_layers(args.output_dir, "direct", "forward")
    direct_cross_max = float(np.max(direct_cross))
    cross_vmax = max(float(np.max(recompute_cross)), direct_cross_max)
    forward_vmax = float(
        max(np.nanmax(recompute_forward), np.nanmax(direct_forward))
    )
    cross_norm = PowerNorm(gamma=GAMMA, vmin=0, vmax=cross_vmax)
    forward_norm = PowerNorm(gamma=GAMMA, vmin=0, vmax=forward_vmax)

    fig, axes = plt.subplots(2, 2, figsize=(15, 6), constrained_layout=True)
    draw(axes[0, 0], recompute_cross, cross_norm)
    draw(
        axes[0, 1],
        recompute_forward,
        forward_norm,
        colorbar=True,
    )
    draw(
        axes[1, 0],
        direct_cross,
        cross_norm,
    )
    draw(
        axes[1, 1],
        direct_forward,
        forward_norm,
        colorbar=True,
    )
    pdf_path = args.output_dir / "cacheblend_figure4_layer_sum.png"
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"[plotted] {pdf_path} layers=sum(0..39)")


if __name__ == "__main__":
    main()
