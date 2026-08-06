#!/usr/bin/env python3
"""Build a compact all-layer paper figure from captured attention matrices."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def metadata(run_dir: Path, mode: str) -> dict:
    path = run_dir / mode / "prompt_metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


def layer_matrix(run_dir: Path, mode: str, layer: int, name: str) -> np.ndarray:
    path = run_dir / mode / f"{mode}_layer_{layer:02d}.npz"
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as loaded:
        return loaded[name].astype(np.float32, copy=False)


def max_query_atlas(run_dir: Path, mode: str, name: str) -> np.ndarray:
    rows = []
    for layer in range(40):
        matrix = layer_matrix(run_dir, mode, layer, name)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            raise RuntimeError(f"invalid {mode} {name} matrix at layer {layer}: {matrix.shape}")
        rows.append(matrix.max(axis=0))
    return np.stack(rows)


def draw_atlas(
    ax: mpl.axes.Axes,
    atlas: np.ndarray,
    vmax: float,
    panel: str,
    xlabel: str,
) -> mpl.image.AxesImage:
    image = ax.imshow(
        atlas,
        aspect="auto",
        origin="upper",
        interpolation="nearest",
        cmap="viridis",
        vmin=0,
        vmax=max(vmax, np.finfo(np.float32).eps),
        rasterized=True,
    )
    ax.text(0.0, 1.025, panel, transform=ax.transAxes, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_yticks([0, 9, 19, 29, 39], ["0", "9", "19", "29", "39"])
    for spine in ax.spines.values():
        spine.set_linewidth(0.35)
    return image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.run_dir / "paper_figure"

    recompute_meta = metadata(args.run_dir, "recompute")
    direct_meta = metadata(args.run_dir, "direct")
    if recompute_meta["prompt_token_ids"] != direct_meta["prompt_token_ids"]:
        raise RuntimeError("recompute and direct prompts are not token-identical")
    skill_start = int(recompute_meta["skill_start"])

    cross = max_query_atlas(args.run_dir, "recompute", "cross")
    forward_recompute = max_query_atlas(args.run_dir, "recompute", "forward")
    forward_direct = max_query_atlas(args.run_dir, "direct", "forward")
    if forward_recompute.shape != forward_direct.shape:
        raise RuntimeError("forward atlas shapes disagree")

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )
    fig = plt.figure(figsize=(7.1, 2.35))
    grid = fig.add_gridspec(
        1,
        5,
        width_ratios=[1.1, 0.07, 2.5, 2.5, 0.07],
        left=0.07,
        right=0.97,
        top=0.91,
        bottom=0.27,
        wspace=0.30,
    )
    axes = [fig.add_subplot(grid[0, index]) for index in (0, 2, 3)]
    cross_color_axis = fig.add_subplot(grid[0, 1])
    forward_color_axis = fig.add_subplot(grid[0, 4])
    cross_image = draw_atlas(
        axes[0],
        cross,
        float(cross.max()),
        "(a)",
        "Cross / recompute: prefix key",
    )
    forward_vmax = float(max(forward_recompute.max(), forward_direct.max()))
    recompute_image = draw_atlas(
        axes[1],
        forward_recompute,
        forward_vmax,
        "(b)",
        "Forward / recompute: key position",
    )
    draw_atlas(
        axes[2],
        forward_direct,
        forward_vmax,
        "(c)",
        "Forward / direct reuse: key position",
    )
    axes[0].set_ylabel("Transformer layer")
    axes[1].set_yticklabels([])
    axes[2].set_yticklabels([])
    for ax in axes[1:]:
        ax.plot(
            skill_start,
            -1.0,
            marker="v",
            color="#D55E00",
            markersize=3.5,
            clip_on=False,
        )
        ax.text(
            skill_start + 5,
            -0.9,
            "S",
            color="#D55E00",
            fontsize=6.5,
            ha="left",
            va="center",
            clip_on=False,
        )
        ax.set_ylim(39.5, -0.5)

    cross_bar = fig.colorbar(cross_image, cax=cross_color_axis)
    cross_bar.ax.set_title("max\nattn.", fontsize=6, pad=2)
    forward_bar = fig.colorbar(recompute_image, cax=forward_color_axis)
    forward_bar.ax.set_title("max\nattn.", fontsize=6, pad=2)
    fig.text(
        0.5,
        0.015,
        r"$S$: Skill start. Full reuse skips $Q_{skill}$, so cross-attention is not computed.",
        ha="center",
        fontsize=6.5,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "attention_layer_atlas.pdf"
    png_path = output_dir / "attention_layer_atlas.png"
    fig.savefig(pdf_path, dpi=600)
    fig.savefig(png_path, dpi=600)
    plt.close(fig)
    print(f"[paper figure] {pdf_path}")
    print(f"[paper figure] {png_path}")


if __name__ == "__main__":
    main()
