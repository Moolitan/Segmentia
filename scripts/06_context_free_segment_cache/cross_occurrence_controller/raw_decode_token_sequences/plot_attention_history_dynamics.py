"""Plot how later decoded tokens attend back to history, across query offsets.

Reads per-offset region-mass dumps (attention_region_mass_*.jsonl) produced by
run_attention_divergence_probe.py + split_dumps_by_query_offset.py, and answers:
as decoding advances (query = the +offset-th thinking token), how much attention
do later tokens put on the injected skill span vs the pre-skill context, and how
is that split across network depth?

Outputs (under --output-dir):
  figures/history_mass_vs_offset.png     lines: region mass vs offset, by depth band
  figures/history_skill_share.png        skill share vs offset, recompute vs rope
  figures/history_depth_offset_heatmap.png  depth × offset mass heatmaps (2 region × 2 mode)
  attention_history_dynamics.csv         tidy per (offset, region, mode, layer) mass

Usage:
  python plot_attention_history_dynamics.py \\
    --base-dir .../attention_divergence/temp0.6/without_occ12 \\
    --offsets 0,30,60,90,120 \\
    --output-dir .../results/.../temp0.6/history_dynamics
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

REGIONS = ["skill_span", "pre_skill_context"]
MODES = ["recompute", "rope"]
NUM_LAYERS = 40
DEPTH_BANDS = {
    "L0-7 (shallow)": range(0, 8),
    "L8-19 (mid)": range(8, 20),
    "L20-31 (mid-deep)": range(20, 32),
    "L32-39 (deep)": range(32, 40),
}
BAND_COLORS = ["#2a9d8f", "#457b9d", "#e07a5f", "#8d3b72"]


def offset_dump_dir(base: Path, offset: int) -> Path:
    # offset 0 lives in the original flat dumps/ dir; others in offset_<NNN>/dumps.
    if offset == 0:
        flat = base / "dumps"
        if flat.exists():
            return flat
    return base / f"offset_{offset:03d}" / "dumps"


def load_offset(dump_dir: Path) -> dict[tuple[str, str, int, str], float]:
    table: dict[tuple[str, str, int, str], float] = {}
    for f in glob.glob(str(dump_dir / "attention_region_mass_*.jsonl")):
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                table[(str(r["case_id"]), str(r["region"]), int(r["layer_index"]), str(r["mode"]))] = float(r["mass_mean"])
    return table


def band_mean(table, cases, region, mode, layers) -> float:
    vals = [table.get((c, region, l, mode)) for c in cases for l in layers]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot attention-to-history dynamics vs decode offset.")
    parser.add_argument("--base-dir", type=Path, required=True,
                        help="Dir containing dumps/ (offset 0) and offset_<NNN>/dumps.")
    parser.add_argument("--offsets", default="0,30,60,90,120")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    offsets = [int(o) for o in str(args.offsets).split(",") if o.strip() != ""]
    tables: dict[int, dict] = {}
    for off in offsets:
        d = offset_dump_dir(args.base_dir, off)
        if not d.exists():
            print(f"[warn] missing dump dir for offset {off}: {d}")
            continue
        tables[off] = load_offset(d)
    offsets = [o for o in offsets if o in tables]
    if not offsets:
        raise FileNotFoundError("No offset dump dirs found under " + str(args.base_dir))

    cases = sorted({k[0] for t in tables.values() for k in t})
    figures_dir = args.output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    all_layers = list(range(NUM_LAYERS))

    # ── CSV: tidy per (offset, region, mode, layer) ──
    csv_path = args.output_dir / "attention_history_dynamics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["offset", "region", "mode", "layer", "mass_mean_over_cases"])
        for off in offsets:
            for region in REGIONS:
                for mode in MODES:
                    for l in all_layers:
                        v = band_mean(tables[off], cases, region, mode, [l])
                        w.writerow([off, region, mode, l, f"{v:.6f}"])

    # ── Figure 1: region mass vs offset, by depth band (per region, per mode) ──
    fig, axes = plt.subplots(len(REGIONS), len(MODES), figsize=(12, 8), squeeze=False,
                             sharex=True, constrained_layout=True)
    for ri, region in enumerate(REGIONS):
        for mi, mode in enumerate(MODES):
            ax = axes[ri][mi]
            for (band, layers), color in zip(DEPTH_BANDS.items(), BAND_COLORS):
                y = [band_mean(tables[off], cases, region, mode, list(layers)) for off in offsets]
                ax.plot(offsets, y, marker="o", color=color, label=band)
            ax.set_title(f"{region}  ·  {mode}")
            ax.set_xlabel("query offset (thinking token index)")
            ax.set_ylabel("attention mass")
            ax.set_xticks(offsets)
            ax.grid(True, alpha=0.3)
            if ri == 0 and mi == 0:
                ax.legend(fontsize=8, title="depth band")
    fig.suptitle("Attention to history vs decode offset, by network depth", fontsize=13)
    fig.savefig(figures_dir / "history_mass_vs_offset.png", dpi=150)
    fig.savefig(figures_dir / "history_mass_vs_offset.pdf")
    plt.close(fig)

    # ── Figure 2: skill share vs offset, recompute vs rope ──
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    mode_colors = {"recompute": "#264653", "rope": "#e07a5f"}
    for mode in MODES:
        share = []
        for off in offsets:
            sk = band_mean(tables[off], cases, "skill_span", mode, all_layers)
            pre = band_mean(tables[off], cases, "pre_skill_context", mode, all_layers)
            share.append(sk / (sk + pre) if (sk + pre) > 0 else np.nan)
        ax.plot(offsets, share, marker="o", color=mode_colors[mode], label=mode, linewidth=2)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.6)
    ax.set_ylim(0.4, 0.8)
    ax.set_xlabel("query offset (thinking token index)")
    ax.set_ylabel("skill share = skill / (skill + pre-skill)")
    ax.set_title("Relative reliance on injected skill span vs decode offset")
    ax.set_xticks(offsets)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(figures_dir / "history_skill_share.png", dpi=150)
    fig.savefig(figures_dir / "history_skill_share.pdf")
    plt.close(fig)

    # ── Figure 3: depth × offset heatmaps (region × mode) ──
    fig, axes = plt.subplots(len(REGIONS), len(MODES), figsize=(12, 9), squeeze=False,
                             constrained_layout=True)
    # shared color scale per region so recompute/rope are comparable
    for ri, region in enumerate(REGIONS):
        mats = {}
        for mode in MODES:
            mat = np.array([[band_mean(tables[off], cases, region, mode, [l]) for off in offsets]
                            for l in all_layers])
            mats[mode] = mat
        vmax = max(np.nanmax(m) for m in mats.values())
        for mi, mode in enumerate(MODES):
            ax = axes[ri][mi]
            im = ax.imshow(mats[mode], aspect="auto", origin="lower", cmap="viridis",
                           vmin=0, vmax=vmax, interpolation="nearest")
            ax.set_title(f"{region}  ·  {mode}")
            ax.set_xlabel("query offset")
            ax.set_ylabel("layer")
            ax.set_xticks(range(len(offsets)))
            ax.set_xticklabels(offsets)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Attention mass on history region: depth × decode offset", fontsize=13)
    fig.savefig(figures_dir / "history_depth_offset_heatmap.png", dpi=150)
    fig.savefig(figures_dir / "history_depth_offset_heatmap.pdf")
    plt.close(fig)

    print(f"cases: {len(cases)}, offsets: {offsets}")
    print(f"figures -> {figures_dir}")
    print(f"csv     -> {csv_path}")


if __name__ == "__main__":
    main()
