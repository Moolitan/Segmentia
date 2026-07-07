"""Plot attention heatmaps from the thinking-start attention probe.

Reads JSONL dumps produced by the vLLM attention probe and generates:

1. Per-case heatmap pair (recompute vs rope):
   - X-axis: token position within each key region
   - Y-axis: layer (0-39)
   - Color: mean attention value (averaged across query-head groups)
   - Two columns: recompute | rope, two rows: pre_skill_context | skill_span

2. Per-case difference heatmap:
   - Same axes, color = rope − recompute (divergence colormap)

3. Region mass comparison (line plot across layers):
   - X-axis: layer index
   - Y-axis: attention mass allocated to region
   - Two lines: recompute vs rope, for each region

4. Aggregate summary across all cases:
   - Mean ± std of region mass delta (rope − recompute) per layer

Usage:
  python plot_attention_divergence_probe.py \\
    --dump-dir .../attention_divergence/temp0.6/without_occ12/dumps \\
    --output-dir .../results/problem_exploration/attention_divergence/temp0.6
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = "/home/wsh/openhands_code_research/results/problem_exploration/raw_decode_token_sequences/attention_divergence/temp0.6"

REGION_NAMES = ["pre_skill_context", "skill_span"]
NUM_LAYERS = 40


def read_jsonl_glob(directory: str, stem_prefix: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(directory).glob(f"{stem_prefix}*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def case_key(row: dict[str, Any]) -> str:
    return str(row["case_id"])


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ── Region mass line plots (per case + aggregate) ──────────────────────

def build_mass_table(
    region_rows: list[dict[str, Any]],
) -> dict[tuple[str, str, str], dict[int, float]]:
    """
    (case_id, mode, region) → {layer_index: mass_mean}
    region mass (区域注意力质量) 指的是某个 query token 在一次 attention 里，把多少 softmax 概率分配给了某个 key 区域 -- 也就是该区域上注意力权重值和。
    probs = torch.softmax(attention_scores, dim=-1)   # 对某个 query token 的所有 key token 计算 softmax，全序列概率上和 = 1
    region_mass = probs[:, region_start:region_end].sum()  # 某个 query token 在某个 key 区域的注意力质量 = 该区域上所有 key token 的 softmax 概率和

    """
    table: dict[tuple[str, str, str], dict[int, float]] = defaultdict(dict)
    for row in region_rows:
        key = (case_key(row), str(row["mode"]), str(row["region"]))
        layer = int(row["layer_index"])
        table[key][layer] = float(row["mass_mean"])
    return table


def plot_region_mass_per_case(
    mass_table: dict[tuple[str, str, str], dict[int, float]],
    figures_dir: str,
) -> int:
    case_ids = sorted({k[0] for k in mass_table})
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    written = 0

    for case_id in case_ids:
        fig, axes = plt.subplots(1, len(REGION_NAMES), figsize=(6 * len(REGION_NAMES), 4),
                                 squeeze=False, constrained_layout=True)
        for col, region in enumerate(REGION_NAMES):
            ax = axes[0, col]
            for mode, color, ls in [("recompute", "#1f77b4", "-"), ("rope", "#d62728", "--")]:
                layer_mass = mass_table.get((case_id, mode, region), {})
                if not layer_mass:
                    continue
                layers = sorted(layer_mass)
                values = [layer_mass[l] for l in layers]
                ax.plot(layers, values, color=color, ls=ls, lw=1.5, label=mode, marker=".", ms=3)
            ax.set_xlabel("Layer")
            ax.set_ylabel("Attention mass")
            ax.set_title(region)
            ax.legend(fontsize=8)
            ax.set_xlim(-0.5, NUM_LAYERS - 0.5)

        fig.suptitle(f"Region mass: {case_id}", fontsize=11)
        path = Path(figures_dir) / f"region_mass_{safe_name(case_id)}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written += 1

    return written


def plot_region_mass_aggregate(
    mass_table: dict[tuple[str, str, str], dict[int, float]],
    figures_dir: str,
) -> None:
    case_ids = sorted({k[0] for k in mass_table})
    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(REGION_NAMES), figsize=(6 * len(REGION_NAMES), 4),
                             squeeze=False, constrained_layout=True)

    for col, region in enumerate(REGION_NAMES):
        ax = axes[0, col]
        for mode, color, ls in [("recompute", "#1f77b4", "-"), ("rope", "#d62728", "--")]:
            per_layer: dict[int, list[float]] = defaultdict(list)
            for cid in case_ids:
                lm = mass_table.get((cid, mode, region), {})
                for layer, val in lm.items():
                    per_layer[layer].append(val)
            if not per_layer:
                continue
            layers = sorted(per_layer)
            means = [np.mean(per_layer[l]) for l in layers]
            stds = [np.std(per_layer[l]) for l in layers]
            ax.plot(layers, means, color=color, ls=ls, lw=1.8, label=f"{mode} (n={len(case_ids)})")
            ax.fill_between(layers,
                            [m - s for m, s in zip(means, stds)],
                            [m + s for m, s in zip(means, stds)],
                            color=color, alpha=0.15)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Attention mass (mean ± std)")
        ax.set_title(region)
        ax.legend(fontsize=8)
        ax.set_xlim(-0.5, NUM_LAYERS - 0.5)

    fig.suptitle(f"Aggregate region mass across {len(case_ids)} cases", fontsize=11)
    fig.savefig(Path(figures_dir) / "region_mass_aggregate.png", dpi=150)
    fig.savefig(Path(figures_dir) / "region_mass_aggregate.pdf")
    plt.close(fig)


def plot_region_mass_delta_aggregate(
    mass_table: dict[tuple[str, str, str], dict[int, float]],
    figures_dir: str,
) -> None:
    case_ids = sorted({k[0] for k in mass_table})
    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(REGION_NAMES), figsize=(6 * len(REGION_NAMES), 4),
                             squeeze=False, constrained_layout=True)

    for col, region in enumerate(REGION_NAMES):
        ax = axes[0, col]
        per_layer_delta: dict[int, list[float]] = defaultdict(list)
        for cid in case_ids:
            rc = mass_table.get((cid, "recompute", region), {})
            rp = mass_table.get((cid, "rope", region), {})
            for layer in set(rc) & set(rp):
                per_layer_delta[layer].append(rp[layer] - rc[layer])
        if not per_layer_delta:
            ax.set_title(f"{region} (no data)")
            continue
        layers = sorted(per_layer_delta)
        means = [np.mean(per_layer_delta[l]) for l in layers]
        stds = [np.std(per_layer_delta[l]) for l in layers]
        ax.plot(layers, means, color="#2ca02c", lw=1.8, label="rope − recompute")
        ax.fill_between(layers,
                        [m - s for m, s in zip(means, stds)],
                        [m + s for m, s in zip(means, stds)],
                        color="#2ca02c", alpha=0.2)
        ax.axhline(0, color="gray", ls=":", lw=0.8)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Δ attention mass (rope − recompute)")
        ax.set_title(region)
        ax.legend(fontsize=8)
        ax.set_xlim(-0.5, NUM_LAYERS - 0.5)

    fig.suptitle(f"Region mass delta across {len(case_ids)} cases", fontsize=11)
    fig.savefig(Path(figures_dir) / "region_mass_delta_aggregate.png", dpi=150)
    fig.savefig(Path(figures_dir) / "region_mass_delta_aggregate.pdf")
    plt.close(fig)


# ── Local-window heatmaps (per case) ──────────────────────────────────

def build_heatmap_matrix(
    window_rows: list[dict[str, Any]],
    case_id: str,
    mode: str,
    window_name: str,
) -> tuple[np.ndarray | None, int, int]:
    """Build a (NUM_LAYERS × window_length) matrix of mean attention values."""
    relevant = [
        r for r in window_rows
        if case_key(r) == case_id
        and str(r["mode"]) == mode
        and str(r["window"]) == window_name
    ]
    if not relevant:
        return None, 0, 0
    window_length = int(relevant[0]["window_length"])
    window_start = int(relevant[0]["window_start"])
    mat = np.full((NUM_LAYERS, window_length), np.nan, dtype=np.float64)
    for row in relevant:
        layer = int(row["layer_index"])
        values = row["attention_mean"]
        if layer < NUM_LAYERS and len(values) == window_length:
            mat[layer, :] = values
    return mat, window_start, window_length


def plot_heatmap_pair(
    window_rows: list[dict[str, Any]],
    figures_dir: str,
) -> int:
    case_ids = sorted({case_key(r) for r in window_rows})
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    written = 0

    for case_id in case_ids:
        fig, axes = plt.subplots(
            len(REGION_NAMES), 3,
            figsize=(20, 4 * len(REGION_NAMES)),
            squeeze=False,
            constrained_layout=True,
        )

        for row_idx, region in enumerate(REGION_NAMES):
            mat_rc, ws_rc, wl_rc = build_heatmap_matrix(window_rows, case_id, "recompute", region)
            mat_rp, ws_rp, wl_rp = build_heatmap_matrix(window_rows, case_id, "rope", region)

            if mat_rc is None and mat_rp is None:
                for c in range(3):
                    axes[row_idx, c].set_title(f"{region} (no data)")
                    axes[row_idx, c].axis("off")
                continue

            vmax = max(
                np.nanmax(mat_rc) if mat_rc is not None else 0,
                np.nanmax(mat_rp) if mat_rp is not None else 0,
                1e-12,
            )
            ws = ws_rc if mat_rc is not None else ws_rp
            wl = wl_rc if mat_rc is not None else wl_rp

            xtick_stride = max(1, wl // 10)
            xtick_pos = list(range(0, wl, xtick_stride))
            xtick_labels = [str(ws + p) for p in xtick_pos]

            for col, (mat, title) in enumerate([(mat_rc, "recompute"), (mat_rp, "rope")]):
                ax = axes[row_idx, col]
                if mat is None:
                    ax.set_title(f"{title} (no data)")
                    ax.axis("off")
                    continue
                im = ax.imshow(mat, aspect="auto", vmin=0, vmax=vmax, cmap="viridis",
                               origin="lower", interpolation="nearest")
                ax.set_ylabel("Layer")
                ax.set_xlabel(f"Token position ({region})")
                ax.set_title(f"{title}")
                ax.set_xticks(xtick_pos)
                ax.set_xticklabels(xtick_labels, fontsize=6, rotation=45)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            # Difference heatmap
            ax_diff = axes[row_idx, 2]
            if mat_rc is not None and mat_rp is not None and mat_rc.shape == mat_rp.shape:
                diff = mat_rp - mat_rc
                abs_max = max(np.nanmax(np.abs(diff)), 1e-12)
                im_diff = ax_diff.imshow(diff, aspect="auto", vmin=-abs_max, vmax=abs_max,
                                         cmap="RdBu_r", origin="lower", interpolation="nearest")
                ax_diff.set_ylabel("Layer")
                ax_diff.set_xlabel(f"Token position ({region})")
                ax_diff.set_title("rope − recompute")
                ax_diff.set_xticks(xtick_pos)
                ax_diff.set_xticklabels(xtick_labels, fontsize=6, rotation=45)
                fig.colorbar(im_diff, ax=ax_diff, fraction=0.046, pad=0.04)
            else:
                ax_diff.set_title("diff (incompatible)")
                ax_diff.axis("off")

        fig.suptitle(f"Attention heatmap: {case_id}", fontsize=12)
        path = Path(figures_dir) / f"heatmap_{safe_name(case_id)}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written += 1

    return written


# ── Summary CSV ───────────────────────────────────────────────────────

def write_summary_csv(
    mass_table: dict[tuple[str, str, str], dict[int, float]],
    output_dir: str,
) -> None:
    case_ids = sorted({k[0] for k in mass_table})
    rows: list[dict[str, Any]] = []
    for cid in case_ids:
        row: dict[str, Any] = {"case_id": cid}
        for region in REGION_NAMES:
            rc = mass_table.get((cid, "recompute", region), {})
            rp = mass_table.get((cid, "rope", region), {})
            rc_mean = float(np.mean(list(rc.values()))) if rc else 0.0
            rp_mean = float(np.mean(list(rp.values()))) if rp else 0.0
            row[f"recompute_{region}_mean"] = rc_mean
            row[f"rope_{region}_mean"] = rp_mean
            row[f"delta_{region}"] = rp_mean - rc_mean
        rows.append(row)

    fieldnames = ["case_id"]
    for region in REGION_NAMES:
        fieldnames.extend([
            f"recompute_{region}_mean",
            f"rope_{region}_mean",
            f"delta_{region}",
        ])
    write_csv(Path(output_dir) / "attention_divergence_summary.csv", rows, fieldnames)


# ── Main ──────────────────────────────────────────────────────────────
DEFAULT_DUMP_DIR = "/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/06_context_free_segment_cache/cross_occurrence_function_vector/attention_divergence/temp0.6/without_occ12/dumps"

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot attention divergence probe outputs.")
    parser.add_argument("--dump-dir", default=DEFAULT_DUMP_DIR,
                        help="Directory holding attention_region_mass_*.jsonl / "
                             "attention_local_windows_*.jsonl for one query offset.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Directory to write figures/ and the summary CSV.")
    args = parser.parse_args()
    dump_dir = args.dump_dir
    output_dir = args.output_dir

    print(f"Reading dumps from {dump_dir}")
    region_rows = read_jsonl_glob(dump_dir, "attention_region_mass_")
    window_rows = read_jsonl_glob(dump_dir, "attention_local_windows_")
    print(f"  region_mass rows: {len(region_rows)}")
    print(f"  local_windows rows: {len(window_rows)}")

    if not region_rows:
        raise FileNotFoundError(f"No attention_region_mass_*.jsonl in {dump_dir}")

    mass_table = build_mass_table(region_rows)
    case_ids = sorted({k[0] for k in mass_table})
    modes = sorted({k[1] for k in mass_table})
    print(f"  cases: {len(case_ids)}, modes: {modes}")

    figures_dir = output_dir + "/figures"

    # 1. Region mass line plots per case
    # 一个 query token 有多少的注意力落在这段区域里的某个地方，不关心具体是哪个 token。
    n = plot_region_mass_per_case(mass_table, figures_dir)
    print(f"  region mass per-case plots: {n}")

    # 2. Aggregate region mass
    plot_region_mass_aggregate(mass_table, figures_dir)
    print("  region mass aggregate plot: done")

    # 3. Aggregate delta
    plot_region_mass_delta_aggregate(mass_table, figures_dir)
    print("  region mass delta aggregate plot: done")

    # 4. Local-window heatmaps per case
    # 区域内部，注意力具体压在哪个 token 上，关心具体是哪个 token。
    if window_rows:
        n = plot_heatmap_pair(window_rows, figures_dir)
        print(f"  local-window heatmap pairs: {n}")
    else:
        print("  no local_windows data, skipping heatmaps")

    # 5. Summary CSV
    write_summary_csv(mass_table, output_dir)
    print(f"  summary CSV: {Path(output_dir) / 'attention_divergence_summary.csv'}")

    print(f"\nAll outputs under {output_dir}")


if __name__ == "__main__":
    main()
