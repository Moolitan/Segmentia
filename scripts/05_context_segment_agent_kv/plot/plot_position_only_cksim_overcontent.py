#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULT_DIR = (
    ROOT
    / "results"
    / "05_context_segment_agent_kv"
    / "CKSim"
    / "position_only_cksim_overcontent"
)

DEFAULT_SOURCE = DEFAULT_RESULT_DIR / "position-only-source-0-200.pt"
DEFAULT_RECOMPUTE = DEFAULT_RESULT_DIR / "position-only-recompute-40960-41160.pt"
DEFAULT_REUSE = DEFAULT_RESULT_DIR / "position-only-rope-shift-40960-41160.pt"
DEFAULT_CSV = DEFAULT_RESULT_DIR / "position_only_overcontent_layer_cksim.csv"
DEFAULT_JSON = DEFAULT_RESULT_DIR / "position_only_overcontent_layer_cksim_summary.json"
DEFAULT_PNG = DEFAULT_RESULT_DIR / "position_only_overcontent_layer_cksim.png"


@dataclass
class LayerCKSimRow:
    comparison: str
    left_path: str
    right_path: str
    layer: str
    layer_index: int
    tokens: int
    key_cksim: float
    value_cksim: float


def setup_matplotlib() -> Any:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib.pyplot as plt

    return plt


def load_entry(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def layer_index(layer: str) -> int:
    match = re.search(r"layers\.(\d+)\.", str(layer))
    if match:
        return int(match.group(1))
    nums = re.findall(r"\d+", str(layer))
    if nums:
        return int(nums[-1])
    raise ValueError(f"Could not parse layer index from {layer!r}")


def segment_tokens(entry: dict[str, Any]) -> int:
    return int(entry["source_end"]) - int(entry["source_start"])


def as_heads(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"expected [tokens, heads, dim], got {tuple(x.shape)}")
    return x.permute(1, 0, 2).contiguous()


def cksim(a: torch.Tensor, b: torch.Tensor, tokens: int) -> float:
    a_heads = as_heads(a[:tokens]).float()
    b_heads = as_heads(b[:tokens]).float()
    if a_heads.shape != b_heads.shape:
        raise ValueError(f"KV shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    scores = F.cosine_similarity(a_heads.flatten(1), b_heads.flatten(1), dim=1)
    return float(scores.mean().item())


def compare_entries(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    comparison: str,
    left_path: Path,
    right_path: Path,
) -> list[LayerCKSimRow]:
    layers = sorted(
        set(left["kv_by_layer"]) & set(right["kv_by_layer"]),
        key=lambda name: layer_index(str(name)),
    )
    if not layers:
        raise ValueError(f"No overlapping KV layers found for {comparison}")

    tokens = min(segment_tokens(left), segment_tokens(right))
    rows: list[LayerCKSimRow] = []
    for layer in layers:
        left_k, left_v = left["kv_by_layer"][layer]
        right_k, right_v = right["kv_by_layer"][layer]
        rows.append(
            LayerCKSimRow(
                comparison=comparison,
                left_path=str(left_path),
                right_path=str(right_path),
                layer=str(layer),
                layer_index=layer_index(str(layer)),
                tokens=tokens,
                key_cksim=cksim(left_k, right_k, tokens),
                value_cksim=cksim(left_v, right_v, tokens),
            )
        )
    return rows


def write_csv(rows: list[LayerCKSimRow], path: str | Path) -> Path:
    if not rows:
        raise ValueError("No rows to write")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    return output


def summarize(rows: list[LayerCKSimRow]) -> dict[str, Any]:
    by_comparison: dict[str, list[LayerCKSimRow]] = {}
    for row in rows:
        by_comparison.setdefault(row.comparison, []).append(row)

    comparisons = {}
    for name, group in by_comparison.items():
        comparisons[name] = {
            "layers": len(group),
            "tokens": group[0].tokens,
            "mean_key_cksim": sum(row.key_cksim for row in group) / len(group),
            "mean_value_cksim": sum(row.value_cksim for row in group) / len(group),
            "min_key_cksim": min(row.key_cksim for row in group),
            "min_value_cksim": min(row.value_cksim for row in group),
        }
    return {"comparisons": comparisons}


def write_summary(rows: list[LayerCKSimRow], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summarize(rows), indent=2), encoding="utf-8")
    return output


def metric_range(rows: list[LayerCKSimRow], pad: float = 0.035) -> tuple[float, float]:
    values = [row.key_cksim for row in rows] + [row.value_cksim for row in rows]
    return max(0.0, min(values) - pad), min(1.02, max(values) + pad)


def save_layer_plot(rows: list[LayerCKSimRow], path: str | Path) -> Path:
    plt = setup_matplotlib()
    comparisons = ["initial_vs_reuse", "recompute_vs_reuse"]
    grouped = {name: [row for row in rows if row.comparison == name] for name in comparisons}
    y_min, y_max = metric_range(rows)

    fig, axes = plt.subplots(
        len(comparisons),
        1,
        figsize=(11.5, 8.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(comparisons) == 1:
        axes = [axes]

    titles = {
        "initial_vs_reuse": "Initial KV vs Reused KV",
        "recompute_vs_reuse": "Recomputed KV vs Reused KV",
    }
    for ax, name in zip(axes, comparisons, strict=True):
        group = grouped[name]
        if not group:
            raise ValueError(f"No rows found for comparison {name}")
        layers = [row.layer_index for row in group]
        ax.plot(
            layers,
            [row.key_cksim for row in group],
            marker="o",
            linewidth=2.1,
            color="#28666e",
            label="Key CKSim",
        )
        ax.plot(
            layers,
            [row.value_cksim for row in group],
            marker="s",
            linewidth=2.1,
            color="#b95738",
            label="Value CKSim",
        )
        ax.axhline(1.0, color="#30343b", linewidth=1.0, linestyle=":")
        ax.set_title(titles[name], fontsize=14)
        ax.set_ylabel("CKSim")
        ax.set_ylim(y_min, y_max)
        ax.grid(color="#d7dde5", linewidth=0.8)
        ax.legend(frameon=False, loc="best")

    axes[-1].set_xlabel("Layer")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Plot layer-wise CKSim for overcontext position-only KV: "
            "initial vs reuse and recompute vs reuse."
        )
    )
    ap.add_argument("--source", default=str(DEFAULT_SOURCE), help="initial source .pt")
    ap.add_argument("--recompute", default=str(DEFAULT_RECOMPUTE), help="recompute .pt")
    ap.add_argument("--reuse", default=str(DEFAULT_REUSE), help="RoPE-shifted reuse .pt")
    ap.add_argument("--csv", default=str(DEFAULT_CSV), help="output CSV path")
    ap.add_argument("--summary", default=str(DEFAULT_JSON), help="output JSON path")
    ap.add_argument("--png", default=str(DEFAULT_PNG), help="output layer plot path")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    source_path = Path(args.source)
    recompute_path = Path(args.recompute)
    reuse_path = Path(args.reuse)

    source = load_entry(source_path)
    recompute = load_entry(recompute_path)
    reuse = load_entry(reuse_path)

    rows = []
    rows.extend(
        compare_entries(
            source,
            reuse,
            comparison="initial_vs_reuse",
            left_path=source_path,
            right_path=reuse_path,
        )
    )
    rows.extend(
        compare_entries(
            recompute,
            reuse,
            comparison="recompute_vs_reuse",
            left_path=recompute_path,
            right_path=reuse_path,
        )
    )

    csv_path = write_csv(rows, args.csv)
    summary_path = write_summary(rows, args.summary)
    png_path = save_layer_plot(rows, args.png)

    stats = summarize(rows)["comparisons"]
    print(f"[done] rows={len(rows)}")
    for name, item in stats.items():
        print(
            f"[done] {name}: mean key={item['mean_key_cksim']:.6f}, "
            f"mean value={item['mean_value_cksim']:.6f}"
        )
    print(f"[done] csv:     {csv_path}")
    print(f"[done] summary: {summary_path}")
    print(f"[done] png:     {png_path}")


if __name__ == "__main__":
    main()
