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
    / "position_only_cksim_test"
)

DEFAULT_SOURCE = DEFAULT_RESULT_DIR / "position-only-source-0-200.pt"
DEFAULT_SHIFTED = DEFAULT_RESULT_DIR / "position-only-rope-shift-3000-3200.pt"
DEFAULT_CSV = DEFAULT_RESULT_DIR / "position_only_cksim.csv"
DEFAULT_JSON = DEFAULT_RESULT_DIR / "position_only_cksim_summary.json"
DEFAULT_LAYER_PNG = DEFAULT_RESULT_DIR / "position_only_layer_cksim.png"
DEFAULT_TOKEN_PNG = DEFAULT_RESULT_DIR / "position_only_token_cksim.png"


@dataclass
class LayerCKSimRow:
    comparison: str
    layer: str
    layer_index: int
    tokens: int
    key_cksim: float
    value_cksim: float
    key_token_mean: float
    value_token_mean: float


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


def as_heads(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"expected [tokens, heads, dim], got {tuple(x.shape)}")
    return x.permute(1, 0, 2).contiguous()


def cksim(a: torch.Tensor, b: torch.Tensor, tokens: int) -> tuple[float, float]:
    a_heads = as_heads(a[:tokens]).float()
    b_heads = as_heads(b[:tokens]).float()
    if a_heads.shape != b_heads.shape:
        raise ValueError(f"KV shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    head_scores = F.cosine_similarity(a_heads.flatten(1), b_heads.flatten(1), dim=1)
    token_scores = F.cosine_similarity(a_heads, b_heads, dim=2)
    return float(head_scores.mean().item()), float(token_scores.mean().item())


def token_cksim(a: torch.Tensor, b: torch.Tensor, tokens: int) -> torch.Tensor:
    scores = F.cosine_similarity(a[:tokens].float(), b[:tokens].float(), dim=2)
    return scores.mean(dim=1)


def compare_entries(
    source: dict[str, Any],
    shifted: dict[str, Any],
    comparison: str,
) -> tuple[list[LayerCKSimRow], dict[str, list[float]]]:
    layers = sorted(
        set(source["kv_by_layer"]) & set(shifted["kv_by_layer"]),
        key=lambda name: layer_index(str(name)),
    )
    if not layers:
        raise ValueError("No overlapping KV layers found")

    source_tokens = int(source["source_end"]) - int(source["source_start"])
    shifted_tokens = int(shifted["source_end"]) - int(shifted["source_start"])
    tokens = min(source_tokens, shifted_tokens)

    rows: list[LayerCKSimRow] = []
    key_token_by_layer = []
    value_token_by_layer = []

    for layer in layers:
        source_k, source_v = source["kv_by_layer"][layer]
        shifted_k, shifted_v = shifted["kv_by_layer"][layer]
        key_score, key_token_mean = cksim(source_k, shifted_k, tokens)
        value_score, value_token_mean = cksim(source_v, shifted_v, tokens)
        key_token_by_layer.append(token_cksim(source_k, shifted_k, tokens))
        value_token_by_layer.append(token_cksim(source_v, shifted_v, tokens))
        rows.append(
            LayerCKSimRow(
                comparison=comparison,
                layer=str(layer),
                layer_index=layer_index(str(layer)),
                tokens=tokens,
                key_cksim=key_score,
                value_cksim=value_score,
                key_token_mean=key_token_mean,
                value_token_mean=value_token_mean,
            )
        )

    per_token = {
        "key": torch.stack(key_token_by_layer).mean(dim=0).tolist(),
        "value": torch.stack(value_token_by_layer).mean(dim=0).tolist(),
    }
    return rows, per_token


def write_csv(rows: list[LayerCKSimRow], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    return output


def write_summary(
    rows: list[LayerCKSimRow],
    per_token: dict[str, list[float]],
    source_path: Path,
    shifted_path: Path,
    path: str | Path,
) -> Path:
    output = Path(path)
    key_scores = [row.key_cksim for row in rows]
    value_scores = [row.value_cksim for row in rows]
    payload = {
        "comparison": rows[0].comparison,
        "source_path": str(source_path),
        "shifted_path": str(shifted_path),
        "layers": len(rows),
        "tokens": rows[0].tokens,
        "mean_key_cksim": sum(key_scores) / len(key_scores),
        "mean_value_cksim": sum(value_scores) / len(value_scores),
        "per_token": per_token,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output


def metric_range(values: list[float], pad: float = 0.04) -> tuple[float, float]:
    return max(0.0, min(values) - pad), min(1.02, max(values) + pad)


def save_layer_curve(rows: list[LayerCKSimRow], path: str | Path) -> Path:
    plt = setup_matplotlib()
    layers = [row.layer_index for row in rows]
    key_scores = [row.key_cksim for row in rows]
    value_scores = [row.value_cksim for row in rows]
    y_min, y_max = metric_range(key_scores + value_scores)

    fig, ax = plt.subplots(figsize=(11, 5.8), constrained_layout=True)
    ax.plot(
        layers,
        key_scores,
        marker="o",
        linewidth=2.2,
        color="#2f6f9f",
        label="Key CKSim",
    )
    ax.plot(
        layers,
        value_scores,
        marker="s",
        linewidth=2.2,
        color="#c65f3b",
        label="Value CKSim",
    )
    ax.axhline(1.0, color="#2f3033", linewidth=1.0, linestyle=":")
    ax.set_title("Position-only KV shift: source vs RoPE-shifted", fontsize=15)
    ax.set_xlabel("Layer")
    ax.set_ylabel("CKSim")
    ax.set_ylim(y_min, y_max)
    ax.grid(color="#d7dde5", linewidth=0.8)
    ax.legend(frameon=False, loc="best")

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def save_token_curve(per_token: dict[str, list[float]], path: str | Path) -> Path:
    plt = setup_matplotlib()
    tokens = list(range(len(per_token["key"])))
    key_scores = per_token["key"]
    value_scores = per_token["value"]
    y_min, y_max = metric_range(key_scores + value_scores)

    fig, ax = plt.subplots(figsize=(11, 5.2), constrained_layout=True)
    ax.plot(
        tokens,
        key_scores,
        linewidth=1.9,
        color="#2f6f9f",
        label="Key token CKSim",
    )
    ax.plot(
        tokens,
        value_scores,
        linewidth=1.9,
        color="#c65f3b",
        label="Value token CKSim",
    )
    ax.axhline(1.0, color="#2f3033", linewidth=1.0, linestyle=":")
    ax.set_title("Token-wise CKSim averaged across layers", fontsize=15)
    ax.set_xlabel("Token index in segment")
    ax.set_ylabel("Average CKSim")
    ax.set_ylim(y_min, y_max)
    ax.grid(color="#d7dde5", linewidth=0.8)
    ax.legend(frameon=False, loc="best")

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Plot CKSim for position-only source KV vs RoPE-shifted KV."
    )
    ap.add_argument("--source", default=str(DEFAULT_SOURCE), help="source .pt path")
    ap.add_argument("--shifted", default=str(DEFAULT_SHIFTED), help="shifted .pt path")
    ap.add_argument("--csv", default=str(DEFAULT_CSV), help="output CSV path")
    ap.add_argument("--summary", default=str(DEFAULT_JSON), help="output JSON path")
    ap.add_argument("--layer-png", default=str(DEFAULT_LAYER_PNG), help="layer plot path")
    ap.add_argument("--token-png", default=str(DEFAULT_TOKEN_PNG), help="token plot path")
    ap.add_argument(
        "--comparison",
        default="source_vs_rope_shifted",
        help="comparison label written to CSV/JSON",
    )
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    source_path = Path(args.source)
    shifted_path = Path(args.shifted)
    source = load_entry(source_path)
    shifted = load_entry(shifted_path)

    rows, per_token = compare_entries(source, shifted, args.comparison)
    csv_path = write_csv(rows, args.csv)
    summary_path = write_summary(rows, per_token, source_path, shifted_path, args.summary)
    layer_png = save_layer_curve(rows, args.layer_png)
    token_png = save_token_curve(per_token, args.token_png)

    mean_key = sum(row.key_cksim for row in rows) / len(rows)
    mean_value = sum(row.value_cksim for row in rows) / len(rows)
    print(f"[done] rows={len(rows)}")
    print(f"[done] mean key   CKSim = {mean_key:.6f}")
    print(f"[done] mean value CKSim = {mean_value:.6f}")
    print(f"[done] csv:       {csv_path}")
    print(f"[done] summary:   {summary_path}")
    print(f"[done] layer png: {layer_png}")
    print(f"[done] token png: {token_png}")


if __name__ == "__main__":
    main()
