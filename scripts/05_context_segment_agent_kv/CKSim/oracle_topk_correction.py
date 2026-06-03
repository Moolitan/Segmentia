#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[3]

SUMMARY_PATH = ROOT / "results" / "05_context_segment_agent_kv" / "CKSim" / "skill_cksim_summary.json"
OUTPUT_ROOT = ROOT / "results" / "05_context_segment_agent_kv" / "CKSim"
KV_CACHE_DIR = OUTPUT_ROOT / "kv_cache"
CSV_PATH = OUTPUT_ROOT / "oracle_topk_correction.csv"
JSON_PATH = OUTPUT_ROOT / "oracle_topk_correction_summary.json"

TOPK_RATIOS = [0.0, 0.01, 0.05, 0.10, 0.20, 0.30]
ALPHAS = [1.0, 0.75, 0.50, 0.25]

# Selection uses full recompute KV, so this is an offline upper-bound selector.
# Options: "key", "value", "kv_mean".
DRIFT_METRIC = "kv_mean"


@dataclass
class CorrectionRow:
    skill_name: str
    layer: str
    layer_idx: int
    skill_tokens: int
    topk_ratio: float
    selected_tokens: int
    alpha: float
    key_cksim: float
    value_cksim: float
    key_gain: float
    value_gain: float


def load_summary() -> dict[str, Any]:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"{SUMMARY_PATH} not found. Run skill_cksim_benchmark.py first."
        )
    return json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))


def load_entry(cache_id: str) -> dict[str, Any]:
    path = KV_CACHE_DIR / f"{cache_id}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def layer_idx(layer_name: str) -> int:
    match = re.search(r"layers\.(\d+)\.", layer_name)
    return int(match.group(1)) if match else -1


def layer_cksim(a: torch.Tensor, b: torch.Tensor, tokens: int) -> float:
    # Saved tensors are [tokens, heads, head_dim]. CKSim averages cosine over heads.
    a_heads = a[:tokens].permute(1, 0, 2).contiguous().float()
    b_heads = b[:tokens].permute(1, 0, 2).contiguous().float()
    return float(
        F.cosine_similarity(a_heads.flatten(1), b_heads.flatten(1), dim=1)
        .mean()
        .item()
    )


def token_cksim(a: torch.Tensor, b: torch.Tensor, tokens: int) -> torch.Tensor:
    a = a[:tokens].float()
    b = b[:tokens].float()
    return F.cosine_similarity(a, b, dim=2).mean(dim=1)


def select_drift(
    reuse_k: torch.Tensor,
    reuse_v: torch.Tensor,
    base_k: torch.Tensor,
    base_v: torch.Tensor,
    tokens: int,
) -> torch.Tensor:
    key_drift = 1.0 - token_cksim(reuse_k, base_k, tokens)
    value_drift = 1.0 - token_cksim(reuse_v, base_v, tokens)
    if DRIFT_METRIC == "key":
        return key_drift
    if DRIFT_METRIC == "value":
        return value_drift
    if DRIFT_METRIC == "kv_mean":
        return (key_drift + value_drift) / 2.0
    raise ValueError(f"Unknown DRIFT_METRIC={DRIFT_METRIC!r}")


def corrected_tensor(
    reuse: torch.Tensor,
    base: torch.Tensor,
    selected: torch.Tensor,
    alpha: float,
    tokens: int,
) -> torch.Tensor:
    corrected = reuse[:tokens].clone()
    if selected.numel() > 0:
        corrected[selected] = alpha * base[:tokens][selected] + (1.0 - alpha) * corrected[selected]
    return corrected


def main() -> None:
    summary = load_summary()
    rows: list[CorrectionRow] = []

    for case in summary["cases"]:
        skill = case["skill_name"]
        tokens = int(case["skill_tokens"])
        base = load_entry(f"cksim-base-{skill}")
        reuse = load_entry(f"cksim-reuse-{skill}")
        layers = sorted(
            set(base["kv_by_layer"].keys()) & set(reuse["kv_by_layer"].keys()),
            key=layer_idx,
        )
        for layer in layers:
            base_k, base_v = base["kv_by_layer"][layer]
            reuse_k, reuse_v = reuse["kv_by_layer"][layer]
            baseline_key = layer_cksim(reuse_k, base_k, tokens)
            baseline_value = layer_cksim(reuse_v, base_v, tokens)
            drift = select_drift(reuse_k, reuse_v, base_k, base_v, tokens)
            for ratio in TOPK_RATIOS:
                selected_count = min(tokens, math.ceil(tokens * ratio))
                if selected_count <= 0:
                    selected = torch.empty(0, dtype=torch.long)
                else:
                    selected = torch.topk(drift, selected_count).indices
                for alpha in ALPHAS:
                    corr_k = corrected_tensor(reuse_k, base_k, selected, alpha, tokens)
                    corr_v = corrected_tensor(reuse_v, base_v, selected, alpha, tokens)
                    key_score = layer_cksim(corr_k, base_k, tokens)
                    value_score = layer_cksim(corr_v, base_v, tokens)
                    rows.append(
                        CorrectionRow(
                            skill_name=skill,
                            layer=layer,
                            layer_idx=layer_idx(layer),
                            skill_tokens=tokens,
                            topk_ratio=ratio,
                            selected_tokens=selected_count,
                            alpha=alpha,
                            key_cksim=key_score,
                            value_cksim=value_score,
                            key_gain=key_score - baseline_key,
                            value_gain=value_score - baseline_value,
                        )
                    )
        print(f"[done] {skill}: tokens={tokens} layers={len(layers)}")

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)

    by_setting: dict[str, dict[str, float]] = {}
    for ratio in TOPK_RATIOS:
        for alpha in ALPHAS:
            subset = [
                row for row in rows
                if row.topk_ratio == ratio and row.alpha == alpha
            ]
            key = f"topk={ratio:.2f},alpha={alpha:.2f}"
            by_setting[key] = {
                "rows": len(subset),
                "mean_key_cksim": sum(r.key_cksim for r in subset) / len(subset),
                "mean_value_cksim": sum(r.value_cksim for r in subset) / len(subset),
                "mean_key_gain": sum(r.key_gain for r in subset) / len(subset),
                "mean_value_gain": sum(r.value_gain for r in subset) / len(subset),
            }

    best_key = max(by_setting.items(), key=lambda item: item[1]["mean_key_cksim"])
    best_value = max(by_setting.items(), key=lambda item: item[1]["mean_value_cksim"])
    payload = {
        "source": "oracle top-k correction over reuse_vs_base KV",
        "drift_metric": DRIFT_METRIC,
        "topk_ratios": TOPK_RATIOS,
        "alphas": ALPHAS,
        "rows": len(rows),
        "csv_path": str(CSV_PATH),
        "by_setting": by_setting,
        "best_key_setting": {best_key[0]: best_key[1]},
        "best_value_setting": {best_value[0]: best_value[1]},
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[done] rows={len(rows)}")
    print(f"[done] csv: {CSV_PATH}")
    print(f"[done] json: {JSON_PATH}")
    print("\nSettings:")
    for setting, stats in by_setting.items():
        print(
            f"  {setting:22s} key={stats['mean_key_cksim']:.6f} "
            f"value={stats['mean_value_cksim']:.6f} "
            f"key_gain={stats['mean_key_gain']:.6f} "
            f"value_gain={stats['mean_value_gain']:.6f}"
        )


if __name__ == "__main__":
    main()
