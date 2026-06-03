#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[3]

MODEL = "/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B"
SUMMARY_PATH = ROOT / "results" / "05_context_segment_agent_kv" / "CKSim" / "skill_cksim_summary.json"
OUTPUT_ROOT = ROOT / "results" / "05_context_segment_agent_kv" / "CKSim"
KV_CACHE_DIR = OUTPUT_ROOT / "kv_cache"
CSV_PATH = OUTPUT_ROOT / "token_level_reuse_gap.csv"
JSON_PATH = OUTPUT_ROOT / "token_level_reuse_gap_summary.json"

TOP_N = 100


@dataclass
class TokenGapRow:
    skill_name: str
    layer: str
    layer_idx: int
    token_idx: int
    token_id: int
    token_text: str
    key_cksim: float
    value_cksim: float
    key_drift: float
    value_drift: float
    combined_drift: float


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


def token_scores(a: torch.Tensor, b: torch.Tensor, tokens: int) -> torch.Tensor:
    # Saved tensors are [tokens, heads, head_dim]. Average cosine over heads.
    a = a[:tokens].float()
    b = b[:tokens].float()
    return F.cosine_similarity(a, b, dim=2).mean(dim=1)


def clean_token_text(text: str) -> str:
    text = text.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
    return text[:80]


def load_tokenizer():
    try:
        return AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    except Exception as exc:
        print(f"[warn] tokenizer unavailable, token_text will be token ids: {exc}")
        return None


def decode_tokens(tokenizer, token_ids: list[int]) -> list[str]:
    if tokenizer is None:
        return [str(t) for t in token_ids]
    return [clean_token_text(tokenizer.decode([t])) for t in token_ids]


def main() -> None:
    summary = load_summary()
    tokenizer = load_tokenizer()
    rows: list[TokenGapRow] = []

    for case in summary["cases"]:
        skill = case["skill_name"]
        tokens = int(case["skill_tokens"])
        base = load_entry(f"cksim-base-{skill}")
        reuse = load_entry(f"cksim-reuse-{skill}")
        token_ids = list((base.get("token_ids") or [])[:tokens])
        if len(token_ids) < tokens:
            token_ids.extend([-1] * (tokens - len(token_ids)))
        token_texts = decode_tokens(tokenizer, token_ids)

        layers = sorted(
            set(base["kv_by_layer"].keys()) & set(reuse["kv_by_layer"].keys()),
            key=layer_idx,
        )
        for layer in layers:
            base_k, base_v = base["kv_by_layer"][layer]
            reuse_k, reuse_v = reuse["kv_by_layer"][layer]
            key = token_scores(reuse_k, base_k, tokens)
            value = token_scores(reuse_v, base_v, tokens)
            for idx in range(tokens):
                key_cksim = float(key[idx].item())
                value_cksim = float(value[idx].item())
                key_drift = 1.0 - key_cksim
                value_drift = 1.0 - value_cksim
                rows.append(
                    TokenGapRow(
                        skill_name=skill,
                        layer=layer,
                        layer_idx=layer_idx(layer),
                        token_idx=idx,
                        token_id=int(token_ids[idx]),
                        token_text=token_texts[idx],
                        key_cksim=key_cksim,
                        value_cksim=value_cksim,
                        key_drift=key_drift,
                        value_drift=value_drift,
                        combined_drift=(key_drift + value_drift) / 2.0,
                    )
                )
        print(f"[done] {skill}: tokens={tokens} layers={len(layers)}")

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)

    worst = sorted(rows, key=lambda r: r.combined_drift, reverse=True)[:TOP_N]
    by_skill: dict[str, dict[str, float]] = {}
    for skill in sorted({row.skill_name for row in rows}):
        subset = [row for row in rows if row.skill_name == skill]
        by_skill[skill] = {
            "rows": len(subset),
            "mean_key_cksim": sum(r.key_cksim for r in subset) / len(subset),
            "mean_value_cksim": sum(r.value_cksim for r in subset) / len(subset),
            "mean_combined_drift": sum(r.combined_drift for r in subset) / len(subset),
        }
    by_layer_bucket: dict[str, dict[str, float]] = {}
    for lo, hi in [(0, 9), (10, 19), (20, 29), (30, 39)]:
        subset = [row for row in rows if lo <= row.layer_idx <= hi]
        by_layer_bucket[f"{lo}-{hi}"] = {
            "rows": len(subset),
            "mean_key_cksim": sum(r.key_cksim for r in subset) / len(subset),
            "mean_value_cksim": sum(r.value_cksim for r in subset) / len(subset),
            "mean_combined_drift": sum(r.combined_drift for r in subset) / len(subset),
        }

    payload = {
        "source": "reuse_vs_base token-level CKSim",
        "rows": len(rows),
        "csv_path": str(CSV_PATH),
        "by_skill": by_skill,
        "by_layer_bucket": by_layer_bucket,
        "worst": [asdict(row) for row in worst],
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[done] rows={len(rows)}")
    print(f"[done] csv: {CSV_PATH}")
    print(f"[done] json: {JSON_PATH}")
    print("\nWorst token gaps:")
    for row in worst[:20]:
        print(
            f"  {row.skill_name:22s} L{row.layer_idx:02d} token={row.token_idx:4d} "
            f"key={row.key_cksim:.4f} value={row.value_cksim:.4f} "
            f"text={row.token_text!r}"
        )


if __name__ == "__main__":
    main()
