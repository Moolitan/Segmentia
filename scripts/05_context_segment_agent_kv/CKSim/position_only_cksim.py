#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoConfig
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb


ROOT = Path(__file__).resolve().parents[3]

MODEL = "/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B"
KV_CACHE_DIR = ROOT / "results" / "05_context_segment_agent_kv" / "CKSim" / "kv_cache"

SKILLS = [
    "internal-comms",
    "brand-guidelines",
    "canvas-design",
    "web-artifacts-builder",
    "theme-factory",
    "slack-gif-creator",
]

# Match Figure 4 style: sweep absolute position shifts.
POSITION_SHIFTS = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000]

# Qwen/Llama-family RoPE in vLLM uses NeoX style. Set to False only if testing a
# model whose vLLM RotaryEmbedding uses GPT-J style rotation.
IS_NEOX_STYLE = True


def load_entry(skill_name: str) -> dict[str, Any]:
    path = KV_CACHE_DIR / f"cksim-offline-{skill_name}.pt"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run skill_cksim_benchmark.py once to generate "
            "offline skill KV files."
        )
    return torch.load(path, map_location="cpu", weights_only=False)


def config_rope_theta() -> float:
    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    return float(getattr(cfg, "rope_theta", 10000.0))


def rotate_half_neox(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_delta_rope(
    k: torch.Tensor,
    delta: int,
    *,
    rope_theta: float,
    is_neox_style: bool,
) -> torch.Tensor:
    if delta == 0:
        return k
    head_dim = k.shape[-1]
    rotary_dim = head_dim
    inv_freq = 1.0 / (
        rope_theta
        ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    # full 用来创建一个指定形状的 Tensor，并把所有元素填成同一个值。
    freqs = torch.full((k.shape[0], 1), float(delta)) * inv_freq.unsqueeze(0)
    cos = freqs.cos().to(dtype=k.dtype)
    sin = freqs.sin().to(dtype=k.dtype)
    # is_neox_style=True 表示使用 NeoX/Llama/Qwen 常见的维度拆分方式，False 则表示使用 GPT-J 常见的维度拆分方式。
    return ApplyRotaryEmb.forward_static(k, cos, sin, is_neox_style)


def as_heads(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"expected [tokens, heads, dim], got {tuple(x.shape)}")
    return x.permute(1, 0, 2).contiguous()


def cksim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_heads = as_heads(a).float()
    b_heads = as_heads(b).float()
    return float(
        F.cosine_similarity(a_heads.flatten(1), b_heads.flatten(1), dim=1)
        .mean()
        .item()
    )


def main() -> None:
    rope_theta = config_rope_theta()
    print(f"model: {MODEL}")
    print(f"kv_cache_dir: {KV_CACHE_DIR}")
    print(f"rope_theta: {rope_theta:g}")
    print(f"is_neox_style: {IS_NEOX_STYLE}")
    print()

    entries = {skill: load_entry(skill) for skill in SKILLS}
    layers = sorted(
        set.intersection(
            *(set(entry["kv_by_layer"].keys()) for entry in entries.values())
        )
    )
    print(f"skills: {', '.join(SKILLS)}")
    print(f"layers: {len(layers)}")
    print()

    print("shift,mean_key_cksim,min_key_cksim,worst_skill,worst_layer")
    summary = []
    for shift in POSITION_SHIFTS:
        scores: list[tuple[float, str, str]] = []
        for skill, entry in entries.items():
            for layer in layers:
                k, _ = entry["kv_by_layer"][layer]
                shifted_k = apply_delta_rope(
                    k,
                    shift,
                    rope_theta=rope_theta,
                    is_neox_style=IS_NEOX_STYLE,
                )
                scores.append((cksim(shifted_k, k), skill, layer))
        mean_score = sum(score for score, _, _ in scores) / len(scores)
        min_score, worst_skill, worst_layer = min(scores, key=lambda x: x[0])
        print(
            f"{shift},{mean_score:.6f},{min_score:.6f},"
            f"{worst_skill},{worst_layer}"
        )
        summary.append(
            {
                "shift": shift,
                "mean_key_cksim": mean_score,
                "min_key_cksim": min_score,
                "worst_skill": worst_skill,
                "worst_layer": worst_layer,
            }
        )

    print()
    print(json.dumps({"position_only_cksim": summary}, indent=2))


if __name__ == "__main__":
    main()
