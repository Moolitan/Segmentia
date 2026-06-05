#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer
from vllm import LLMEngine, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import TokensPrompt
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb


ROOT = Path(__file__).resolve().parents[3]

MODEL = "/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B"
OUTPUT_DIR = (
    ROOT
    / "results"
    / "05_context_segment_agent_kv"
    / "CKSim"
    / "position_only_cksim_test"
)

SOURCE_CACHE_ID = "position-only-source-0-200"
SHIFTED_CACHE_ID = "position-only-rope-shift-3000-3200"

SOURCE_START = 0
SOURCE_END = 200
TARGET_START = 3000
TARGET_END = 3200

TP = 1
MAX_MODEL_LEN = 4096
GPU_MEM_UTIL = 0.9

# Qwen/Llama-family RoPE in vLLM uses NeoX style.
IS_NEOX_STYLE = True

BASE_TEXT = """
Context segment caching is useful when an assistant repeatedly receives the
same block of background knowledge across many turns. The cached paragraph can
describe product rules, deployment constraints, terminology, ownership notes,
and examples that rarely change during a session. A practical evaluation should
use normal prose rather than a single repeated token, because natural text gives
the model varied subwords, punctuation, and local dependencies. This sample
therefore reads like a compact engineering note. It explains that the service
keeps a stable segment, computes its key value cache once, and later moves that
segment to a different absolute position. Only the positional component of the
key cache should change under RoPE; the value cache is position independent and
is copied unchanged. The shifted cache can then be compared with a full
recompute to verify whether the position-only correction behaves as expected.
The script stores every layer so downstream analysis can inspect early, middle,
and late transformer blocks independently.
"""


def config_rope_theta(model: str) -> float:
    cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
    return float(getattr(cfg, "rope_theta", 10000.0))


def build_200_token_text(model: str) -> tuple[str, list[int]]:
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    token_ids = tokenizer.encode(BASE_TEXT, add_special_tokens=False)
    if len(token_ids) < SOURCE_END:
        repeats = (SOURCE_END // max(1, len(token_ids))) + 2
        token_ids = tokenizer.encode(BASE_TEXT * repeats, add_special_tokens=False)
    token_ids = token_ids[:SOURCE_END]
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    checked = tokenizer.encode(text, add_special_tokens=False)
    if checked[:SOURCE_END] != token_ids or len(checked) < SOURCE_END:
        raise RuntimeError(
            "decoded 200-token text did not round-trip through the tokenizer"
        )
    return text, token_ids


def drain(engine: LLMEngine) -> None:
    while engine.has_unfinished_requests():
        engine.step()


def prefill_and_save_kv(
    model: str,
    output_dir: Path,
    cache_id: str,
    token_ids: list[int],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR"] = str(output_dir)
    os.environ.pop("VLLM_CONTEXT_SEGMENT_KV_DIR", None)

    engine_args = EngineArgs(
        model=model,
        tensor_parallel_size=TP,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM_UTIL,
        enable_prefix_caching=True,
        enforce_eager=False,
    )
    engine = LLMEngine.from_engine_args(engine_args)

    cfg = json.dumps(
        {
            "sources": [
                {
                    "cache_id": cache_id,
                    "source_start": SOURCE_START,
                    "source_end": SOURCE_END,
                }
            ]
        }
    )
    params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        extra_args={"context_segment_cache": cfg},
    )
    engine.add_request(
        request_id="position-only-source-prefill",
        prompt=TokensPrompt(prompt_token_ids=token_ids),
        params=params,
    )
    drain(engine)

    path = output_dir / f"{cache_id}.pt"
    if not path.exists():
        raise FileNotFoundError(f"expected vLLM to save KV at {path}")
    return path


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
    freqs = torch.full((k.shape[0], 1), float(delta)) * inv_freq.unsqueeze(0)
    cos = freqs.cos().to(dtype=k.dtype)
    sin = freqs.sin().to(dtype=k.dtype)
    return ApplyRotaryEmb.forward_static(k, cos, sin, is_neox_style)


def load_entry(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def save_shifted_kv(
    source_path: Path,
    output_dir: Path,
    *,
    shifted_cache_id: str,
    rope_theta: float,
) -> Path:
    entry = load_entry(source_path)
    delta = TARGET_START - int(entry["source_start"])
    if int(entry["source_end"]) - int(entry["source_start"]) != SOURCE_END:
        raise ValueError(
            f"expected {SOURCE_END} source tokens, got "
            f"{int(entry['source_end']) - int(entry['source_start'])}"
        )

    shifted_kv_by_layer = {}
    for layer, (k, v) in entry["kv_by_layer"].items():
        shifted_k = apply_delta_rope(
            k,
            delta,
            rope_theta=rope_theta,
            is_neox_style=IS_NEOX_STYLE,
        )
        shifted_kv_by_layer[layer] = (shifted_k.cpu(), v.detach().cpu())

    shifted = {
        "cache_id": shifted_cache_id,
        "source_start": TARGET_START,
        "source_end": TARGET_END,
        "token_ids": entry.get("token_ids"),
        "kv_by_layer": shifted_kv_by_layer,
        "metadata": {
            "source_cache_id": entry["cache_id"],
            "source_range": [entry["source_start"], entry["source_end"]],
            "target_range": [TARGET_START, TARGET_END],
            "rope_delta": delta,
            "rope_theta": rope_theta,
            "is_neox_style": IS_NEOX_STYLE,
            "note": "key cache rerotated with RoPE; value cache copied unchanged",
        },
    }
    path = output_dir / f"{shifted_cache_id}.pt"
    torch.save(shifted, path)
    return path


def write_text_artifacts(output_dir: Path, text: str, token_ids: list[int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "source_text.txt").write_text(text, encoding="utf-8")
    (output_dir / "source_tokens.json").write_text(
        json.dumps(
            {
                "model": MODEL,
                "token_count": len(token_ids),
                "token_ids": token_ids,
                "source_range": [SOURCE_START, SOURCE_END],
                "target_range": [TARGET_START, TARGET_END],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute KV for a 200-token text at [0,200), then save a RoPE-shifted "
            "copy for [3000,3200) without recomputing the model forward pass."
        )
    )
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument(
        "--skip-prefill",
        action="store_true",
        help="Reuse an existing source .pt file and only write the shifted cache.",
    )
    args = parser.parse_args()

    model = args.model
    output_dir = Path(args.output_dir)
    source_path = output_dir / f"{SOURCE_CACHE_ID}.pt"

    text, token_ids = build_200_token_text(model)
    if len(token_ids) != SOURCE_END:
        raise RuntimeError(f"expected 200 tokens, got {len(token_ids)}")
    write_text_artifacts(output_dir, text, token_ids)

    if not args.skip_prefill:
        source_path = prefill_and_save_kv(model, output_dir, SOURCE_CACHE_ID, token_ids)

    rope_theta = config_rope_theta(model)
    shifted_path = save_shifted_kv(
        source_path,
        output_dir,
        shifted_cache_id=SHIFTED_CACHE_ID,
        rope_theta=rope_theta,
    )

    print(f"model: {model}")
    print(f"token_count: {len(token_ids)}")
    print(f"source_kv: {source_path}")
    print(f"shifted_kv: {shifted_path}")
    print(f"source_text: {output_dir / 'source_text.txt'}")
    print(f"source_tokens: {output_dir / 'source_tokens.json'}")


if __name__ == "__main__":
    main()
