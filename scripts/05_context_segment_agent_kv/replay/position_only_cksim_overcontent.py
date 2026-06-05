#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
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
    / "position_only_cksim_overcontent"
)

SOURCE_CACHE_ID = "position-only-source-0-200"
RECOMPUTE_CACHE_ID = "position-only-recompute-40960-41160"
SHIFTED_CACHE_ID = "position-only-rope-shift-40960-41160"

SOURCE_START = 0
SOURCE_END = 200
TARGET_START = 40960
TARGET_END = 41160
FILLER_START = SOURCE_END
FILLER_END = TARGET_START

TP = 1
SOURCE_MAX_MODEL_LEN = 4096
OVERCONTEXT_MAX_MODEL_LEN = 45056
MAX_NUM_BATCHED_TOKENS = 8192
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


def config_max_position_embeddings(model: str) -> int | None:
    cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
    value = getattr(cfg, "max_position_embeddings", None)
    return int(value) if value is not None else None


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


def build_overcontext_prompt(
    model: str,
    source_token_ids: list[int],
) -> tuple[str, list[int], list[int]]:
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    filler_seed = """
    This is neutral filler text for an over-context position-only probe. It is
    intentionally unrelated to the cached source span and only exists to move the
    repeated source text past the pretrained context window.
    """
    filler_seed_ids = tokenizer.encode(filler_seed, add_special_tokens=False)
    if not filler_seed_ids:
        raise RuntimeError("filler text produced no tokens")

    filler_len = FILLER_END - FILLER_START
    repeats = (filler_len // len(filler_seed_ids)) + 1
    filler_token_ids = (filler_seed_ids * repeats)[:filler_len]
    prompt_token_ids = source_token_ids + filler_token_ids + source_token_ids
    if len(prompt_token_ids) != TARGET_END:
        raise RuntimeError(
            f"expected {TARGET_END} prompt tokens, got {len(prompt_token_ids)}"
        )
    text = tokenizer.decode(prompt_token_ids, skip_special_tokens=False)
    return text, prompt_token_ids, filler_token_ids


def drain(engine: LLMEngine) -> None:
    while engine.has_unfinished_requests():
        engine.step()


def maybe_allow_long_max_model_len(
    model: str,
    max_model_len: int,
    *,
    allow_long_max_model_len: bool,
) -> dict[str, int]:
    config_max_len = config_max_position_embeddings(model)
    if config_max_len is None or max_model_len <= config_max_len:
        return {}
    if not allow_long_max_model_len:
        raise ValueError(
            f"requested max_model_len={max_model_len}, but model config declares "
            f"max_position_embeddings={config_max_len}. vLLM rejects this unless "
            "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 is set."
        )
    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    return {"max_position_embeddings": max_model_len}


def prefill_and_save_kv(
    model: str,
    output_dir: Path,
    cache_id: str,
    token_ids: list[int],
    *,
    source_start: int,
    source_end: int,
    request_id: str,
    max_model_len: int,
    max_num_batched_tokens: int,
    gpu_memory_utilization: float,
    allow_long_max_model_len: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if len(token_ids) + 1 > max_model_len:
        raise ValueError(
            f"request {request_id} has {len(token_ids)} prompt tokens plus one "
            f"generation token, exceeding max_model_len={max_model_len}"
        )
    hf_overrides = maybe_allow_long_max_model_len(
        model,
        max_model_len,
        allow_long_max_model_len=allow_long_max_model_len,
    )
    os.environ["VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR"] = str(output_dir)
    os.environ.pop("VLLM_CONTEXT_SEGMENT_KV_DIR", None)

    engine_args = EngineArgs(
        model=model,
        trust_remote_code=True,
        hf_overrides=hf_overrides,
        tensor_parallel_size=TP,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        disable_log_stats=True,
        enforce_eager=False,
    )
    engine = LLMEngine.from_engine_args(engine_args)

    cfg = json.dumps(
        {
            "sources": [
                {
                    "cache_id": cache_id,
                    "source_start": source_start,
                    "source_end": source_end,
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
        request_id=request_id,
        prompt=TokensPrompt(prompt_token_ids=token_ids),
        params=params,
    )
    drain(engine)
    del engine
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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


def write_text_artifacts(
    output_dir: Path,
    model: str,
    source_text: str,
    source_token_ids: list[int],
    overcontext_text: str,
    overcontext_token_ids: list[int],
    filler_token_ids: list[int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "source_text.txt").write_text(source_text, encoding="utf-8")
    (output_dir / "overcontext_prompt.txt").write_text(
        overcontext_text, encoding="utf-8"
    )
    (output_dir / "tokens.json").write_text(
        json.dumps(
            {
                "model": model,
                "source_token_count": len(source_token_ids),
                "filler_token_count": len(filler_token_ids),
                "overcontext_token_count": len(overcontext_token_ids),
                "source_token_ids": source_token_ids,
                "filler_token_ids": filler_token_ids,
                "overcontext_token_ids": overcontext_token_ids,
                "source_range": [SOURCE_START, SOURCE_END],
                "filler_range": [FILLER_START, FILLER_END],
                "target_range": [TARGET_START, TARGET_END],
                "pretrained_context_window": TARGET_START,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute KV for a 200-token text at [0,200), pad arbitrary content "
            "to 40960 tokens, recompute that same text at [40960,41160), and "
            "also save a RoPE-shifted copy without recomputing that span."
        )
    )
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument(
        "--skip-prefill",
        action="store_true",
        help=(
            "Reuse existing source/recompute .pt files and only write the "
            "RoPE-shifted cache."
        ),
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=MAX_NUM_BATCHED_TOKENS,
        help=(
            "vLLM chunked-prefill batch token budget. Lower this if the 41160-token "
            "recompute prefill runs out of memory."
        ),
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=GPU_MEM_UTIL,
        help="Forwarded to vLLM EngineArgs.",
    )
    parser.add_argument(
        "--no-allow-long-max-model-len",
        action="store_true",
        help=(
            "Do not set VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 when overcontext "
            "max_model_len exceeds the model config."
        ),
    )
    args = parser.parse_args()

    model = args.model
    output_dir = Path(args.output_dir)
    source_path = output_dir / f"{SOURCE_CACHE_ID}.pt"
    recompute_path = output_dir / f"{RECOMPUTE_CACHE_ID}.pt"

    source_text, source_token_ids = build_200_token_text(model)
    if len(source_token_ids) != SOURCE_END:
        raise RuntimeError(f"expected 200 tokens, got {len(source_token_ids)}")
    overcontext_text, overcontext_token_ids, filler_token_ids = build_overcontext_prompt(
        model, source_token_ids
    )
    write_text_artifacts(
        output_dir,
        model,
        source_text,
        source_token_ids,
        overcontext_text,
        overcontext_token_ids,
        filler_token_ids,
    )

    if not args.skip_prefill:
        source_path = prefill_and_save_kv(
            model,
            output_dir,
            SOURCE_CACHE_ID,
            source_token_ids,
            source_start=SOURCE_START,
            source_end=SOURCE_END,
            request_id="position-only-source-prefill",
            max_model_len=SOURCE_MAX_MODEL_LEN,
            max_num_batched_tokens=args.max_num_batched_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            allow_long_max_model_len=not args.no_allow_long_max_model_len,
        )
        recompute_path = prefill_and_save_kv(
            model,
            output_dir,
            RECOMPUTE_CACHE_ID,
            overcontext_token_ids,
            source_start=TARGET_START,
            source_end=TARGET_END,
            request_id="position-only-overcontext-recompute",
            max_model_len=OVERCONTEXT_MAX_MODEL_LEN,
            max_num_batched_tokens=args.max_num_batched_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            allow_long_max_model_len=not args.no_allow_long_max_model_len,
        )
    else:
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        if not recompute_path.exists():
            raise FileNotFoundError(recompute_path)

    rope_theta = config_rope_theta(model)
    shifted_path = save_shifted_kv(
        source_path,
        output_dir,
        shifted_cache_id=SHIFTED_CACHE_ID,
        rope_theta=rope_theta,
    )

    print(f"model: {model}")
    print(f"source_token_count: {len(source_token_ids)}")
    print(f"filler_token_count: {len(filler_token_ids)}")
    print(f"overcontext_token_count: {len(overcontext_token_ids)}")
    print(f"source_kv: {source_path}")
    print(f"recompute_kv: {recompute_path}")
    print(f"shifted_kv: {shifted_path}")
    print(f"source_text: {output_dir / 'source_text.txt'}")
    print(f"overcontext_prompt: {output_dir / 'overcontext_prompt.txt'}")
    print(f"tokens: {output_dir / 'tokens.json'}")


if __name__ == "__main__":
    main()
