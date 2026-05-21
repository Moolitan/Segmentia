#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

from vllm import LLMEngine, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import TokensPrompt
from vllm.outputs import RequestOutput



MODEL = "/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B"
TP = 1
MAX_MODEL_LEN = 16384
GPU_MEM_UTIL = 0.9


KV_SAVE_DIR = ROOT / "results" / "05_context_segment_agent_kv" / "Latency_test" / "kv_cache"

OUTPUT_PATH = ROOT / "results" / "05_context_segment_agent_kv" / "Latency_test" / "latency_results.csv"

PREFIX_SIZES  = [128, 512, 1024, 2048]   # tokens before the injected segment
SEGMENT_SIZES = [512, 1024, 2048, 4096]  # tokens in the cached segment

WARMUP_TOKENS = 256  # single warmup prefill to heat up GPU kernels (0 = skip)

# Fixed token IDs used for synthetic sequences.
_SEGMENT_TOK = 2   # segment body (consistent between offline prefill and injection)
_QUERY_TOK   = 3   # one dummy "query" token appended after the segment


def make_unique_prefix_tokens(n: int, seed: int) -> list[int]:
    # Unique random tokens per measurement so no accidental prefix-cache hits
    # bleed between different (prefix_size, seg_size) pairs.
    # Avoid 0 (pad), 1 (BOS), 2 (EOS); use IDs 10..999 which are safe for
    # most BPE vocabularies including Qwen3.
    rng = random.Random(seed)
    return [rng.randint(10, 999) for _ in range(n)]


def make_segment_tokens(n: int) -> list[int]:
    return [_SEGMENT_TOK] * n



def _drain(engine: LLMEngine) -> None:
    while engine.has_unfinished_requests():
        engine.step()


def measure_latency(
    engine: LLMEngine,
    request_id: str,
    tokens: list[int],
    sampling_params: SamplingParams,
) -> float:
    engine.add_request(
        request_id=request_id,
        prompt=TokensPrompt(prompt_token_ids=tokens),
        params=sampling_params,
    )
    start = time.perf_counter()
    while engine.has_unfinished_requests():
        engine.step()
    end = time.perf_counter()
    return end - start




def offline_prefill_segment(
    engine: LLMEngine,
    cache_id: str,
    segment_tokens: list[int],
    req_id: str,
) -> None:
 
    kv_cfg = json.dumps({
        "sources": [{
            "cache_id": cache_id,
            "source_start": 0,
            "source_end": len(segment_tokens),
        }]
    })
    params = SamplingParams(
        max_tokens=1,
        temperature=0.0,
        extra_args={"context_segment_cache": kv_cfg},
    )
    measure_latency(engine, req_id, segment_tokens, params)
    pt_path = KV_SAVE_DIR / f"{cache_id}.pt"
    print(f"  [offline] {cache_id}: {len(segment_tokens)} tokens  →  {pt_path}")




@dataclass
class LatencyResult:
    mode: str            # "baseline" | "kv_inject"
    prefix_tokens: int
    segment_tokens: int
    total_tokens: int
    latency_ms: float
    speedup: float       # baseline_ms / inject_ms; 1.0 for baseline rows




def main() -> None:
    parser = argparse.ArgumentParser(
        description="Latency benchmark: baseline vs ContextSegmentKV (no HTTP)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--offline-prefill-only", action="store_true",
        help="Skip Phase 1 (offline prefill) and kv_inject measurements",
    )
    args = parser.parse_args()

    # KV save/load dir must be set as env vars BEFORE engine creation —
    # gpu_model_runner reads them during initialization.
    print(f"KV_SAVE_DIR: {KV_SAVE_DIR}")
    KV_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR"] = str(KV_SAVE_DIR)
    os.environ["VLLM_CONTEXT_SEGMENT_KV_DIR"]      = str(KV_SAVE_DIR)
    print(f"[kv dir] {KV_SAVE_DIR}")

    engine_args = EngineArgs(
        model=MODEL,
        tensor_parallel_size=TP,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM_UTIL,
        # enable_prefix_caching=True is required: without it, the first
        # scheduling step never calls allocate_new_computed_blocks, so
        # num_cached_block[request_id] is never set, and the injection step
        # (which passes num_external_computed_tokens>0) hits the
        # "new request" branch and asserts len(req_blocks)==0 — crash.
        # Prefix-cache pollution between measurements is prevented by using
        # unique random prefix tokens per (prefix_size, seg_size) pair.
        enable_prefix_caching=True,
        enforce_eager=False,
    )
    engine = LLMEngine.from_engine_args(engine_args)

    req_counter = [0]

    def next_id(tag: str) -> str:
        rid = f"{tag}-{req_counter[0]}"
        req_counter[0] += 1
        return rid

    # warmup
    if WARMUP_TOKENS > 0:
        print(f"\n[warmup] {WARMUP_TOKENS} tokens ...")
        measure_latency(engine, next_id("warmup"), make_unique_prefix_tokens(WARMUP_TOKENS, seed=0),
                     SamplingParams(max_tokens=1, temperature=0.0))
        print("[warmup] done\n")

    # Phase 1: offline prefill — always runs; saves .pt to KV_SAVE_DIR
    print("=" * 50)
    print("Phase 1: offline prefill (saving .pt files)")
    print("=" * 50)
    for seg_size in SEGMENT_SIZES:
        offline_prefill_segment(
            engine,
            cache_id=f"latency-seg-{seg_size}",
            segment_tokens=make_segment_tokens(seg_size),
            req_id=next_id("offline"),
        )
    print()

    if args.offline_prefill_only:
        print("[done] offline prefill only — exiting")
        return

    # Phase 2: Latency measurement
    print("=" * 50)
    print("Phase 2: Latency measurement")
    print("=" * 50)

    results: list[LatencyResult] = []

    for seg_size in SEGMENT_SIZES:
        cache_id    = f"latency-seg-{seg_size}"
        seg_tokens  = make_segment_tokens(seg_size)

        for prefix_size in PREFIX_SIZES:
            inject_end = prefix_size + seg_size   # end of injected span
            # +1 for the trailing query token that gives the scheduler a real
            # token to process after injection (needed because injection itself
            # sets num_computed_tokens = inject_end but produces no output token,
            # leaving num_new_tokens = num_tokens_with_spec - num_computed_tokens = 0
            # if the prompt ends exactly at inject_end).
            total = inject_end + 1
            if total > MAX_MODEL_LEN:
                print(f"  [skip] prefix={prefix_size} + seg={seg_size} + 1 = {total} > {MAX_MODEL_LEN}")
                continue

            # Unique seed per pair → unique prefix tokens → no cross-measurement
            # prefix-cache hits. Same tokens used for both baseline and kv_inject.
            pair_seed = seg_size * 100000 + prefix_size
            prefix_tokens = make_unique_prefix_tokens(prefix_size, seed=pair_seed)
            # prefix | segment | query-token
            full_tokens = prefix_tokens + seg_tokens + [_QUERY_TOK]

            # baseline: full fresh prefill, no KV cache involvement
            t_base = measure_latency(
                engine, next_id("base"), full_tokens,
                SamplingParams(max_tokens=1, temperature=0.0),
            )
            base_ms = round(t_base * 1000, 2)
            results.append(LatencyResult("baseline", prefix_size, seg_size, total, base_ms, 1.0))
            print(f"  baseline   prefix={prefix_size:5d}  seg={seg_size:5d}  "
                  f"total={total:6d}  Latency={base_ms:7.1f} ms")

            # kv_inject: prefix prefilled normally; segment KV spliced from registry;
            # query token at position inject_end triggers the decode output.
            if not args.offline_prefill_only:
                kv_cfg = json.dumps({
                    "targets": [{
                        "cache_id": cache_id,
                        "mode": "rope",
                        "target_start": prefix_size,
                        "target_end": inject_end,
                    }]
                })
                t_inject = measure_latency(
                    engine, next_id("inject"), full_tokens,
                    SamplingParams(max_tokens=1, temperature=0.0,
                                   extra_args={"context_segment_cache": kv_cfg}),
                )
                inject_ms = round(t_inject * 1000, 2)
                speedup   = round(base_ms / inject_ms, 2) if inject_ms > 0 else 0.0
                results.append(LatencyResult("kv_inject", prefix_size, seg_size, total, inject_ms, speedup))
                print(f"  kv_inject  prefix={prefix_size:5d}  seg={seg_size:5d}  "
                      f"total={total:6d}  Latency={inject_ms:7.1f} ms  speedup={speedup:.2f}x")

    # Output
    out = Path(OUTPUT_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(r) for r in results)
    print(f"\n[done] {len(results)} measurements  →  {out}")

    hdr = f"{'mode':10} {'prefix':>8} {'segment':>8} {'total':>7} {'Latency(ms)':>10} {'speedup':>8}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in results:
        spd = f"{r.speedup:.2f}x" if r.mode == "kv_inject" else "       -"
        print(f"{r.mode:10} {r.prefix_tokens:>8} {r.segment_tokens:>8} "
              f"{r.total_tokens:>7} {r.latency_ms:>10.1f} {spd:>8}")


if __name__ == "__main__":
    main()
