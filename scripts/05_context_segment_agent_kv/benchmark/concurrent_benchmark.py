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


MODEL = "/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B"
TP = 1
MAX_MODEL_LEN = 16384
GPU_MEM_UTIL = 0.9

KV_SAVE_DIR = (
    ROOT / "results" / "05_context_segment_agent_kv" / "concurrent_benchmark" / "kv_cache"
)
OUTPUT_PATH = (
    ROOT / "results" / "05_context_segment_agent_kv" / "concurrent_benchmark" / "concurrent_results.csv"
)

# Each concurrent batch contains one request per prefix size.
PREFIX_SIZES  = [128, 512, 1024, 2048]   # tokens before the injected segment
SEGMENT_SIZES = [512, 1024, 2048, 4096]  # tokens in the cached segment

WARMUP_TOKENS = 256

_SEGMENT_TOK = 2
_QUERY_TOK   = 3



def make_unique_prefix_tokens(n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randint(10, 999) for _ in range(n)]


def make_segment_tokens(n: int) -> list[int]:
    return [_SEGMENT_TOK] * n


def _drain(engine: LLMEngine) -> None:
    while engine.has_unfinished_requests():
        engine.step()


def measure_single_ttft(
    engine: LLMEngine,
    request_id: str,
    tokens: list[int],
    sampling_params: SamplingParams,
) -> float:
    """Single-request TTFT — used for warmup and offline prefill."""
    engine.add_request(
        request_id=request_id,
        prompt=TokensPrompt(prompt_token_ids=tokens),
        params=sampling_params,
    )
    start = time.perf_counter()
    while engine.has_unfinished_requests():
        for out in engine.step():
            if out.request_id == request_id and out.outputs and out.outputs[0].token_ids:
                ttft = time.perf_counter() - start
                engine.abort_request([request_id])
                _drain(engine)
                return ttft
    raise RuntimeError(f"request {request_id!r} produced no output")


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
    measure_single_ttft(engine, req_id, segment_tokens, params)
    pt_path = KV_SAVE_DIR / f"{cache_id}.pt"
    print(f"  [offline] {cache_id}: {len(segment_tokens)} tokens  →  {pt_path}")


@dataclass
class RequestSpec:
    req_id: str
    tokens: list[int]
    params: SamplingParams
    prefix_size: int


def measure_concurrent_ttft(
    engine: LLMEngine,
    specs: list[RequestSpec],
) -> dict[str, float]:
    """Submit all requests simultaneously, return per-request TTFT (seconds)."""
    for spec in specs:
        engine.add_request(
            request_id=spec.req_id,
            prompt=TokensPrompt(prompt_token_ids=spec.tokens),
            params=spec.params,
        )

    start = time.perf_counter()
    ttfts: dict[str, float] = {}
    pending = {spec.req_id for spec in specs}

    while engine.has_unfinished_requests():
        for out in engine.step():
            if (
                out.request_id in pending
                and out.outputs
                and out.outputs[0].token_ids
            ):
                ttfts[out.request_id] = time.perf_counter() - start
                pending.discard(out.request_id)
                engine.abort_request([out.request_id])

    _drain(engine)

    if pending:
        raise RuntimeError(f"requests never produced output: {pending}")
    return ttfts


@dataclass
class ConcurrentResult:
    mode: str           # "baseline" | "kv_inject"
    seg_size: int
    n_requests: int
    prefix_sizes: str   # e.g. "128,512,1024,2048"
    avg_ttft_ms: float
    max_ttft_ms: float
    total_time_ms: float
    throughput_rps: float
    speedup_avg: float  # baseline_avg / inject_avg; 1.0 for baseline rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Concurrent TTFT benchmark: N users sharing one skill segment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--offline-prefill-only", action="store_true",
        help="Run Phase 1 (offline prefill) only, then exit",
    )
    args = parser.parse_args()

    KV_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR"] = str(KV_SAVE_DIR)
    os.environ["VLLM_CONTEXT_SEGMENT_KV_DIR"]      = str(KV_SAVE_DIR)
    print(f"[kv dir] {KV_SAVE_DIR}")

    engine_args = EngineArgs(
        model=MODEL,
        tensor_parallel_size=TP,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM_UTIL,
        enable_prefix_caching=True,
        enforce_eager=False,
    )
    engine = LLMEngine.from_engine_args(engine_args)

    req_counter = [0]

    def next_id(tag: str) -> str:
        rid = f"{tag}-{req_counter[0]}"
        req_counter[0] += 1
        return rid

    # Warmup
    if WARMUP_TOKENS > 0:
        print(f"\n[warmup] {WARMUP_TOKENS} tokens ...")
        measure_single_ttft(
            engine, next_id("warmup"),
            make_unique_prefix_tokens(WARMUP_TOKENS, seed=0),
            SamplingParams(max_tokens=1, temperature=0.0),
        )
        print("[warmup] done\n")

    # Phase 1: offline prefill
    print("=" * 60)
    print("Phase 1: offline prefill (saving .pt files)")
    print("=" * 60)
    for seg_size in SEGMENT_SIZES:
        offline_prefill_segment(
            engine,
            cache_id=f"conc-seg-{seg_size}",
            segment_tokens=make_segment_tokens(seg_size),
            req_id=next_id("offline"),
        )
    print()

    if args.offline_prefill_only:
        print("[done] offline prefill only — exiting")
        return

    # Phase 2: concurrent measurement
    print("=" * 60)
    print("Phase 2: concurrent TTFT measurement")
    print(f"  Batch = {len(PREFIX_SIZES)} requests with prefix sizes {PREFIX_SIZES}")
    print("=" * 60)

    results: list[ConcurrentResult] = []

    for seg_size in SEGMENT_SIZES:
        cache_id   = f"conc-seg-{seg_size}"
        seg_tokens = make_segment_tokens(seg_size)

        # Skip any prefix where total > MAX_MODEL_LEN
        valid_prefixes = [
            p for p in PREFIX_SIZES
            if p + seg_size + 1 <= MAX_MODEL_LEN
        ]
        if not valid_prefixes:
            print(f"  [skip] seg={seg_size}: all prefix sizes exceed MAX_MODEL_LEN")
            continue

        n = len(valid_prefixes)
        prefix_str = ",".join(str(p) for p in valid_prefixes)

        # Build token sequences — unique seed per (seg_size, prefix_size) pair
        # so there are no accidental prefix-cache hits across measurements.
        def build_tokens(prefix_size: int) -> list[int]:
            seed = seg_size * 100000 + prefix_size
            return make_unique_prefix_tokens(prefix_size, seed) + seg_tokens + [_QUERY_TOK]

        # --- baseline ---
        base_specs = [
            RequestSpec(
                req_id=next_id("base"),
                tokens=build_tokens(p),
                params=SamplingParams(max_tokens=1, temperature=0.0),
                prefix_size=p,
            )
            for p in valid_prefixes
        ]
        base_ttfts = measure_concurrent_ttft(engine, base_specs)
        base_values = [base_ttfts[s.req_id] * 1000 for s in base_specs]
        base_avg = round(sum(base_values) / n, 2)
        base_max = round(max(base_values), 2)
        base_total = round(max(base_values), 2)  # wall-clock = slowest request
        base_tput  = round(n / (base_total / 1000), 2)

        results.append(ConcurrentResult(
            mode="baseline", seg_size=seg_size, n_requests=n,
            prefix_sizes=prefix_str,
            avg_ttft_ms=base_avg, max_ttft_ms=base_max,
            total_time_ms=base_total, throughput_rps=base_tput,
            speedup_avg=1.0,
        ))
        print(f"  baseline  seg={seg_size:5d}  N={n}"
              f"  avg={base_avg:7.1f}ms  max={base_max:7.1f}ms"
              f"  tput={base_tput:.2f} req/s")

        # --- kv_inject ---
        inject_specs = []
        for p in valid_prefixes:
            inject_end = p + seg_size
            kv_cfg = json.dumps({
                "targets": [{
                    "cache_id": cache_id,
                    "mode": "rope",
                    "target_start": p,
                    "target_end": inject_end,
                }]
            })
            inject_specs.append(RequestSpec(
                req_id=next_id("inject"),
                tokens=build_tokens(p),
                params=SamplingParams(
                    max_tokens=1, temperature=0.0,
                    extra_args={"context_segment_cache": kv_cfg},
                ),
                prefix_size=p,
            ))

        inject_ttfts = measure_concurrent_ttft(engine, inject_specs)
        inj_values = [inject_ttfts[s.req_id] * 1000 for s in inject_specs]
        inj_avg   = round(sum(inj_values) / n, 2)
        inj_max   = round(max(inj_values), 2)
        inj_total = round(max(inj_values), 2)
        inj_tput  = round(n / (inj_total / 1000), 2)
        speedup   = round(base_avg / inj_avg, 2) if inj_avg > 0 else 0.0
        tput_speedup = round(inj_tput / base_tput, 2) if base_tput > 0 else 0.0

        results.append(ConcurrentResult(
            mode="kv_inject", seg_size=seg_size, n_requests=n,
            prefix_sizes=prefix_str,
            avg_ttft_ms=inj_avg, max_ttft_ms=inj_max,
            total_time_ms=inj_total, throughput_rps=inj_tput,
            speedup_avg=speedup,
        ))
        print(f"  kv_inject seg={seg_size:5d}  N={n}"
              f"  avg={inj_avg:7.1f}ms  max={inj_max:7.1f}ms"
              f"  tput={inj_tput:.2f} req/s"
              f"  speedup(avg)={speedup:.2f}x  tput_speedup={tput_speedup:.2f}x")

        # Per-request breakdown
        for spec, base_ms, inj_ms in zip(
            inject_specs,
            base_values,
            inj_values,
        ):
            sp = round(base_ms / inj_ms, 2) if inj_ms > 0 else 0.0
            print(f"    prefix={spec.prefix_size:5d}  "
                  f"base={base_ms:7.1f}ms  inj={inj_ms:7.1f}ms  speedup={sp:.2f}x")

    # Write CSV
    out = Path(OUTPUT_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(r) for r in results)
    print(f"\n[done] {len(results)} rows  →  {out}")


if __name__ == "__main__":
    main()
