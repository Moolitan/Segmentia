"""Run the explicit-name probe with chunk 1 recomputed and chunk 3 reused.

The local vLLM server must already be running with prefix caching disabled and
the existing cache directory loaded, for example:

    export PREFIX_CACHING=0
    export VLLM_CONTEXT_SEGMENT_KV_DIR=/mnt/Large_Language_Model_Lab_1/wsh/CacheBlend/output/two_chunk_full_reuse_verify/pilot1/kv
    bash scripts/vllm_start.sh
    python scripts/06_context_free_segment_cache/cacheblend_verify/run_chunk1_recompute_chunk3_reuse.py

This probe writes no result file. It prints the request configuration,
reasoning, visible answer, finish reason, usage, and elapsed time.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_DIR = SCRIPT_DIR.parent / "module"
for import_dir in (SCRIPT_DIR, MODULE_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from run_cacheblend_verify import (
    CASES_BY_NAME,
    combined_content_for_case,
    messages_for,
    prepare_spans,
)

from vllm_client import chat_completion, extract_response  # noqa: E402


DEFAULT_KV_DIR = Path(
    "/mnt/Large_Language_Model_Lab_1/wsh/CacheBlend/output/"
    "two_chunk_full_reuse_verify/pilot1/kv"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default=os.environ.get("VLLM_SERVED_NAME", "Qwen3"))
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--kv-dir", type=Path, default=DEFAULT_KV_DIR)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=1111)
    args = parser.parse_args()

    case = CASES_BY_NAME["explicit_full_name"]
    combined_content = combined_content_for_case(case)
    spans = prepare_spans(args.base_url, args.model, args.api_key, case)
    chunk_3_matches = [span for span in spans if span["name"] == "chunk_3"]
    if len(chunk_3_matches) != 1:
        raise RuntimeError(
            f"Expected exactly one chunk_3 span, got {len(chunk_3_matches)}"
        )
    chunk_3 = chunk_3_matches[0]
    cache_path = args.kv_dir / f"{chunk_3['cache_id']}.pt"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing chunk 3 KV cache: {cache_path}")

    context_segment_cache = {
        "targets": [
            {
                "cache_id": chunk_3["cache_id"],
                "mode": "rope",
                "target_start": chunk_3["target_start"],
                "target_end": chunk_3["target_end"],
            }
        ]
    }
    response, elapsed = chat_completion(
        args.base_url,
        args.model,
        messages_for(combined_content),
        None,
        args.api_key,
        max_tokens=args.max_tokens,
        request_id="cacheblend-verify-explicit-chunk1-recompute-chunk3-reuse",
        context_segment_cache=context_segment_cache,
        temperature=0.0,
        top_p=1.0,
        seed=args.seed,
    )
    parsed = extract_response(response)

    print("=== Prompt ===")
    print(combined_content)
    print("\n=== Execution ===")
    print("chunk_1: recompute")
    print(
        "chunk_3: reuse "
        f"cache_id={chunk_3['cache_id']} "
        f"source=[{chunk_3['source_start']},{chunk_3['source_end']}) "
        f"target=[{chunk_3['target_start']},{chunk_3['target_end']}) "
        "mode=rope"
    )
    print("\n=== Reasoning ===")
    print(parsed["reasoning"])
    print("\n=== Answer ===")
    print(parsed["content"])
    print("\n=== Metadata ===")
    print(f"finish_reason: {parsed['finish_reason']}")
    print(f"usage: {response.get('usage', {})}")
    print(f"elapsed_sec: {elapsed:.4f}")


if __name__ == "__main__":
    main()
