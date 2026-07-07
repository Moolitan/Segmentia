"""Two-pair full-KV-reuse sanity experiment.

This is a zero-recompute extreme point related to CacheBlend, not the full
CacheBlend algorithm. Three evidence chunks are cached independently. Two
pairwise prompts then compare full recomputation with complete K/V injection.
Only RoPE key positions are corrected; reused chunk values remain unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_DIR = SCRIPT_DIR.parent / "module"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from vllm_client import (  # noqa: E402
    chat_completion,
    extract_response,
    post_json,
    tokenize_chat,
)


CHUNKS = (
    {
        "name": "chunk_1",
        "cache_id": "cacheblend-verify-chunk-1",
        "text": "Elon DeLi scored 13 goals at 2026 HUST cups.",
    },
    {
        "name": "chunk_2",
        "cache_id": "cacheblend-verify-chunk-2",
        "text": "Herry scored 8 goals at 2026 HUST cups.",
    },
    {
        "name": "chunk_3",
        "cache_id": "cacheblend-verify-chunk-3",
        "text": "Herry Msask scored 8 goals at 2026 HUST cups.", 
    },
)
CHUNKS_BY_NAME = {str(chunk["name"]): chunk for chunk in CHUNKS}
CASES = (
    {
        "name": "alias_only",
        "chunk_names": ("chunk_1", "chunk_2"),
    },
    {
        "name": "explicit_full_name",
        "chunk_names": ("chunk_1", "chunk_3"),
    },
)
CASES_BY_NAME = {str(case["name"]): case for case in CASES}
QUERY = "How many goals did DeLi score? How many goals did Msask score? Who scored more goals at 2026 HUST cups?"
MODES = ("recompute", "full_kv_reuse")


def messages_for(content: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": ""},
        {"role": "user", "content": content},
    ]


def tokenize_text(
    base_url: str,
    model: str,
    api_key: str,
    text: str,
) -> list[int]:
    response = post_json(
        base_url,
        "/tokenize",
        {
            "model": model,
            "prompt": text,
            "add_special_tokens": False,
        },
        api_key,
    )
    return list(response["tokens"])


def find_unique_subsequence(haystack: list[int], needle: list[int], label: str) -> tuple[int, int]:
    if not needle:
        raise ValueError(f"{label} tokenized to an empty sequence")
    starts = [
        start
        for start in range(len(haystack) - len(needle) + 1)
        if haystack[start : start + len(needle)] == needle
    ]
    if len(starts) != 1:
        raise RuntimeError(
            f"Expected one exact token match for {label}, found {len(starts)}"
        )
    return starts[0], starts[0] + len(needle)


def chunks_for_case(case: dict[str, Any]) -> list[dict[str, str]]:
    return [CHUNKS_BY_NAME[name] for name in case["chunk_names"]]


def combined_content_for_case(case: dict[str, Any]) -> str:
    return "\n".join(
        [str(chunk["text"]) for chunk in chunks_for_case(case)] + [QUERY]
    )


def prepare_spans(
    base_url: str,
    model: str,
    api_key: str,
    case: dict[str, Any],
) -> list[dict[str, Any]]:
    combined_content = combined_content_for_case(case)
    target_messages = messages_for(combined_content)
    target_tokens = tokenize_chat(
        base_url,
        model,
        target_messages,
        None,
        api_key,
        add_generation_prompt=True,
    )
    prepared: list[dict[str, Any]] = []
    for chunk in chunks_for_case(case):
        # Include the separator after each fact in the cached span. Qwen3 merges
        # the sentence-final period and following newline into one token, so the
        # bare sentence tokenization is not identical inside the combined prompt.
        cached_text = f"{chunk['text']}\n"
        chunk_tokens = tokenize_text(
            base_url,
            model,
            api_key,
            cached_text,
        )
        source_messages = messages_for(cached_text)
        source_tokens = tokenize_chat(
            base_url,
            model,
            source_messages,
            None,
            api_key,
            add_generation_prompt=True,
        )
        source_start, source_end = find_unique_subsequence(
            source_tokens,
            chunk_tokens,
            f"{chunk['name']} standalone prompt",
        )
        target_start, target_end = find_unique_subsequence(
            target_tokens,
            chunk_tokens,
            f"{chunk['name']} combined prompt",
        )
        if source_tokens[source_start:source_end] != target_tokens[target_start:target_end]:
            raise RuntimeError(f"Token identity check failed for {chunk['name']}")
        prepared.append(
            {
                **chunk,
                "case": case["name"],
                "cached_text": cached_text,
                "source_start": source_start,
                "source_end": source_end,
                "target_start": target_start,
                "target_end": target_end,
                "token_count": len(chunk_tokens),
                "source_messages": source_messages,
            }
        )
    return prepared


def prepare_all_cases(
    base_url: str,
    model: str,
    api_key: str,
) -> dict[str, list[dict[str, Any]]]:
    return {
        str(case["name"]): prepare_spans(base_url, model, api_key, case)
        for case in CASES
    }


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def collect(args: argparse.Namespace) -> None:
    args.kv_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.kv_dir / "collection_manifest.json"
    expected_paths = [
        args.kv_dir / f"{chunk['cache_id']}.pt"
        for chunk in CHUNKS
    ]
    existing = [path for path in [manifest_path, *expected_paths] if path.exists()]
    if existing:
        raise FileExistsError(
            "Collection outputs already exist; use a new RUN_DIR:\n"
            + "\n".join(f"  {path}" for path in existing)
        )

    spans_by_case = prepare_all_cases(args.base_url, args.model, args.api_key)
    source_by_cache_id: dict[str, dict[str, Any]] = {}
    targets_by_cache_id: dict[str, list[dict[str, Any]]] = {}
    for case_name, spans in spans_by_case.items():
        for span in spans:
            cache_id = str(span["cache_id"])
            previous = source_by_cache_id.setdefault(cache_id, span)
            source_keys = ("source_start", "source_end", "cached_text")
            if any(previous[key] != span[key] for key in source_keys):
                raise RuntimeError(
                    f"Inconsistent standalone source for cache_id={cache_id}"
                )
            targets_by_cache_id.setdefault(cache_id, []).append(
                {
                    "case": case_name,
                    "target_start": span["target_start"],
                    "target_end": span["target_end"],
                }
            )

    records: list[dict[str, Any]] = []
    for chunk in CHUNKS:
        span = source_by_cache_id[str(chunk["cache_id"])]
        cfg = {
            "sources": [
                {
                    "cache_id": span["cache_id"],
                    "source_start": span["source_start"],
                    "source_end": span["source_end"],
                }
            ]
        }
        response, elapsed = chat_completion(
            args.base_url,
            args.model,
            span["source_messages"],
            None,
            args.api_key,
            max_tokens=1,
            request_id=f"cacheblend-verify-collect-{span['name']}",
            context_segment_cache=cfg,
            temperature=0.0,
        )
        output_path = args.kv_dir / f"{span['cache_id']}.pt"
        if not output_path.exists():
            raise RuntimeError(
                f"vLLM did not save {span['name']} KV at {output_path}"
            )
        records.append(
            {
                key: value
                for key, value in span.items()
                if key not in {"case", "source_messages", "target_start", "target_end"}
            }
            | {
                "targets": targets_by_cache_id[str(span["cache_id"])],
                "kv_path": str(output_path),
                "elapsed_sec": round(elapsed, 4),
                "usage": response.get("usage", {}),
            }
        )

    atomic_write_json(
        manifest_path,
        {
            "experiment": "cacheblend_two_pair_full_kv_reuse_verify",
            "model": args.model,
            "query": QUERY,
            "cases": [
                {
                    "name": case["name"],
                    "chunk_names": list(case["chunk_names"]),
                    "combined_content": combined_content_for_case(case),
                }
                for case in CASES
            ],
            "records": records,
        },
    )
    print(f"[done] collected two independent chunk caches: {manifest_path}")


def decode(args: argparse.Namespace) -> None:
    if args.mode not in MODES:
        raise ValueError(f"Unsupported mode: {args.mode}")
    if args.output.exists():
        raise FileExistsError(f"Decode output already exists: {args.output}")

    case = CASES_BY_NAME[args.case]
    combined_content = combined_content_for_case(case)
    spans = prepare_spans(args.base_url, args.model, args.api_key, case)
    cfg = None
    if args.mode == "full_kv_reuse":
        missing = [
            args.kv_dir / f"{span['cache_id']}.pt"
            for span in spans
            if not (args.kv_dir / f"{span['cache_id']}.pt").exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing chunk KV files:\n"
                + "\n".join(f"  {path}" for path in missing)
            )
        cfg = {
            "targets": [
                {
                    "cache_id": span["cache_id"],
                    "mode": "rope",
                    "target_start": span["target_start"],
                    "target_end": span["target_end"],
                }
                for span in spans
            ]
        }

    response, elapsed = chat_completion(
        args.base_url,
        args.model,
        messages_for(combined_content),
        None,
        args.api_key,
        max_tokens=args.max_tokens,
        request_id=f"cacheblend-verify-{args.case}-{args.mode}",
        context_segment_cache=cfg,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        seed=args.seed,
    )
    parsed = extract_response(response)
    atomic_write_json(
        args.output,
        {
            "experiment": "cacheblend_two_pair_full_kv_reuse_verify",
            "case": args.case,
            "mode": args.mode,
            "model": args.model,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "seed": args.seed,
            "chunks": [
                {key: chunk[key] for key in ("name", "text")}
                for chunk in chunks_for_case(case)
            ],
            "query": QUERY,
            "combined_content": combined_content,
            "spans": [
                {
                    key: span[key]
                    for key in (
                        "name",
                        "cache_id",
                        "cached_text",
                        "source_start",
                        "source_end",
                        "target_start",
                        "target_end",
                        "token_count",
                    )
                }
                for span in spans
            ],
            "context_segment_cache": cfg,
            "elapsed_sec": round(elapsed, 4),
            "usage": response.get("usage", {}),
            "finish_reason": parsed["finish_reason"],
            "reasoning": parsed["reasoning"],
            "answer": parsed["content"],
            "raw_response": response,
        },
    )
    print(
        f"[done] case={args.case} mode={args.mode} answer={parsed['content']!r}"
    )
    print(f"[done] output={args.output}")


def summarize(args: argparse.Namespace) -> None:
    if args.summary_output.exists():
        raise FileExistsError(
            f"Summary output already exists; choose a new SUMMARY_OUTPUT: "
            f"{args.summary_output}"
        )
    rows = []
    for case in CASES:
        for mode in MODES:
            path = args.run_dir / "raw" / str(case["name"]) / f"{mode}.json"
            if not path.exists():
                raise FileNotFoundError(f"Missing decode output: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "case": case["name"],
                    "chunk_names": list(case["chunk_names"]),
                    "mode": mode,
                    "temperature": payload.get("temperature"),
                    "top_p": payload.get("top_p"),
                    "top_k": payload.get("top_k"),
                    "min_p": payload.get("min_p"),
                    "seed": payload.get("seed"),
                    "answer": payload["answer"],
                    "reasoning": payload["reasoning"],
                    "finish_reason": payload["finish_reason"],
                    "elapsed_sec": payload["elapsed_sec"],
                    "usage": payload["usage"],
                    "source_path": str(path),
                }
            )
    atomic_write_json(
        args.summary_output,
        {
            "experiment": "cacheblend_two_pair_full_kv_reuse_verify",
            "expected_answer": "Elon DeLi",
            "automatic_judgment": None,
            "judgment_note": (
                "Answers are intentionally left for direct inspection; no heuristic "
                "entity parser is treated as human verification."
            ),
            "rows": rows,
        },
    )
    print(f"[done] comparison summary={args.summary_output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)

    def add_server_args(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--base-url", required=True)
        subparser.add_argument("--model", default="Qwen3")
        subparser.add_argument("--api-key", default="EMPTY")
        subparser.add_argument("--kv-dir", type=Path, required=True)

    collect_parser = subparsers.add_parser("collect")
    add_server_args(collect_parser)

    decode_parser = subparsers.add_parser("decode")
    add_server_args(decode_parser)
    decode_parser.add_argument("--case", choices=tuple(CASES_BY_NAME), required=True)
    decode_parser.add_argument("--mode", choices=MODES, required=True)
    decode_parser.add_argument("--output", type=Path, required=True)
    decode_parser.add_argument("--max-tokens", type=int, default=512)
    decode_parser.add_argument("--temperature", type=float, default=0.6)
    decode_parser.add_argument("--top-p", type=float, default=0.95)
    decode_parser.add_argument("--top-k", type=int, default=20)
    decode_parser.add_argument("--min-p", type=float, default=0.0)
    decode_parser.add_argument("--seed", type=int, default=1111)

    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--run-dir", type=Path, required=True)
    summary_parser.add_argument("--summary-output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.action == "collect":
        collect(args)
    elif args.action == "decode":
        decode(args)
    else:
        summarize(args)


if __name__ == "__main__":
    main()
