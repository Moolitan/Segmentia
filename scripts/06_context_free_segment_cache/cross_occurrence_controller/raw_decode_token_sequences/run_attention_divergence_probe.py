"""Capture attention map at the first thinking token position.

For each case, captures attention at one query position:
  - prompt_tokens + 1: the query that generates the first thinking content
    token (gen_idx 2, after <think>\\n). At this position, recompute and
    rope have seen exactly the same input (the prompt only), so any
    attention difference is purely from the KV cache mechanism.

Key regions observed (keys the query attends to):
  - pre_skill_context: 512 tokens before skill injection
  - skill_span: all skill tokens

Usage:
  python run_attention_divergence_probe.py \\
    --task internal_comms_incident_update \\
    --mode recompute --run-name recompute \\
    --probe-marker .../current_attention_probe.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parents[1]
MODULE_DIR = PACKAGE_DIR / "module"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from config import (  # noqa: E402
    DEFAULT_KV_DIR,
    DEFAULT_SERVED_MODEL,
    DEFAULT_TASKS,
    DEFAULT_VLLM_PORT,
)
from replay import context_config_for_case, selected_cases  # noqa: E402
from trace_utils import convert_messages, load_invocations, load_system_prompt, load_tools  # noqa: E402
from vllm_client import chat_completion, tokenize_chat  # noqa: E402

OCCURRENCES = (3,)
PRE_SKILL_WINDOW = 512

# Query positions to probe, expressed as token offsets after the first
# thinking-content query (offset 0 = prompt_tokens + 1). A single decode pass
# is enough: the vLLM probe hook honours a full ``query_positions`` list and
# dumps each position as its own set of rows (keyed by query_abs_position).
DEFAULT_QUERY_OFFSETS = (0, 30, 60, 90, 120)


def case_label(case: dict[str, Any]) -> str:
    def _safe(v: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", v).strip("_")
    return (
        f"inv{int(case['invocation_index']):03d}--"
        f"{_safe(str(case['task']))}--"
        f"{_safe(str(case['skill']))}--"
        f"occ{int(case['occurrence'])}"
    )


def span(start: int, end: int) -> dict[str, int]:
    s = max(0, int(start))
    e = max(s, int(end))
    return {"start": s, "end": e, "length": e - s}


def build_probe_marker(
    *,
    case: dict[str, Any],
    mode: str,
    run_name: str,
    prompt_tokens: int,
    target_start: int,
    target_end: int,
    request_id: str,
    query_offsets: list[int],
) -> dict[str, Any]:
    # gen_idx 0 = <think>, gen_idx 1 = \n, gen_idx 2 = first content word.
    # Query that generates gen_idx i (i>=1) is at prompt_tokens + i - 1.
    think_start_query = prompt_tokens + 1

    # One absolute query position per requested offset; offset 0 reproduces the
    # original first-thinking-token probe and anchors the offset split.
    query_positions = [think_start_query + int(off) for off in query_offsets]

    pre_skill_start = max(0, target_start - PRE_SKILL_WINDOW)

    regions = [
        {"region": "pre_skill_context", **span(pre_skill_start, target_start)},
        {"region": "skill_span", **span(target_start, target_end)},
    ]
    local_windows = {
        "pre_skill_context": span(pre_skill_start, target_start),
        "skill_span": span(target_start, target_end),
    }

    return {
        "enabled": True,
        "marker_version": 2,
        "case_id": f"{case['task']}/{case['skill']}/occ{case['occurrence']}",
        "mode": mode,
        "run_name": run_name,
        "task": str(case["task"]),
        "skill": str(case["skill"]),
        "occurrence": int(case["occurrence"]),
        "invocation_index": int(case["invocation_index"]),
        "request_id": request_id,
        "query_positions": query_positions,
        "query": {
            "decision_query_abs_position": think_start_query,
            "prompt_tokens": prompt_tokens,
        },
        "query_offsets": [int(off) for off in query_offsets],
        "base_query_position": think_start_query,
        "regions": regions,
        "local_windows": local_windows,
    }


def append_manifest(path: Path, marker: dict[str, Any]) -> None:
    """Record base position + offset→abs-position map so dumps can later be
    split into per-offset directories without guessing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "request_id": marker["request_id"],
        "case_id": marker["case_id"],
        "mode": marker["mode"],
        "prompt_tokens": marker["query"]["prompt_tokens"],
        "base_query_position": marker["base_query_position"],
        "query_offsets": marker["query_offsets"],
        "offset_to_position": {
            str(off): marker["base_query_position"] + off
            for off in marker["query_offsets"]
        },
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_probe_marker(path: Path, marker: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(marker, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def clear_probe_marker(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=DEFAULT_TASKS, required=True)
    parser.add_argument("--mode", choices=["recompute", "rope"], required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--probe-marker", type=Path, required=True,
                        help="Path to current_attention_probe.json (read by vLLM)")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--vllm-port", type=int, default=DEFAULT_VLLM_PORT)
    parser.add_argument("--model", default=os.environ.get("VLLM_SERVED_NAME", DEFAULT_SERVED_MODEL))
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--kv-dir", default=str(DEFAULT_KV_DIR))
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--min-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--query-offsets", default=",".join(str(o) for o in DEFAULT_QUERY_OFFSETS),
        help="Comma-separated token offsets after the first thinking query "
             "(0 = first thinking token). Each becomes its own probed position.",
    )
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="Optional JSONL manifest to append per-case query-position maps "
             "(used to split combined dumps into per-offset directories).",
    )
    args = parser.parse_args()

    query_offsets = [int(o) for o in str(args.query_offsets).split(",") if o.strip() != ""]
    if not query_offsets:
        raise ValueError("--query-offsets produced an empty list")

    cases = selected_cases([args.task], list(OCCURRENCES), include_first_occurrence=True)
    if not cases:
        raise ValueError(f"No cases for task={args.task}")

    system_prompt = load_system_prompt()
    tools = load_tools()
    invocations = load_invocations(args.task)
    base_url = args.base_url or f"http://127.0.0.1:{args.vllm_port}"

    clear_probe_marker(args.probe_marker)

    for case in cases:
        inv_idx = int(case["invocation_index"])
        invocation = invocations[inv_idx - 1]
        messages, _ = convert_messages(invocation["messages"], system_prompt)

        label = case_label(case)

        prompt_tokens = len(tokenize_chat(
            base_url, args.model, messages, tools, args.api_key,
            add_generation_prompt=True,
        ))

        target_start = int(case["target_start"])
        target_end = int(case["target_end"])

        request_id = (
            f"cf-attn-div-{args.run_name}-{case['task']}-{case['skill']}"
            f"-occ{case['occurrence']}-inv{inv_idx}"
        )

        marker = build_probe_marker(
            case=case,
            mode=args.mode,
            run_name=args.run_name,
            prompt_tokens=prompt_tokens,
            target_start=target_start,
            target_end=target_end,
            request_id=request_id,
            query_offsets=query_offsets,
        )
        write_probe_marker(args.probe_marker, marker)
        if args.manifest is not None:
            append_manifest(args.manifest, marker)

        cfg = context_config_for_case(args.mode, case, dump_kv_for_cksim=False)
        try:
            response, elapsed = chat_completion(
                base_url, args.model, messages, tools, args.api_key,
                max_tokens=args.max_tokens,
                request_id=request_id,
                context_segment_cache=cfg,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                seed=args.seed,
                logprobs=False,
            )
            print(
                f"[ok] run={args.run_name} mode={args.mode} "
                f"{case['task']}/{case['skill']} occ{case['occurrence']} "
                f"inv{inv_idx} prompt={prompt_tokens} "
                f"query_abs={marker['query_positions']} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
        except RuntimeError as exc:
            print(
                f"[error] run={args.run_name} {case['task']}/{case['skill']} "
                f"occ{case['occurrence']}: {exc}",
                flush=True,
            )
        finally:
            clear_probe_marker(args.probe_marker)


if __name__ == "__main__":
    main()
