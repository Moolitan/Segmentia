"""Collect free-generation thinking/action boundary diagnostics.

This script replays the same configured Segmentia headline cases as
run_decode_compare.py, but only for recompute vs rope. It saves complete
reasoning/action outputs plus token-level logprobs so offline analysis can
study:

  thinking semantics -> action boundary margin -> action divergence

It does not start vLLM. Use run_thinking_action_diagnostic.sh for the restart
wrapper.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import (  # noqa: E402
    DEFAULT_KV_DIR,
    DEFAULT_SERVED_MODEL,
    DEFAULT_VLLM_PORT,
    RESULTS_DIR,
    parse_tasks,
)
from run_decode_compare import (  # noqa: E402
    context_config_for_case,
    resolve_cache_id,
    selected_cases,
)
from run_margin_diagnostic import action_label, logprob_entries  # noqa: E402
from trace_utils import convert_messages, load_invocations, load_system_prompt, load_tools  # noqa: E402
from vllm_client import chat_completion, extract_response  # noqa: E402

DEFAULT_OUT_DIR = RESULTS_DIR / "thinking_to_action_divergence"
DEFAULT_FREE_JSONL = DEFAULT_OUT_DIR / "data" / "free_generation_rows.jsonl"
DEFAULT_TOKEN_JSONL = DEFAULT_OUT_DIR / "data" / "token_logprob_rows.jsonl"
DEFAULT_CASE_CSV = DEFAULT_OUT_DIR / "tables" / "thinking_action_case_summary.csv"

SUPPORTED_MODES = {"recompute", "rope"}
FUNCTION_NAMES = ("Write", "Edit", "Read", "Bash")


def case_key_from_case(case: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(case["task"]),
        str(case["skill"]),
        int(case["occurrence"]),
        int(case["invocation_index"]),
    )


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def existing_completed(path: Path) -> set[tuple[str, str, int, int, str]]:
    done: set[tuple[str, str, int, int, str]] = set()
    for row in load_jsonl(path):
        if row.get("error"):
            continue
        done.add(
            (
                str(row["task"]),
                str(row["skill"]),
                int(row["occurrence"]),
                int(row["invocation_index"]),
                str(row["mode"]),
            )
        )
    return done


def token_spans(entries: list[dict[str, Any]]) -> tuple[str, list[tuple[int, int, int]]]:
    text_parts: list[str] = []
    spans: list[tuple[int, int, int]] = []
    cursor = 0
    for entry in entries:
        token = str(entry.get("generated_token") or "")
        idx = int(entry["token_index"])
        start = cursor
        cursor += len(token)
        spans.append((idx, start, cursor))
        text_parts.append(token)
    return "".join(text_parts), spans


def token_index_at_char(spans: list[tuple[int, int, int]], char_idx: int | None) -> int | None:
    if char_idx is None:
        return None
    for idx, start, end in spans:
        if start <= char_idx < end:
            return idx
        if start == end == char_idx:
            return idx
    return None


def first_nonspace_char(text: str, start: int | None) -> int | None:
    if start is None:
        return None
    for pos in range(start, len(text)):
        if not text[pos].isspace():
            return pos
    return None


def first_function_after(text: str, start: int | None) -> tuple[str | None, int | None]:
    if start is None:
        return None, None
    best_name: str | None = None
    best_pos: int | None = None
    for name in FUNCTION_NAMES:
        pos = text.find(name, start)
        if pos >= 0 and (best_pos is None or pos < best_pos):
            best_name = name
            best_pos = pos
    return best_name, best_pos


def row_at_index(entries: list[dict[str, Any]], idx: int | None) -> dict[str, Any] | None:
    if idx is None:
        return None
    for entry in entries:
        if int(entry["token_index"]) == idx:
            return entry
    return None


def margin_at(entries: list[dict[str, Any]], idx: int | None) -> float | None:
    row = row_at_index(entries, idx)
    if row is None or row.get("margin") is None:
        return None
    return float(row["margin"])


def top1_at(entries: list[dict[str, Any]], idx: int | None) -> str | None:
    row = row_at_index(entries, idx)
    if row is None:
        return None
    token = row.get("top1_token")
    return str(token) if token is not None else None


def generated_at(entries: list[dict[str, Any]], idx: int | None) -> str | None:
    row = row_at_index(entries, idx)
    if row is None:
        return None
    token = row.get("generated_token")
    return str(token) if token is not None else None


def structured_function_name(tool_calls: list[dict[str, Any]]) -> str | None:
    if not tool_calls:
        return None
    fn = (tool_calls[0] or {}).get("function") or {}
    name = fn.get("name")
    return str(name) if name else None


def locate_boundaries(
    entries: list[dict[str, Any]],
    tool_calls: list[dict[str, Any]],
    *,
    error: str | None,
) -> dict[str, Any]:
    if error:
        return {
            "think_end_found": False,
            "think_end_token_index": None,
            "visible_start_token_index": None,
            "tool_call_start_token_index": None,
            "function_name_token_index": None,
            "function_name": structured_function_name(tool_calls),
            "action_boundary_type": None,
            "action_boundary_token_index": None,
            "action_boundary_margin": None,
            "action_boundary_top1": None,
            "action_boundary_generated_token": None,
            "boundary_status": "error",
        }
    if not entries:
        return {
            "think_end_found": False,
            "think_end_token_index": None,
            "visible_start_token_index": None,
            "tool_call_start_token_index": None,
            "function_name_token_index": None,
            "function_name": structured_function_name(tool_calls),
            "action_boundary_type": None,
            "action_boundary_token_index": None,
            "action_boundary_margin": None,
            "action_boundary_top1": None,
            "action_boundary_generated_token": None,
            "boundary_status": "no_logprobs",
        }

    text, spans = token_spans(entries)
    think_end_char = text.find("</think>")
    think_end_found = think_end_char >= 0
    think_end_idx = token_index_at_char(spans, think_end_char) if think_end_found else None
    post_think_char = think_end_char + len("</think>") if think_end_found else None
    visible_char = first_nonspace_char(text, post_think_char)
    visible_idx = token_index_at_char(spans, visible_char)

    tool_char = text.find("<tool_call>", post_think_char or 0)
    if tool_char < 0:
        tool_char = None
    tool_idx = token_index_at_char(spans, tool_char)

    function_name, function_char = first_function_after(text, post_think_char)
    function_idx = token_index_at_char(spans, function_char)
    structured_name = structured_function_name(tool_calls)
    if function_name is None:
        function_name = structured_name

    if not think_end_found:
        boundary_type = None
        boundary_idx = None
        status = "no_think_end"
    elif function_idx is not None:
        boundary_type = "function_name"
        boundary_idx = function_idx
        status = "token_boundary"
    elif tool_idx is not None:
        boundary_type = "tool_call_start"
        boundary_idx = tool_idx
        status = "token_boundary"
    elif tool_calls:
        boundary_type = None
        boundary_idx = None
        status = "structured_tool_only"
    elif visible_idx is not None:
        boundary_type = "visible_start"
        boundary_idx = visible_idx
        status = "text_boundary"
    else:
        boundary_type = None
        boundary_idx = None
        status = "no_visible_boundary"

    return {
        "think_end_found": think_end_found,
        "think_end_token_index": think_end_idx,
        "visible_start_token_index": visible_idx,
        "tool_call_start_token_index": tool_idx,
        "function_name_token_index": function_idx,
        "function_name": function_name,
        "action_boundary_type": boundary_type,
        "action_boundary_token_index": boundary_idx,
        "action_boundary_margin": margin_at(entries, boundary_idx),
        "action_boundary_top1": top1_at(entries, boundary_idx),
        "action_boundary_generated_token": generated_at(entries, boundary_idx),
        "boundary_status": status,
    }


def boundary_type_for_token(boundaries: dict[str, Any], token_index: int) -> str:
    labels = [
        ("think_end", boundaries.get("think_end_token_index")),
        ("visible_start", boundaries.get("visible_start_token_index")),
        ("tool_call_start", boundaries.get("tool_call_start_token_index")),
        ("function_name", boundaries.get("function_name_token_index")),
    ]
    matched = [label for label, idx in labels if idx is not None and int(idx) == token_index]
    return "+".join(matched) if matched else "none"


def region_for_token(boundaries: dict[str, Any], token_index: int) -> str:
    think_end = boundaries.get("think_end_token_index")
    if think_end is None:
        return "thinking" if not boundaries.get("think_end_found") else "unknown"
    return "thinking" if token_index <= int(think_end) else "post_think"


def make_token_rows(
    case: dict[str, Any],
    mode: str,
    entries: list[dict[str, Any]],
    boundaries: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for entry in entries:
        idx = int(entry["token_index"])
        rows.append(
            {
                **case,
                "mode": mode,
                **entry,
                "region": region_for_token(boundaries, idx),
                "boundary_type": boundary_type_for_token(boundaries, idx),
            }
        )
    return rows


def write_case_summary(free_jsonl: Path, case_csv: Path) -> None:
    rows = load_jsonl(free_jsonl)
    fields = [
        "task",
        "skill",
        "occurrence",
        "invocation_index",
        "mode",
        "action_label",
        "finish_reason",
        "completion_tokens",
        "reasoning_chars",
        "content_chars",
        "tool_call_count",
        "think_end_found",
        "think_end_token_index",
        "visible_start_token_index",
        "tool_call_start_token_index",
        "function_name_token_index",
        "function_name",
        "action_boundary_type",
        "action_boundary_token_index",
        "action_boundary_margin",
        "boundary_status",
        "error",
    ]
    case_csv.parent.mkdir(parents=True, exist_ok=True)
    with case_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            usage = row.get("usage") or {}
            writer.writerow(
                {
                    "task": row.get("task"),
                    "skill": row.get("skill"),
                    "occurrence": row.get("occurrence"),
                    "invocation_index": row.get("invocation_index"),
                    "mode": row.get("mode"),
                    "action_label": row.get("action_label"),
                    "finish_reason": row.get("finish_reason"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "reasoning_chars": len(row.get("reasoning") or ""),
                    "content_chars": len(row.get("content") or ""),
                    "tool_call_count": len(row.get("tool_calls") or []),
                    "think_end_found": row.get("think_end_found"),
                    "think_end_token_index": row.get("think_end_token_index"),
                    "visible_start_token_index": row.get("visible_start_token_index"),
                    "tool_call_start_token_index": row.get("tool_call_start_token_index"),
                    "function_name_token_index": row.get("function_name_token_index"),
                    "function_name": row.get("function_name"),
                    "action_boundary_type": row.get("action_boundary_type"),
                    "action_boundary_token_index": row.get("action_boundary_token_index"),
                    "action_boundary_margin": row.get("action_boundary_margin"),
                    "boundary_status": row.get("boundary_status"),
                    "error": row.get("error"),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--modes", default="recompute,rope")
    parser.add_argument("--occurrences", default="2,3")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--vllm-port", type=int, default=DEFAULT_VLLM_PORT)
    parser.add_argument("--model", default=os.environ.get("VLLM_SERVED_NAME", DEFAULT_SERVED_MODEL))
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--kv-dir", default=str(DEFAULT_KV_DIR))
    parser.add_argument("--free-jsonl", default=str(DEFAULT_FREE_JSONL))
    parser.add_argument("--token-jsonl", default=str(DEFAULT_TOKEN_JSONL))
    parser.add_argument("--case-csv", default=str(DEFAULT_CASE_CSV))
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--top-logprobs", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    invalid = sorted(set(modes) - SUPPORTED_MODES)
    if invalid:
        raise ValueError(f"Unsupported modes for this diagnostic: {invalid}")

    tasks = parse_tasks(args.tasks)
    occurrences = [int(x) for x in args.occurrences.split(",") if x.strip()]
    base_url = args.base_url or f"http://127.0.0.1:{args.vllm_port}"
    free_jsonl = Path(args.free_jsonl)
    token_jsonl = Path(args.token_jsonl)
    case_csv = Path(args.case_csv)
    if not args.append:
        for path in (free_jsonl, token_jsonl, case_csv):
            if path.exists():
                path.unlink()

    completed = existing_completed(free_jsonl) if args.skip_existing else set()
    system_prompt = load_system_prompt()
    tools = load_tools()
    cases = selected_cases(tasks, occurrences)

    for case in cases:
        invocation = load_invocations(case["task"])[case["invocation_index"] - 1]
        messages, _ = convert_messages(invocation["messages"], system_prompt)
        for mode in modes:
            key = (*case_key_from_case(case), mode)
            if key in completed:
                print(
                    f"[skip-existing] {mode:9s} {case['task']} {case['skill']} "
                    f"occ{case['occurrence']}",
                    flush=True,
                )
                continue
            cfg = context_config_for_case(mode, case, dump_kv_for_cksim=False)
            request_id = (
                f"cf-think-action-{mode}-{case['task']}-{case['skill']}"
                f"-occ{case['occurrence']}"
            )
            try:
                response, elapsed = chat_completion(
                    base_url,
                    args.model,
                    messages,
                    tools,
                    args.api_key,
                    max_tokens=args.max_tokens,
                    request_id=request_id,
                    context_segment_cache=cfg,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    logprobs=True,
                    top_logprobs=args.top_logprobs,
                )
                parts = extract_response(response)
                entries = logprob_entries(response)
                error = None
            except RuntimeError as exc:
                response = {}
                elapsed = 0.0
                parts = {
                    "text": "",
                    "content": "",
                    "reasoning": "",
                    "tool_calls": [],
                    "finish_reason": None,
                }
                entries = []
                error = str(exc)

            boundaries = locate_boundaries(entries, parts["tool_calls"], error=error)
            usage = response.get("usage", {})
            free_row = {
                **case,
                "mode": mode,
                "temperature": args.temperature,
                "cache_id": resolve_cache_id(mode, case),
                "model": args.model,
                "max_tokens": args.max_tokens,
                "top_logprobs_requested": args.top_logprobs,
                "latency_s": round(elapsed, 4),
                "usage": usage,
                "error": error,
                "text": parts["text"],
                "content": parts["content"],
                "reasoning": parts["reasoning"],
                "tool_calls": parts["tool_calls"],
                "finish_reason": parts["finish_reason"],
                "action_label": action_label(
                    {"error": error, "tool_calls": parts["tool_calls"]}
                ),
                **boundaries,
            }
            append_jsonl(free_jsonl, [free_row])
            append_jsonl(token_jsonl, make_token_rows(case, mode, entries, boundaries))
            status = "error" if error else "ok"
            print(
                f"[{status}] {mode:9s} {case['task']} {case['skill']} "
                f"occ{case['occurrence']} logprob_tokens={len(entries)} "
                f"boundary={boundaries['boundary_status']}",
                flush=True,
            )

    write_case_summary(free_jsonl, case_csv)
    metadata = {
        "model": args.model,
        "base_url": base_url,
        "tasks": tasks,
        "modes": modes,
        "occurrences": occurrences,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "top_logprobs": args.top_logprobs,
        "free_jsonl": str(free_jsonl),
        "token_jsonl": str(token_jsonl),
        "case_csv": str(case_csv),
        "offline_kv_dir_expected_by_server": str(Path(args.kv_dir)),
    }
    meta_path = free_jsonl.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] free rows: {free_jsonl}")
    print(f"[done] token rows: {token_jsonl}")
    print(f"[done] case summary: {case_csv}")
    print(f"[done] metadata: {meta_path}")


if __name__ == "__main__":
    main()
