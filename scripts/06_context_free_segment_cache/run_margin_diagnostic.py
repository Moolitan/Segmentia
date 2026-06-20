"""Collect token logprob margins for Segmentia action-divergence diagnosis.

P0 question:
  Do reuse divergences concentrate at low-margin action decision points?

This script replays the same configured cases as run_decode_compare.py with
logprobs enabled, then writes:
  - token-level top-k logprob rows
  - case/mode-level margin summaries

It does not start vLLM. Use run_margin_diagnostic.sh for the restart wrapper.
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
from trace_utils import convert_messages, load_invocations, load_system_prompt, load_tools  # noqa: E402
from vllm_client import chat_completion, extract_response  # noqa: E402

DEFAULT_HEADLINE_JSONL = (
    RESULTS_DIR / "headline_semantic_action_gap" / "data" / "decode_outputs.jsonl"
)
DEFAULT_OUT_DIR = RESULTS_DIR / "logprob_margin_diagnostic"
DEFAULT_TOKEN_JSONL = DEFAULT_OUT_DIR / "data" / "logprob_margin_rows.jsonl"
DEFAULT_CASE_CSV = DEFAULT_OUT_DIR / "tables" / "margin_case_summary.csv"

STRUCTURAL_MARKERS = (
    "tool",
    "call",
    "function",
    "read",
    "write",
    "edit",
    "text",
    "assistant",
    "<|",
    "{",
    "}",
    '"name"',
)


def action_label(row: dict[str, Any] | None) -> str:
    if row is None:
        return "missing"
    if row.get("error"):
        return "error"
    names: list[str] = []
    for tc in row.get("tool_calls") or []:
        fn = (tc or {}).get("function") or {}
        name = fn.get("name")
        if name:
            names.append(str(name))
    if names:
        return "tool:" + "+".join(names)
    return "text"


def case_key_from_row(row: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(row["task"]),
        str(row["skill"]),
        int(row["occurrence"]),
        int(row["invocation_index"]),
    )


def case_key_from_case(case: dict[str, Any]) -> tuple[str, str, int, int]:
    return (
        str(case["task"]),
        str(case["skill"]),
        int(case["occurrence"]),
        int(case["invocation_index"]),
    )


def load_headline_actions(path: Path) -> dict[tuple[str, str, int, int], dict[str, str]]:
    actions: dict[tuple[str, str, int, int], dict[str, str]] = {}
    if not path.exists():
        return actions
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get("sample_index", 0)) != 0:
                continue
            key = case_key_from_row(row)
            actions.setdefault(key, {})[str(row["mode"])] = action_label(row)
    return actions


def normalize_top_logprobs(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        return [
            {"token": str(tok), "logprob": float(lp), "bytes": None}
            for tok, lp in raw.items()
        ]
    rows: list[dict[str, Any]] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                rows.append(
                    {
                        "token": str(item.get("token", "")),
                        "logprob": (
                            float(item["logprob"])
                            if item.get("logprob") is not None
                            else None
                        ),
                        "bytes": item.get("bytes"),
                    }
                )
    return rows


def logprob_entries(response: dict[str, Any]) -> list[dict[str, Any]]:
    """Return OpenAI/vLLM chat logprob entries in a stable shape."""
    choice = (response.get("choices") or [{}])[0]
    logprobs = choice.get("logprobs") or {}
    content = logprobs.get("content")
    if content is None:
        # Some OpenAI-compatible endpoints return completion-style arrays.
        tokens = logprobs.get("tokens") or []
        token_lps = logprobs.get("token_logprobs") or []
        top_lps = logprobs.get("top_logprobs") or []
        content = []
        for i, token in enumerate(tokens):
            content.append(
                {
                    "token": token,
                    "logprob": token_lps[i] if i < len(token_lps) else None,
                    "top_logprobs": top_lps[i] if i < len(top_lps) else None,
                }
            )
    if not isinstance(content, list):
        return []

    rows: list[dict[str, Any]] = []
    for idx, entry in enumerate(content):
        if not isinstance(entry, dict):
            continue
        token = str(entry.get("token", ""))
        lp = entry.get("logprob")
        generated = {
            "token": token,
            "logprob": float(lp) if lp is not None else None,
            "bytes": entry.get("bytes"),
        }
        candidates = normalize_top_logprobs(entry.get("top_logprobs"))
        if generated["logprob"] is not None and all(
            c["token"] != generated["token"] for c in candidates
        ):
            candidates.append(generated)
        candidates = [
            c for c in candidates if c.get("logprob") is not None
        ]
        candidates.sort(key=lambda c: float(c["logprob"]), reverse=True)
        top1 = candidates[0] if len(candidates) >= 1 else None
        top2 = candidates[1] if len(candidates) >= 2 else None
        margin = (
            float(top1["logprob"]) - float(top2["logprob"])
            if top1 is not None and top2 is not None
            else None
        )
        token_lower = token.lower()
        rows.append(
            {
                "token_index": idx,
                "generated_token": token,
                "generated_logprob": generated["logprob"],
                "top1_token": top1["token"] if top1 else None,
                "top1_logprob": top1["logprob"] if top1 else None,
                "top2_token": top2["token"] if top2 else None,
                "top2_logprob": top2["logprob"] if top2 else None,
                "margin": margin,
                "is_structural_heuristic": any(
                    marker in token_lower for marker in STRUCTURAL_MARKERS
                ),
                "top_logprobs": candidates,
            }
        )
    return rows


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_token_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def existing_case_modes(path: Path) -> set[tuple[str, str, int, int, str]]:
    existing = set()
    for row in load_token_rows(path):
        existing.add(
            (
                row["task"],
                row["skill"],
                int(row["occurrence"]),
                int(row["invocation_index"]),
                row["mode"],
            )
        )
    return existing


def min_margin_row(rows: list[dict[str, Any]], *, window: int | None = None):
    candidates = [
        r
        for r in rows
        if r.get("margin") is not None
        and (window is None or int(r["token_index"]) < window)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda r: float(r["margin"]))


def mean_margin(rows: list[dict[str, Any]], *, window: int | None = None):
    vals = [
        float(r["margin"])
        for r in rows
        if r.get("margin") is not None
        and (window is None or int(r["token_index"]) < window)
    ]
    return sum(vals) / len(vals) if vals else None


def write_case_summary(
    token_jsonl: Path,
    case_csv: Path,
    headline_actions: dict[tuple[str, str, int, int], dict[str, str]],
    *,
    decision_window: int,
) -> None:
    grouped: dict[tuple[str, str, int, int, str], list[dict[str, Any]]] = {}
    for row in load_token_rows(token_jsonl):
        key = (
            row["task"],
            row["skill"],
            int(row["occurrence"]),
            int(row["invocation_index"]),
            row["mode"],
        )
        grouped.setdefault(key, []).append(row)

    out_rows: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        task, skill, occurrence, invocation_index, mode = key
        rows = [r for r in rows if r.get("token_index") is not None]
        if not rows:
            continue
        case_key = (task, skill, occurrence, invocation_index)
        rows.sort(key=lambda r: int(r["token_index"]))
        actions = headline_actions.get(case_key, {})
        recompute_action = actions.get("recompute", "missing")
        mode_action = actions.get(mode, "missing")
        diverged = (
            None
            if "missing" in (recompute_action, mode_action)
            else mode_action != recompute_action
        )

        min_all = min_margin_row(rows)
        min_window = min_margin_row(rows, window=decision_window)
        structural = [r for r in rows if r.get("is_structural_heuristic")]
        min_structural = min_margin_row(structural, window=decision_window)

        recompute_rows = grouped.get((task, skill, occurrence, invocation_index, "recompute"), [])
        recompute_min = min_margin_row(recompute_rows, window=decision_window)
        idx = int(recompute_min["token_index"]) if recompute_min else None
        this_at_idx = next(
            (r for r in rows if idx is not None and int(r["token_index"]) == idx),
            None,
        )

        out_rows.append(
            {
                "task": task,
                "skill": skill,
                "occurrence": occurrence,
                "invocation_index": invocation_index,
                "mode": mode,
                "recompute_action": recompute_action,
                "mode_action": mode_action,
                "diverged_from_recompute": diverged,
                "tokens_with_logprobs": len(rows),
                "decision_window": decision_window,
                "min_margin": min_all.get("margin") if min_all else None,
                "min_margin_token_index": min_all.get("token_index") if min_all else None,
                "min_margin_generated_token": min_all.get("generated_token") if min_all else None,
                "min_margin_top1": min_all.get("top1_token") if min_all else None,
                "min_margin_top2": min_all.get("top2_token") if min_all else None,
                "mean_margin_decision_window": mean_margin(
                    rows, window=decision_window
                ),
                "min_margin_decision_window": (
                    min_window.get("margin") if min_window else None
                ),
                "min_margin_decision_token_index": (
                    min_window.get("token_index") if min_window else None
                ),
                "min_margin_decision_generated_token": (
                    min_window.get("generated_token") if min_window else None
                ),
                "min_structural_margin_decision_window": (
                    min_structural.get("margin") if min_structural else None
                ),
                "min_structural_token_index": (
                    min_structural.get("token_index") if min_structural else None
                ),
                "min_structural_generated_token": (
                    min_structural.get("generated_token") if min_structural else None
                ),
                "recompute_min_margin_decision_window": (
                    recompute_min.get("margin") if recompute_min else None
                ),
                "recompute_min_margin_token_index": idx,
                "top1_changed_at_recompute_min_index": (
                    None
                    if recompute_min is None or this_at_idx is None
                    else this_at_idx.get("top1_token") != recompute_min.get("top1_token")
                ),
                "generated_changed_at_recompute_min_index": (
                    None
                    if recompute_min is None or this_at_idx is None
                    else this_at_idx.get("generated_token")
                    != recompute_min.get("generated_token")
                ),
                "top1_at_recompute_min_index": (
                    this_at_idx.get("top1_token") if this_at_idx else None
                ),
                "generated_at_recompute_min_index": (
                    this_at_idx.get("generated_token") if this_at_idx else None
                ),
            }
        )

    case_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(out_rows[0].keys()) if out_rows else [
        "task",
        "skill",
        "occurrence",
        "invocation_index",
        "mode",
    ]
    with case_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--modes", default="recompute,direct,rope")
    parser.add_argument("--occurrences", default="2,3")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--vllm-port", type=int, default=DEFAULT_VLLM_PORT)
    parser.add_argument("--model", default=os.environ.get("VLLM_SERVED_NAME", DEFAULT_SERVED_MODEL))
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--kv-dir", default=str(DEFAULT_KV_DIR))
    parser.add_argument("--headline-jsonl", default=str(DEFAULT_HEADLINE_JSONL))
    parser.add_argument("--token-jsonl", default=str(DEFAULT_TOKEN_JSONL))
    parser.add_argument("--case-csv", default=str(DEFAULT_CASE_CSV))
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--top-logprobs", type=int, default=10)
    parser.add_argument("--decision-window", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    tasks = parse_tasks(args.tasks)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    occurrences = [int(x) for x in args.occurrences.split(",") if x.strip()]
    base_url = args.base_url or f"http://127.0.0.1:{args.vllm_port}"
    token_jsonl = Path(args.token_jsonl)
    case_csv = Path(args.case_csv)
    if token_jsonl.exists() and not args.append:
        token_jsonl.unlink()

    headline_actions = load_headline_actions(Path(args.headline_jsonl))
    existing = existing_case_modes(token_jsonl) if args.skip_existing else set()

    system_prompt = load_system_prompt()
    tools = load_tools()
    cases = selected_cases(tasks, occurrences)

    for case in cases:
        invocation = load_invocations(case["task"])[case["invocation_index"] - 1]
        messages, _ = convert_messages(invocation["messages"], system_prompt)
        for mode in modes:
            key = (*case_key_from_case(case), mode)
            if key in existing:
                print(
                    f"[skip-existing] {mode:9s} {case['task']} {case['skill']} "
                    f"occ{case['occurrence']}",
                    flush=True,
                )
                continue
            cfg = context_config_for_case(mode, case, dump_kv_for_cksim=False)
            request_id = (
                f"cf-margin-{mode}-{case['task']}-{case['skill']}"
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

            case_key = case_key_from_case(case)
            actions = headline_actions.get(case_key, {})
            rows = []
            for entry in entries:
                rows.append(
                    {
                        **case,
                        "mode": mode,
                        "temperature": args.temperature,
                        "cache_id": resolve_cache_id(mode, case),
                        "model": args.model,
                        "max_tokens": args.max_tokens,
                        "top_logprobs_requested": args.top_logprobs,
                        "latency_s": round(elapsed, 4),
                        "error": error,
                        "finish_reason": parts["finish_reason"],
                        "headline_recompute_action": actions.get("recompute", "missing"),
                        "headline_mode_action": actions.get(mode, "missing"),
                        "headline_diverged_from_recompute": (
                            None
                            if actions.get("recompute") is None
                            or actions.get(mode) is None
                            else actions.get(mode) != actions.get("recompute")
                        ),
                        **entry,
                    }
                )
            if not rows:
                rows.append(
                    {
                        **case,
                        "mode": mode,
                        "temperature": args.temperature,
                        "cache_id": resolve_cache_id(mode, case),
                        "model": args.model,
                        "max_tokens": args.max_tokens,
                        "top_logprobs_requested": args.top_logprobs,
                        "latency_s": round(elapsed, 4),
                        "error": error or "no logprob entries returned",
                        "finish_reason": parts["finish_reason"],
                        "headline_recompute_action": actions.get("recompute", "missing"),
                        "headline_mode_action": actions.get(mode, "missing"),
                        "headline_diverged_from_recompute": None,
                        "token_index": None,
                        "generated_token": None,
                        "generated_logprob": None,
                        "top1_token": None,
                        "top1_logprob": None,
                        "top2_token": None,
                        "top2_logprob": None,
                        "margin": None,
                        "is_structural_heuristic": None,
                        "top_logprobs": [],
                    }
                )
            append_jsonl(token_jsonl, rows)
            status = "error" if error else "ok"
            print(
                f"[{status}] {mode:9s} {case['task']} {case['skill']} "
                f"occ{case['occurrence']} logprob_tokens={len(entries)}",
                flush=True,
            )

    write_case_summary(
        token_jsonl,
        case_csv,
        headline_actions,
        decision_window=args.decision_window,
    )
    print(f"[done] token rows: {token_jsonl}")
    print(f"[done] case summary: {case_csv}")


if __name__ == "__main__":
    main()
