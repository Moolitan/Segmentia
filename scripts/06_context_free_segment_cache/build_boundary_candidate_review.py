"""Build B/C review and boundary-candidate scoring coverage tables.

This is an offline-only helper for the thinking-to-action diagnostic. It does
not call vLLM. It answers two narrow questions:

1. Which B/C cases should be manually reviewed for intent-parser false positives?
2. Can existing top-k boundary logprobs support candidate scoring, or do we need
   a later forced-scoring run for missing candidates?
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from config import RESULTS_DIR  # noqa: E402
from trace_utils import load_tools  # noqa: E402

DEFAULT_ROOT = RESULTS_DIR / "thinking_to_action_divergence"
DEFAULT_PAIR_CSV = DEFAULT_ROOT / "tables" / "thinking_pair_summary.csv"
DEFAULT_FREE_JSONL = DEFAULT_ROOT / "data" / "free_generation_rows.jsonl"
DEFAULT_TOKEN_JSONL = DEFAULT_ROOT / "data" / "token_logprob_rows.jsonl"
DEFAULT_REVIEW_CSV = DEFAULT_ROOT / "tables" / "bc_case_manual_review.csv"
DEFAULT_SCORING_CSV = DEFAULT_ROOT / "tables" / "boundary_candidate_scoring_plan.csv"

BC_CATEGORIES = {
    "B_thinking_similar_action_diverged",
    "C_thinking_different_action_diverged",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def case_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return (str(row["task"]), str(row["skill"]), int(row["occurrence"]))


def mode_key(row: dict[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(row["task"]),
        str(row["skill"]),
        int(row["occurrence"]),
        str(row["mode"]),
    )


def token_key(row: dict[str, Any]) -> tuple[str, str, int, str, int]:
    return (
        str(row["task"]),
        str(row["skill"]),
        int(row["occurrence"]),
        str(row["mode"]),
        int(row["token_index"]),
    )


def sort_key(row: dict[str, Any]) -> tuple[str, int, str, int]:
    return (
        str(row["task"]),
        int(row.get("invocation_index") or 0),
        str(row["skill"]),
        int(row["occurrence"]),
    )


def text_excerpt(text: str, limit: int = 700) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    head = compact[: limit // 2].rstrip()
    tail = compact[-limit // 2 :].lstrip()
    return f"{head} [...] {tail}"


def parse_action_tools(action_label: str) -> list[str]:
    label = (action_label or "").strip()
    if not label.startswith("tool:"):
        return []
    body = label[len("tool:") :]
    return [part.strip() for part in body.split("+") if part.strip()]


def first_action_tool(action_label: str) -> str | None:
    tools = parse_action_tools(action_label)
    return tools[0] if tools else None


def action_modality(action_label: str) -> str:
    return "tool" if (action_label or "").startswith("tool:") else "text"


def csv_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def candidate_rows_for_pair(
    pair: dict[str, str],
    free_rows: dict[tuple[str, str, int, str], dict[str, Any]],
    available_tools: list[str],
) -> list[dict[str, Any]]:
    recompute_action = pair["recompute_action"]
    rope_action = pair["rope_action"]
    rec_modality = action_modality(recompute_action)
    rope_modality = action_modality(rope_action)
    rec_first_tool = first_action_tool(recompute_action)
    rope_first_tool = first_action_tool(rope_action)

    observed_tools = [
        tool for tool in (rec_first_tool, rope_first_tool) if tool is not None
    ]
    rows: list[dict[str, Any]] = []

    if rec_modality != rope_modality:
        rows.append(
            {
                "candidate_group": "tool_text_boundary",
                "candidate": "<tool_call>",
                "candidate_kind": "special_token",
                "boundary_anchor": "tool_call_start",
                "candidate_source": "observed_modality_divergence",
                "observed_divergence_candidate": True,
            }
        )
        for mode in ("recompute", "rope"):
            free = free_rows.get((*case_key(pair), mode), {})
            if action_modality(str(free.get("action_label") or "")) == "text":
                generated = free.get("action_boundary_generated_token")
                candidate = str(generated) if generated is not None else "visible_text_start"
                rows.append(
                    {
                        "candidate_group": "tool_text_boundary",
                        "candidate": candidate,
                        "candidate_kind": "observed_text_token",
                        "boundary_anchor": "visible_start",
                        "candidate_source": f"observed_{mode}_text_start",
                        "observed_divergence_candidate": True,
                    }
                )

    if observed_tools:
        for tool in sorted(set(observed_tools)):
            rows.append(
                {
                    "candidate_group": "function_name_boundary",
                    "candidate": tool,
                    "candidate_kind": "tool_name",
                    "boundary_anchor": "function_name",
                    "candidate_source": "observed_first_tool_divergence",
                    "observed_divergence_candidate": True,
                }
            )

        for tool in available_tools:
            if tool in observed_tools:
                continue
            rows.append(
                {
                    "candidate_group": "function_name_boundary",
                    "candidate": tool,
                    "candidate_kind": "tool_name",
                    "boundary_anchor": "function_name",
                    "candidate_source": "available_tool_expansion",
                    "observed_divergence_candidate": False,
                }
            )

    # Preserve order while removing duplicates introduced by text/tool cases.
    seen: set[tuple[str, str, str]] = set()
    deduped = []
    for row in rows:
        key = (
            str(row["candidate_group"]),
            str(row["candidate"]),
            str(row["candidate_kind"]),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def normalize_token(token: Any) -> str:
    return str(token or "").strip()


def find_candidate_in_topk(
    top_logprobs: list[dict[str, Any]],
    candidate: str,
    candidate_kind: str,
) -> tuple[str, int | None, float | None, str | None]:
    normalized_candidate = normalize_token(candidate)
    for rank, item in enumerate(top_logprobs, start=1):
        token = str(item.get("token") or "")
        normalized_token = normalize_token(token)
        if candidate_kind == "special_token":
            matched = token == candidate or normalized_token == normalized_candidate
        elif candidate_kind == "tool_name":
            matched = normalized_token == normalized_candidate
        else:
            matched = token == candidate or normalized_token == normalized_candidate
        if matched:
            logprob = item.get("logprob")
            return (
                "in_topk",
                rank,
                float(logprob) if logprob is not None else None,
                token,
            )
    return ("missing_from_topk", None, None, None)


def build_review_rows(
    pairs: list[dict[str, str]],
    free_rows: dict[tuple[str, str, int, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    review_rows: list[dict[str, Any]] = []
    for pair in pairs:
        for mode in ("recompute", "rope"):
            free = free_rows.get((*case_key(pair), mode), {})
            review_rows.append(
                {
                    "task": pair["task"],
                    "skill": pair["skill"],
                    "occurrence": pair["occurrence"],
                    "invocation_index": pair["invocation_index"],
                    "category": pair["category"],
                    "mode": mode,
                    "action_label": free.get("action_label"),
                    "function_name": free.get("function_name"),
                    "boundary_status": free.get("boundary_status"),
                    "action_boundary_type": free.get("action_boundary_type"),
                    "action_boundary_token_index": free.get("action_boundary_token_index"),
                    "action_boundary_margin": free.get("action_boundary_margin"),
                    "action_boundary_top1": free.get("action_boundary_top1"),
                    "intent_label": pair.get(f"{mode}_intent"),
                    "mentioned_tools": pair.get(f"{mode}_tools"),
                    "mentioned_files": pair.get(f"{mode}_files"),
                    "embedding_cosine": pair.get("embedding_cosine"),
                    "grounding_conflict": pair.get("grounding_conflict"),
                    "reasoning_excerpt": text_excerpt(str(free.get("reasoning") or "")),
                    "content_excerpt": text_excerpt(str(free.get("content") or ""), limit=300),
                    "manual_next_action_intent": "",
                    "manual_intent_matches_pair": "",
                    "manual_notes": "",
                }
            )
    return review_rows


def build_scoring_rows(
    pairs: list[dict[str, str]],
    free_rows: dict[tuple[str, str, int, str], dict[str, Any]],
    token_rows: dict[tuple[str, str, int, str, int], dict[str, Any]],
    available_tools: list[str],
) -> list[dict[str, Any]]:
    scoring_rows: list[dict[str, Any]] = []
    for pair in pairs:
        candidates = candidate_rows_for_pair(pair, free_rows, available_tools)
        for mode in ("recompute", "rope"):
            free = free_rows.get((*case_key(pair), mode), {})
            for candidate in candidates:
                anchor = str(candidate["boundary_anchor"])
                if anchor == "tool_call_start":
                    boundary_idx = free.get("tool_call_start_token_index")
                    if boundary_idx in (None, ""):
                        boundary_idx = free.get("visible_start_token_index")
                elif anchor == "visible_start":
                    boundary_idx = free.get("visible_start_token_index")
                elif anchor == "function_name":
                    boundary_idx = free.get("function_name_token_index")
                    if boundary_idx in (None, ""):
                        boundary_idx = free.get("visible_start_token_index")
                else:
                    boundary_idx = free.get("action_boundary_token_index")

                token_row = None
                if boundary_idx not in (None, ""):
                    token_row = token_rows.get((*case_key(pair), mode, int(boundary_idx)))
                top_logprobs = (token_row or {}).get("top_logprobs") or []
                generated_logprob = (token_row or {}).get("generated_logprob")
                if token_row is None:
                    support_status = "missing_boundary_logprobs"
                    rank = None
                    logprob = None
                    matched_token = None
                else:
                    support_status, rank, logprob, matched_token = find_candidate_in_topk(
                        top_logprobs,
                        str(candidate["candidate"]),
                        str(candidate["candidate_kind"]),
                    )
                if (
                    support_status == "missing_from_topk"
                    and candidate["candidate_kind"] == "observed_text_token"
                    and str(free.get("action_boundary_generated_token") or "")
                    == str(candidate["candidate"])
                ):
                    support_status = "observed_generated_token"
                    rank = 1 if free.get("action_boundary_top1") == candidate["candidate"] else None
                    logprob = float(generated_logprob) if generated_logprob is not None else None
                    matched_token = str(candidate["candidate"])

                scoring_rows.append(
                    {
                        "task": pair["task"],
                        "skill": pair["skill"],
                        "occurrence": pair["occurrence"],
                        "invocation_index": pair["invocation_index"],
                        "category": pair["category"],
                        "mode": mode,
                        "observed_action": free.get("action_label"),
                        "observed_function_name": free.get("function_name"),
                        "boundary_status": free.get("boundary_status"),
                        "boundary_type": free.get("action_boundary_type"),
                        "boundary_anchor": anchor,
                        "boundary_token_index": boundary_idx,
                        "boundary_generated_token": (token_row or {}).get(
                            "generated_token"
                        ),
                        "boundary_top1_token": (token_row or {}).get("top1_token"),
                        "boundary_margin": (token_row or {}).get("margin"),
                        "candidate_group": candidate["candidate_group"],
                        "candidate": candidate["candidate"],
                        "candidate_kind": candidate["candidate_kind"],
                        "candidate_source": candidate["candidate_source"],
                        "observed_divergence_candidate": candidate[
                            "observed_divergence_candidate"
                        ],
                        "support_status": support_status,
                        "candidate_rank": rank,
                        "candidate_logprob": logprob,
                        "matched_token": matched_token,
                        "topk_size": len(top_logprobs),
                        "needs_forced_scoring": support_status
                        in {"missing_from_topk", "missing_boundary_logprobs"},
                    }
                )
    return scoring_rows


def summarize(scoring_rows: list[dict[str, Any]]) -> dict[str, Any]:
    case_ids = {
        (r["task"], r["skill"], r["occurrence"])
        for r in scoring_rows
        if csv_bool(r.get("observed_divergence_candidate"))
    }
    observed_rows = [
        r for r in scoring_rows if csv_bool(r.get("observed_divergence_candidate"))
    ]
    missing_observed = [r for r in observed_rows if csv_bool(r.get("needs_forced_scoring"))]
    return {
        "bc_cases": len(case_ids),
        "observed_candidate_rows": len(observed_rows),
        "observed_candidate_rows_in_topk_or_generated": len(observed_rows)
        - len(missing_observed),
        "observed_candidate_rows_need_forced_scoring": len(missing_observed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-csv", type=Path, default=DEFAULT_PAIR_CSV)
    parser.add_argument("--free-jsonl", type=Path, default=DEFAULT_FREE_JSONL)
    parser.add_argument("--token-jsonl", type=Path, default=DEFAULT_TOKEN_JSONL)
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--scoring-csv", type=Path, default=DEFAULT_SCORING_CSV)
    args = parser.parse_args()

    pair_rows = [
        row for row in load_csv(args.pair_csv) if row.get("category") in BC_CATEGORIES
    ]
    pair_rows.sort(key=sort_key)
    free_by_mode = {mode_key(row): row for row in load_jsonl(args.free_jsonl)}
    token_by_index = {token_key(row): row for row in load_jsonl(args.token_jsonl)}
    available_tools = [
        tool["function"]["name"]
        for tool in load_tools()
        if tool.get("type") == "function" and tool.get("function", {}).get("name")
    ]

    review_rows = build_review_rows(pair_rows, free_by_mode)
    scoring_rows = build_scoring_rows(
        pair_rows,
        free_by_mode,
        token_by_index,
        available_tools,
    )

    write_csv(
        args.review_csv,
        review_rows,
        [
            "task",
            "skill",
            "occurrence",
            "invocation_index",
            "category",
            "mode",
            "action_label",
            "function_name",
            "boundary_status",
            "action_boundary_type",
            "action_boundary_token_index",
            "action_boundary_margin",
            "action_boundary_top1",
            "intent_label",
            "mentioned_tools",
            "mentioned_files",
            "embedding_cosine",
            "grounding_conflict",
            "reasoning_excerpt",
            "content_excerpt",
            "manual_next_action_intent",
            "manual_intent_matches_pair",
            "manual_notes",
        ],
    )
    write_csv(
        args.scoring_csv,
        scoring_rows,
        [
            "task",
            "skill",
            "occurrence",
            "invocation_index",
            "category",
            "mode",
            "observed_action",
            "observed_function_name",
            "boundary_status",
            "boundary_type",
            "boundary_anchor",
            "boundary_token_index",
            "boundary_generated_token",
            "boundary_top1_token",
            "boundary_margin",
            "candidate_group",
            "candidate",
            "candidate_kind",
            "candidate_source",
            "observed_divergence_candidate",
            "support_status",
            "candidate_rank",
            "candidate_logprob",
            "matched_token",
            "topk_size",
            "needs_forced_scoring",
        ],
    )
    stats = summarize(scoring_rows)
    print(
        "Wrote "
        f"{args.review_csv} ({len(review_rows)} rows), "
        f"{args.scoring_csv} ({len(scoring_rows)} rows). "
        f"Observed candidate rows needing forced scoring: "
        f"{stats['observed_candidate_rows_need_forced_scoring']}/"
        f"{stats['observed_candidate_rows']}"
    )


if __name__ == "__main__":
    main()
