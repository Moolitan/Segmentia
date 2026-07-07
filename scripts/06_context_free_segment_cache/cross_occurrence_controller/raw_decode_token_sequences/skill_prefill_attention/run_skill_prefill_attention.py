"""Capture normal-prefill attention for Skill tokens and the final prompt token."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[5]
MODULE_DIR = ROOT / "scripts/06_context_free_segment_cache/module"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from config import (  # noqa: E402
    DEFAULT_SERVED_MODEL,
    DEFAULT_TASKS,
    DEFAULT_VLLM_PORT,
)
from replay import context_config_for_case, selected_cases  # noqa: E402
from trace_utils import (  # noqa: E402
    convert_messages,
    load_invocations,
    load_system_prompt,
    load_tools,
)
from vllm_client import chat_completion, tokenize_chat  # noqa: E402


OCCURRENCES = (3,)
SKILL_QUERY_TOKENS = 30
PRE_SKILL_WINDOW = 512


def safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def case_label(case: dict[str, Any]) -> str:
    return (
        f"inv{int(case['invocation_index']):03d}--"
        f"{safe_component(str(case['task']))}--"
        f"{safe_component(str(case['skill']))}--"
        f"occ{int(case['occurrence'])}"
    )


def span(start: int, end: int) -> dict[str, int]:
    bounded_start = max(0, int(start))
    bounded_end = max(bounded_start, int(end))
    return {
        "start": bounded_start,
        "end": bounded_end,
        "length": bounded_end - bounded_start,
    }


def build_probe_marker(
    *,
    case: dict[str, Any],
    prompt_tokens: int,
    request_id: str,
) -> dict[str, Any]:
    target_start = int(case["target_start"])
    target_end = int(case["target_end"])
    if target_end - target_start < SKILL_QUERY_TOKENS:
        raise ValueError(
            f"{case_label(case)}: target span has only "
            f"{target_end - target_start} tokens"
        )
    if target_start < PRE_SKILL_WINDOW:
        raise ValueError(
            f"{case_label(case)}: target_start={target_start} is shorter than "
            f"the required {PRE_SKILL_WINDOW}-token context"
        )
    expected_prompt_tokens = target_end + 3
    if int(prompt_tokens) != expected_prompt_tokens:
        raise ValueError(
            f"{case_label(case)}: expected prompt_tokens=target_end+3="
            f"{expected_prompt_tokens} for <|im_start|>assistant\\n, "
            f"found {prompt_tokens}"
        )

    final_assistant_query = int(prompt_tokens) - 1
    if final_assistant_query < target_end:
        raise ValueError(
            f"{case_label(case)}: final prompt query "
            f"{final_assistant_query} precedes target_end={target_end}"
        )

    pre_skill = span(target_start - PRE_SKILL_WINDOW, target_start)
    context_and_skill = span(target_start - PRE_SKILL_WINDOW, target_end)
    queries: list[dict[str, Any]] = []
    for offset in range(SKILL_QUERY_TOKENS):
        queries.append(
            {
                "query_name": f"skill_token_{offset:02d}",
                "query_abs_position": target_start + offset,
                "regions": [],
                "local_windows": {"pre_skill_context": pre_skill},
            }
        )
    queries.append(
        {
            "query_name": "final_assistant_prompt_token",
            "query_abs_position": final_assistant_query,
            "regions": [],
            "local_windows": {"context_and_skill": context_and_skill},
        }
    )

    return {
        "enabled": True,
        "marker_version": 3,
        "case_id": (
            f"{case['task']}/{case['skill']}/occ{int(case['occurrence'])}"
        ),
        "mode": "recompute",
        "run_name": "recompute_prefill_attention",
        "task": str(case["task"]),
        "skill": str(case["skill"]),
        "occurrence": int(case["occurrence"]),
        "invocation_index": int(case["invocation_index"]),
        "request_id": request_id,
        "prompt_tokens": int(prompt_tokens),
        "target_start": target_start,
        "target_end": target_end,
        "skill_query_tokens": SKILL_QUERY_TOKENS,
        "pre_skill_window": PRE_SKILL_WINDOW,
        "queries": queries,
    }


def write_probe_marker(path: Path, marker: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(marker, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def clear_probe_marker(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def append_manifest(path: Path, marker: dict[str, Any]) -> None:
    record = {
        key: marker[key]
        for key in (
            "case_id",
            "mode",
            "task",
            "skill",
            "occurrence",
            "invocation_index",
            "request_id",
            "prompt_tokens",
            "target_start",
            "target_end",
            "skill_query_tokens",
            "pre_skill_window",
            "queries",
        )
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=DEFAULT_TASKS, required=True)
    parser.add_argument("--probe-marker", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--vllm-port", type=int, default=DEFAULT_VLLM_PORT)
    parser.add_argument(
        "--model",
        default=os.environ.get("VLLM_SERVED_NAME", DEFAULT_SERVED_MODEL),
    )
    parser.add_argument(
        "--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = selected_cases(
        [args.task], list(OCCURRENCES), include_first_occurrence=True
    )
    if not cases:
        raise ValueError(f"No occurrence-3 cases for task={args.task}")
    cases.sort(key=lambda case: int(case["invocation_index"]))

    system_prompt = load_system_prompt()
    tools = load_tools()
    invocations = load_invocations(args.task)
    base_url = args.base_url or f"http://127.0.0.1:{args.vllm_port}"
    clear_probe_marker(args.probe_marker)

    for case in cases:
        invocation_index = int(case["invocation_index"])
        invocation = invocations[invocation_index - 1]
        messages, _ = convert_messages(
            invocation["messages"], system_prompt
        )
        prompt_tokens = len(
            tokenize_chat(
                base_url,
                args.model,
                messages,
                tools,
                args.api_key,
                add_generation_prompt=True,
            )
        )
        request_id = (
            f"cf-skill-prefill-{safe_component(str(case['task']))}-"
            f"{safe_component(str(case['skill']))}-"
            f"occ{int(case['occurrence'])}-inv{invocation_index}"
        )
        marker = build_probe_marker(
            case=case,
            prompt_tokens=prompt_tokens,
            request_id=request_id,
        )
        write_probe_marker(args.probe_marker, marker)
        try:
            chat_completion(
                base_url,
                args.model,
                messages,
                tools,
                args.api_key,
                max_tokens=1,
                request_id=request_id,
                context_segment_cache=context_config_for_case(
                    "recompute", case, dump_kv_for_cksim=False
                ),
                temperature=0.0,
            )
        finally:
            clear_probe_marker(args.probe_marker)
        append_manifest(args.manifest, marker)
        print(
            f"[ok] {marker['case_id']} inv={invocation_index} "
            f"prompt_tokens={prompt_tokens} "
            f"skill_queries=[{marker['target_start']},"
            f"{marker['target_start'] + SKILL_QUERY_TOKENS}) "
            f"final_assistant_query={prompt_tokens - 1}",
            flush=True,
        )


if __name__ == "__main__":
    main()
