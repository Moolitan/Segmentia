"""Save occurrence-1 skill KV computed in each task's real trace context.

The vLLM server must be started with ``VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR``
pointing to the same directory passed through ``--kv-dir``. Unlike
``prefill_clean_skill_kv.py``, this script keeps the real system prompt,
tools, and conversation history preceding the first skill occurrence.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_DIR = SCRIPT_DIR / "module"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from config import (  # noqa: E402
    DEFAULT_CONTEXTUAL_KV_DIR,
    DEFAULT_SERVED_MODEL,
    DEFAULT_VLLM_PORT,
    iter_task_skills,
    parse_tasks,
)
from trace_utils import (  # noqa: E402
    convert_messages,
    find_skill_segments,
    load_invocations,
    load_system_prompt,
    load_tools,
    span_token_offsets,
    target_tokens_for_occurrence,
)
from vllm_client import chat_completion, tokenize_chat  # noqa: E402


def contextual_cache_id(task: str, skill: str) -> str:
    """Keep separate occurrence-1 sources for the same skill across tasks."""
    return f"contextual-occ1-skill-{task}-{skill}"


def first_mismatch(source: list[int], target: list[int]) -> int | None:
    if source == target:
        return None
    return next(
        (
            index
            for index, (source_id, target_id) in enumerate(zip(source, target))
            if source_id != target_id
        ),
        min(len(source), len(target)),
    )


def prepare_source(
    *,
    base_url: str,
    model: str,
    api_key: str,
    task: str,
    skill: str,
    record: dict[str, Any],
    system_prompt: str,
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build and validate one task-specific occurrence-1 source request."""
    invocation_index = int(record["invocation_indices"][0])
    configured_start, configured_end = (
        int(record["token_spans"][0][0]),
        int(record["token_spans"][0][1]),
    )
    invocations = load_invocations(task)
    if invocation_index < 1 or invocation_index > len(invocations):
        raise IndexError(
            f"Invalid invocation_index={invocation_index} for task={task}; "
            f"available={len(invocations)}"
        )

    invocation = invocations[invocation_index - 1]
    messages, _ = convert_messages(invocation["messages"], system_prompt)
    segments = [segment for segment in find_skill_segments(messages) if segment[0] == skill]
    if len(segments) != 1:
        raise RuntimeError(
            f"Expected exactly one occurrence of skill={skill} in "
            f"task={task} invocation={invocation_index}, got {segments}"
        )

    _name, msg_idx, char_start, char_end = segments[0]
    observed_start, observed_end = span_token_offsets(
        base_url,
        model,
        messages,
        tools,
        api_key,
        msg_idx,
        char_start,
        char_end,
    )
    if (observed_start, observed_end) != (configured_start, configured_end):
        raise RuntimeError(
            f"Configured occurrence-1 span mismatch for task={task} skill={skill}: "
            f"configured=[{configured_start},{configured_end}), "
            f"observed=[{observed_start},{observed_end})"
        )

    all_tokens = tokenize_chat(
        base_url,
        model,
        messages,
        tools,
        api_key,
        add_generation_prompt=True,
    )
    source_tokens = all_tokens[configured_start:configured_end]
    checks: list[dict[str, Any]] = []
    for occurrence in range(1, len(record["token_spans"]) + 1):
        target_tokens = target_tokens_for_occurrence(
            base_url,
            model,
            task,
            skill,
            occurrence,
            system_prompt,
            tools,
            api_key,
        )
        mismatch = first_mismatch(source_tokens, target_tokens)
        checks.append(
            {
                "occurrence": occurrence,
                "ok": mismatch is None,
                "source_tokens": len(source_tokens),
                "target_tokens": len(target_tokens),
                "first_mismatch": mismatch,
            }
        )
    failed_checks = [check for check in checks if not check["ok"]]
    if failed_checks:
        raise RuntimeError(
            f"Token identity mismatch for task={task} skill={skill}: {failed_checks}"
        )

    return {
        "task": task,
        "skill": skill,
        "invocation_index": invocation_index,
        "turn": invocation["turn"],
        "invocation": invocation["invocation"],
        "source_start": configured_start,
        "source_end": configured_end,
        "source_tokens": source_tokens,
        "messages": messages,
        "checks": checks,
        "cache_id": contextual_cache_id(task, skill),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--vllm-port", type=int, default=DEFAULT_VLLM_PORT)
    parser.add_argument(
        "--model",
        default=os.environ.get("VLLM_SERVED_NAME", DEFAULT_SERVED_MODEL),
    )
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--kv-dir", type=Path, default=DEFAULT_CONTEXTUAL_KV_DIR)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    tasks = parse_tasks(args.tasks)
    base_url = args.base_url or f"http://127.0.0.1:{args.vllm_port}"
    kv_dir = args.kv_dir
    kv_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or kv_dir / "manifest.json"
    system_prompt = load_system_prompt()
    tools = load_tools()
    task_skills = iter_task_skills(tasks)
    task_order = {task: index for index, task in enumerate(tasks)}
    task_skills.sort(
        key=lambda item: (
            task_order[item[0]],
            int(item[2]["invocation_indices"][0]),
        )
    )

    expected_paths = [
        kv_dir / f"{contextual_cache_id(task, skill)}.pt"
        for task, skill, _record in task_skills
    ]
    existing_paths = [path for path in expected_paths if path.exists()]
    if existing_paths:
        rendered = "\n".join(f"  {path}" for path in existing_paths)
        raise FileExistsError(
            "Contextual KV outputs already exist. Use an empty output directory and "
            "restart vLLM before recollecting:\n"
            f"{rendered}"
        )
    if manifest_path.exists():
        raise FileExistsError(f"Manifest already exists: {manifest_path}")

    # Validate every configured source before collecting any KV, so a stale span
    # or token mismatch cannot leave a partially collected comparison set.
    prepared = [
        prepare_source(
            base_url=base_url,
            model=args.model,
            api_key=args.api_key,
            task=task,
            skill=skill,
            record=record,
            system_prompt=system_prompt,
            tools=tools,
        )
        for task, skill, record in task_skills
    ]

    records: list[dict[str, Any]] = []
    for source in prepared:
        cfg = {
            "sources": [
                {
                    "cache_id": source["cache_id"],
                    "source_start": source["source_start"],
                    "source_end": source["source_end"],
                }
            ]
        }
        response, elapsed = chat_completion(
            base_url,
            args.model,
            source["messages"],
            tools,
            args.api_key,
            max_tokens=args.max_tokens,
            request_id=(
                f"contextual-occ1-prefill-{source['task']}-{source['skill']}"
            ),
            context_segment_cache=cfg,
        )
        out_path = kv_dir / f"{source['cache_id']}.pt"
        if not out_path.exists():
            raise RuntimeError(
                f"vLLM did not save contextual KV for task={source['task']} "
                f"skill={source['skill']}; expected={out_path}"
            )
        record = {
            key: value
            for key, value in source.items()
            if key not in {"messages", "source_tokens"}
        }
        record.update(
            {
                "status": "collected",
                "source_token_count": len(source["source_tokens"]),
                "path": str(out_path),
                "latency_s": round(elapsed, 4),
                "usage": response.get("usage", {}),
            }
        )
        records.append(record)
        print(
            f"[collect] task={source['task']:30s} skill={source['skill']:24s} "
            f"inv={source['invocation_index']:02d} "
            f"span=[{source['source_start']},{source['source_end']}) "
            f"saved={out_path}",
            flush=True,
        )

    manifest = {
        "source_semantics": (
            "task-specific occurrence-1 skill KV with real system prompt, tools, "
            "and preceding trace conversation context"
        ),
        "model": args.model,
        "base_url": base_url,
        "tasks": tasks,
        "kv_dir": str(kv_dir),
        "cache_id_format": "contextual-occ1-skill-{task}-{skill}",
        "records": records,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[done] manifest={manifest_path} records={len(records)}")


if __name__ == "__main__":
    main()
