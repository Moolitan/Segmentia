"""Offline prefill wrapped skill text and save ContextSegmentKV .pt files.

Run this against a vLLM server started with VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR
pointing to the desired output directory.
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
for path in (SCRIPT_DIR, MODULE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import (  # noqa: E402
    DEFAULT_KV_DIR,
    DEFAULT_SERVED_MODEL,
    DEFAULT_TASKS,
    DEFAULT_VLLM_PORT,
    SKILL_TOKEN_LOCATIONS,
    cache_id_for_skill,
    iter_task_skills,
    parse_tasks,
)
from trace_utils import (  # noqa: E402
    extract_first_skill_text,
    find_skill_segments,
    load_system_prompt,
    load_tools,
    minimal_skill_messages,
    span_token_offsets,
    target_tokens_for_occurrence,
)
from vllm_client import chat_completion, tokenize_chat  # noqa: E402


def unique_skill_sources(tasks: list[str]) -> list[tuple[str, str]]:
    first_task_by_skill: dict[str, str] = {}
    for task, skill, _record in iter_task_skills(tasks):
        first_task_by_skill.setdefault(skill, task)
    return [(task, skill) for skill, task in sorted(first_task_by_skill.items())]


def verify_target_tokens(
    *,
    base_url: str,
    model: str,
    api_key: str,
    tasks: list[str],
    skill: str,
    source_tokens: list[int],
    system_prompt: str,
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for task in tasks:
        if skill not in SKILL_TOKEN_LOCATIONS[task]["skills"]:
            continue
        spans = SKILL_TOKEN_LOCATIONS[task]["skills"][skill]["token_spans"]
        for occurrence in range(1, len(spans) + 1):
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
            ok = source_tokens == target_tokens
            mismatch = None
            if not ok:
                mismatch = next(
                    (
                        idx
                        for idx, (src_id, tgt_id) in enumerate(
                            zip(source_tokens, target_tokens)
                        )
                        if src_id != tgt_id
                    ),
                    min(len(source_tokens), len(target_tokens)),
                )
            checks.append(
                {
                    "task": task,
                    "skill": skill,
                    "occurrence": occurrence,
                    "ok": ok,
                    "source_tokens": len(source_tokens),
                    "target_tokens": len(target_tokens),
                    "first_mismatch": mismatch,
                }
            )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--vllm-port", type=int, default=DEFAULT_VLLM_PORT)
    parser.add_argument("--model", default=os.environ.get("VLLM_SERVED_NAME", DEFAULT_SERVED_MODEL))
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--kv-dir", default=str(DEFAULT_KV_DIR))
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--verify-target-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--strict-token-identity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()

    tasks = parse_tasks(args.tasks)
    base_url = args.base_url or f"http://127.0.0.1:{args.vllm_port}"
    kv_dir = Path(args.kv_dir)
    kv_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else kv_dir / "manifest.json"
    system_prompt = load_system_prompt()
    tools = load_tools()

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for source_task, skill in unique_skill_sources(tasks):
        skill_text = extract_first_skill_text(source_task, skill, system_prompt)
        messages = minimal_skill_messages(skill, skill_text)
        segments = find_skill_segments(messages)
        skill_segments = [seg for seg in segments if seg[0] == skill]
        if len(skill_segments) != 1:
            raise RuntimeError(f"Expected one segment for skill={skill}, got {segments}")
        _name, msg_idx, char_start, char_end = skill_segments[0]
        source_start, source_end = span_token_offsets(
            base_url,
            args.model,
            messages,
            None,
            args.api_key,
            msg_idx,
            char_start,
            char_end,
        )
        all_tokens = tokenize_chat(
            base_url,
            args.model,
            messages,
            None,
            args.api_key,
            add_generation_prompt=True,
        )
        source_tokens = all_tokens[source_start:source_end]
        cache_id = cache_id_for_skill(skill)

        checks: list[dict[str, Any]] = []
        if args.verify_target_tokens:
            checks = verify_target_tokens(
                base_url=base_url,
                model=args.model,
                api_key=args.api_key,
                tasks=tasks,
                skill=skill,
                source_tokens=source_tokens,
                system_prompt=system_prompt,
                tools=tools,
            )
            failures.extend(check for check in checks if not check["ok"])
            if args.strict_token_identity and any(not check["ok"] for check in checks):
                print(
                    f"[fail] token identity mismatch for skill={skill}; "
                    "not collecting KV for this skill",
                    flush=True,
                )
                records.append(
                    {
                        "skill": skill,
                        "source_task": source_task,
                        "cache_id": cache_id,
                        "status": "token_identity_mismatch",
                        "source_start": source_start,
                        "source_end": source_end,
                        "source_tokens": len(source_tokens),
                        "checks": checks,
                    }
                )
                continue

        cfg = {
            "sources": [
                {
                    "cache_id": cache_id,
                    "source_start": source_start,
                    "source_end": source_end,
                }
            ]
        }
        response, elapsed = chat_completion(
            base_url,
            args.model,
            messages,
            None,
            args.api_key,
            max_tokens=args.max_tokens,
            request_id=f"context-free-prefill-{skill}",
            context_segment_cache=cfg,
        )
        out_path = kv_dir / f"{cache_id}.pt"
        record = {
            "skill": skill,
            "source_task": source_task,
            "cache_id": cache_id,
            "status": "collected",
            "source_start": source_start,
            "source_end": source_end,
            "source_tokens": len(source_tokens),
            "expected_path": str(out_path),
            "path_exists_after_request": out_path.exists(),
            "latency_s": round(elapsed, 4),
            "usage": response.get("usage", {}),
            "checks": checks,
        }
        records.append(record)
        print(
            f"[collect] {skill:24s} cache_id={cache_id} "
            f"tokens={len(source_tokens)} saved={out_path.exists()}",
            flush=True,
        )

    manifest = {
        "model": args.model,
        "base_url": base_url,
        "tasks": tasks,
        "kv_dir": str(kv_dir),
        "records": records,
        "token_identity_failures": failures,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] manifest: {manifest_path}")
    if failures and args.strict_token_identity:
        raise SystemExit(
            "Token identity mismatch found. Use trace-occ1 collection or inspect manifest."
        )


if __name__ == "__main__":
    main()
