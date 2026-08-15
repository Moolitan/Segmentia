#!/usr/bin/env python3
"""Replay one prepared source or target request through Segmentia lookup."""
from __future__ import annotations

import argparse
import json
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from capture_common import (
    atomic_write_json,
    convert_messages,
    convert_tools,
    locate_skill_segment,
    sha256_text,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_PATH = Path(
    "/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B"
)


def post_json(
    *, base_url: str, path: str, api_key: str, payload: dict[str, Any]
) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=720) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {path}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, OSError, socket.timeout) as exc:
        raise RuntimeError(f"Connection failure from {path}: {exc}") from exc
    if not isinstance(body, dict):
        raise TypeError(f"Expected JSON object from {path}")
    return body


def selected_case(prepared_path: Path, case_id: str) -> dict[str, Any]:
    prepared = json.loads(prepared_path.read_text(encoding="utf-8"))
    matches = [case for case in prepared.get("cases", []) if case.get("case_id") == case_id]
    if len(matches) != 1:
        raise ValueError(f"Expected one prepared case_id={case_id!r}, got {len(matches)}")
    return matches[0]


def load_tokenizer(path: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-cases", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument(
        "--phase", choices=("source", "target_reuse", "target_full"), required=True
    )
    parser.add_argument(
        "--endpoint-name",
        choices=("source", "target"),
        help="Override which prepared endpoint is replayed for this phase.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--traces-dir", type=Path, default=ROOT / "src" / "traces")
    parser.add_argument("--skills-dir", type=Path, default=ROOT / "skills")
    parser.add_argument("--tokenizer-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--base-url", default="http://127.0.0.1:8100")
    parser.add_argument("--model", default="Qwen3")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--separator", required=True)
    parser.add_argument(
        "--correction-mode",
        choices=("none", "prefix_k_headwise"),
        default="none",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Render and validate locally without contacting vLLM.",
    )
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Replay output already exists: {args.output}")
    case = selected_case(args.prepared_cases, args.case_id)
    endpoint_name = args.endpoint_name or (
        "source" if args.phase == "source" else "target"
    )
    endpoint = case[endpoint_name]
    skill = str(case["skill"])
    skill_content = (args.skills_dir / skill / "SKILL.md").read_text(encoding="utf-8")
    if sha256_text(skill_content) != case["skill_sha256"]:
        raise RuntimeError(
            f"Canonical Skill changed after case preparation: skill={skill!r}"
        )

    invocation_path = Path(endpoint["trace_path"])
    invocation = json.loads(invocation_path.read_text(encoding="utf-8"))
    anthropic_messages = invocation.get("messages")
    if not isinstance(anthropic_messages, list):
        raise TypeError(f"Trace messages are not a list: {invocation_path}")
    system_prompt = (args.traces_dir / "_system_prompt.txt").read_text(encoding="utf-8")
    raw_tools = json.loads((args.traces_dir / "_tools.json").read_text(encoding="utf-8"))
    if not isinstance(raw_tools, list):
        raise TypeError("src/traces/_tools.json must contain a list")
    tools = convert_tools(raw_tools)
    messages, tool_id_to_skill = convert_messages(
        anthropic_messages=anthropic_messages,
        system_prompt=system_prompt,
        skills_dir=args.skills_dir,
        separator=args.separator,
    )
    target_tool_call_id = str(endpoint["target_tool_call_id"])
    if tool_id_to_skill.get(target_tool_call_id) != skill:
        raise RuntimeError(
            f"Prepared target tool ID does not map to skill={skill!r}: "
            f"id={target_tool_call_id!r} mapping={tool_id_to_skill.get(target_tool_call_id)!r}"
        )

    tokenizer = load_tokenizer(args.tokenizer_path)
    segment_start, segment_end, prompt_ids, separator_ids = locate_skill_segment(
        tokenizer=tokenizer,
        messages=messages,
        tools=tools,
        tool_id_to_skill=tool_id_to_skill,
        target_tool_call_id=target_tool_call_id,
        separator=args.separator,
    )
    request_id = f"segmentia-m0-{args.case_id}-{args.phase}"
    segmentia_lookup: dict[str, Any] = {
        "segment_start": segment_start,
        "segment_end": segment_end,
    }
    if args.correction_mode == "prefix_k_headwise":
        segmentia_lookup.update(
            {
                "correction_mode": "prefix_k_headwise",
                "cache_end": segment_end - len(separator_ids),
                "prefix_tokens": 256,
                "calibration_start": 132,
                "calibration_end": 256,
                "minimum_reuse_tokens": 256,
                "correction_alpha": 0.6,
            }
        )
    completion_payload: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "tools": tools,
        "max_tokens": 1,
        "temperature": 0,
        "request_id": request_id,
        "chat_template_kwargs": {"enable_thinking": True},
        "kv_transfer_params": {
            "lmcache_segmentia_lookup": segmentia_lookup
        },
    }
    record: dict[str, Any] = {
        "schema_version": 1,
        "case_id": args.case_id,
        "phase": args.phase,
        "endpoint": endpoint_name,
        "skill": skill,
        "skill_sha256": case["skill_sha256"],
        "task": endpoint["task"],
        "trace_path": str(invocation_path.resolve()),
        "turn": endpoint["turn"],
        "invocation": endpoint["invocation"],
        "target_tool_call_id": target_tool_call_id,
        "request_id": request_id,
        "model": args.model,
        "tokenizer_path": str(args.tokenizer_path.resolve()),
        "separator": args.separator,
        "effective_separator_tokens": separator_ids,
        "prompt_token_count": len(prompt_ids),
        "prompt_token_ids": prompt_ids,
        "segment_start": segment_start,
        "segment_end": segment_end,
        "segment_token_count": segment_end - segment_start,
        "segment_token_ids": prompt_ids[segment_start:segment_end],
        "correction_mode": args.correction_mode,
        "request": completion_payload,
        "status": "prepared" if args.prepare_only else "sending",
    }
    atomic_write_json(args.output, record)
    if args.prepare_only:
        print(
            f"[prepared] case={args.case_id} phase={args.phase} "
            f"span=[{segment_start},{segment_end}) tokens={segment_end - segment_start}"
        )
        return

    tokenize_payload: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "tools": tools,
        "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    remote_tokenize = post_json(
        base_url=args.base_url,
        path="/tokenize",
        api_key=args.api_key,
        payload=tokenize_payload,
    )
    remote_ids = remote_tokenize.get("tokens")
    if remote_ids != prompt_ids:
        raise RuntimeError(
            "Local tokenizer and vLLM /tokenize disagree; refusing to capture "
            f"local_tokens={len(prompt_ids)} remote_tokens="
            f"{len(remote_ids) if isinstance(remote_ids, list) else None}"
        )

    started = time.perf_counter()
    response = post_json(
        base_url=args.base_url,
        path="/v1/chat/completions",
        api_key=args.api_key,
        payload=completion_payload,
    )
    record.update(
        status="completed",
        elapsed_s=round(time.perf_counter() - started, 6),
        response=response,
        response_id=response.get("id"),
    )
    atomic_write_json(args.output, record)
    print(
        f"[captured] case={args.case_id} phase={args.phase} task={endpoint['task']} "
        f"span=[{segment_start},{segment_end}) tokens={segment_end - segment_start}"
    )


if __name__ == "__main__":
    main()
