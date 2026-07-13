"""Replay one task's trace JSON files as independent CSKCache requests."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "scripts" / "06_context_free_segment_cache" / "module"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from config import DEFAULT_TASKS  # noqa: E402
from trace_utils import (  # noqa: E402
    load_invocations,
    load_system_prompt,
    load_tools,
    skill_name_from_read_path,
)


DEFAULT_MODEL_PATH = Path(
    "/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B"
)
DEFAULT_KV_DIR = Path(
    "/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/07_cskcache/"
    "offline_skill_kv"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/07_cskcache/"
    "agent_trace_replay"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=DEFAULT_TASKS, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--model", default=os.environ.get("VLLM_SERVED_NAME", "Qwen3"))
    parser.add_argument("--base-url", default="http://127.0.0.1:8013")
    parser.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    parser.add_argument("--kv-dir", type=Path, default=DEFAULT_KV_DIR)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--request-timeout", type=float, default=720.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate ordering and exact skill token spans without sending requests.",
    )
    return parser.parse_args()


def request_json(
    url: str,
    payload: dict[str, Any],
    api_key: str,
    timeout: float,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Request-Id": str(payload["request_id"]),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc


def cache_paths(kv_dir: Path, cache_id: str) -> tuple[Path, Path]:
    import hashlib

    digest = hashlib.sha256(cache_id.encode("utf-8")).hexdigest()[:32]
    return kv_dir / f"{digest}.pt", kv_dir / f"{digest}.json"


def require_cache_entry(kv_dir: Path, cache_id: str, expected_tokens: int) -> None:
    payload_path, sidecar_path = cache_paths(kv_dir, cache_id)
    if not payload_path.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(
            f"Missing offline CSKCache entry for cache_id={cache_id}: "
            f"{payload_path}, {sidecar_path}"
        )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("cache_id") != cache_id:
        raise ValueError(f"CSKCache sidecar cache_id mismatch: {sidecar_path}")
    if int(sidecar.get("num_tokens", -1)) != expected_tokens:
        raise ValueError(
            f"CSKCache token length mismatch for {cache_id}: "
            f"offline={sidecar.get('num_tokens')} trace={expected_tokens}"
        )


def all_subsequence_starts(tokens: list[int], needle: list[int]) -> list[int]:
    if not needle or len(needle) > len(tokens):
        return []
    first = needle[0]
    width = len(needle)
    return [
        index
        for index, token in enumerate(tokens[: len(tokens) - width + 1])
        if token == first and tokens[index : index + width] == needle
    ]


def render_prompt_tokens(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return [int(token) for token in rendered]


def convert_trace_messages(
    anthropic_messages: list[dict[str, Any]],
    system_prompt: str,
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Convert one trace without the legacy 06 context_segment wrapper.

    Qwen's tool-response template adds one newline after tool content. When a
    SKILL.md already ends in a newline, removing that newline from the message
    makes the rendered prompt contain the original Markdown exactly once,
    followed immediately by the template's closing tool-response marker.
    """

    tool_id_to_skill: dict[str, str] = {}
    for message in anthropic_messages:
        if message["role"] != "assistant":
            continue
        for block in message["content"]:
            if block.get("type") != "tool_use" or block.get("name") != "Read":
                continue
            skill = skill_name_from_read_path(block["input"].get("file_path", ""))
            if skill:
                tool_id_to_skill[block["id"]] = skill

    converted: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    skill_bodies: list[tuple[str, str]] = []
    for message in anthropic_messages:
        role = message["role"]
        content = message["content"]
        if role == "user":
            tool_results = [block for block in content if block.get("type") == "tool_result"]
            texts = [block["text"] for block in content if block.get("type") == "text"]
            for block in tool_results:
                tool_id = block["tool_use_id"]
                body = block["content"]
                skill = tool_id_to_skill.get(tool_id)
                if skill is not None:
                    skill_bodies.append((skill, body))
                    if body.endswith("\n"):
                        body = body[:-1]
                converted.append(
                    {"role": "tool", "tool_call_id": tool_id, "content": body}
                )
            if texts:
                converted.append({"role": "user", "content": "\n".join(texts)})
        elif role == "assistant":
            texts = [block["text"] for block in content if block.get("type") == "text"]
            tool_uses = [block for block in content if block.get("type") == "tool_use"]
            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": "\n".join(texts),
            }
            if tool_uses:
                assistant_message["tool_calls"] = [
                    {
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(
                                block["input"], ensure_ascii=False
                            ),
                        },
                    }
                    for block in tool_uses
                ]
            converted.append(assistant_message)
    return converted, skill_bodies


def new_skill_reuse(
    *,
    skill_bodies: list[tuple[str, str]],
    previous_counts: Counter[str],
    tokenizer: Any,
    prompt_tokens: list[int],
    kv_dir: Path,
) -> tuple[dict[str, Any] | None, Counter[str]]:
    current_counts = Counter(name for name, _ in skill_bodies)
    if any(current_counts[name] < count for name, count in previous_counts.items()):
        raise ValueError("Trace skill history is not monotonic")

    additions = current_counts - previous_counts
    if sum(additions.values()) > 1:
        raise ValueError(
            "One request introduced multiple skill bodies, but the current "
            f"CSKCache request protocol supports one reuse span: {dict(additions)}"
        )
    if not additions:
        return None, current_counts

    skill = next(iter(additions))
    occurrence = current_counts[skill]
    matching_bodies = [body for name, body in skill_bodies if name == skill]
    trace_text = matching_bodies[occurrence - 1]

    skill_path = ROOT / "skills" / skill / "SKILL.md"
    local_text = skill_path.read_text(encoding="utf-8")
    if trace_text != local_text:
        raise ValueError(
            f"Trace body differs from {skill_path}; refusing to reuse unrelated KV"
        )
    skill_tokens = tokenizer.encode(local_text, add_special_tokens=False)
    starts = all_subsequence_starts(prompt_tokens, skill_tokens)
    if len(starts) < occurrence:
        raise ValueError(
            f"Could not find occurrence {occurrence} of exact {skill} token sequence "
            f"in rendered prompt; found={len(starts)}"
        )
    target_start = starts[occurrence - 1]
    target_end = target_start + len(skill_tokens)
    require_cache_entry(kv_dir, skill, len(skill_tokens))
    return (
        {
            "operation": "reuse",
            "cache_id": skill,
            "target_start": target_start,
            "target_end": target_end,
            "occurrence": occurrence,
        },
        current_counts,
    )


def response_record(response: dict[str, Any]) -> dict[str, Any]:
    choice = (response.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return {
        "response_id": response.get("id"),
        "finish_reason": choice.get("finish_reason"),
        "message": message,
        "usage": response.get("usage", {}),
    }


def main() -> None:
    args = parse_args()
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    output = args.output or (DEFAULT_OUTPUT_ROOT / f"{args.task}.jsonl")
    partial = output.with_suffix(output.suffix + ".partial")
    if not args.plan_only:
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists: {output}; use --overwrite")
        if partial.exists() and not args.overwrite:
            raise FileExistsError(
                f"Incomplete task output exists: {partial}; replay from task start "
                "with --overwrite"
            )
        output.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    system_prompt = load_system_prompt()
    tools = load_tools()
    invocations = load_invocations(args.task)
    previous_counts: Counter[str] = Counter()

    for invocation_index, invocation in enumerate(invocations, start=1):
        messages, skill_bodies = convert_trace_messages(
            invocation["messages"], system_prompt
        )
        prompt_tokens = render_prompt_tokens(tokenizer, messages, tools)
        reuse, previous_counts = new_skill_reuse(
            skill_bodies=skill_bodies,
            previous_counts=previous_counts,
            tokenizer=tokenizer,
            prompt_tokens=prompt_tokens,
            kv_dir=args.kv_dir,
        )
        available = args.max_model_len - len(prompt_tokens)
        if available <= 0:
            raise ValueError(
                f"Prompt exceeds --max-model-len at invocation={invocation_index}: "
                f"prompt_tokens={len(prompt_tokens)}"
            )
        request_max_tokens = min(args.max_tokens, available)
        request_id = f"cskcache-trace-{args.task}-inv{invocation_index:03d}"
        row: dict[str, Any] = {
            "task": args.task,
            "invocation_index": invocation_index,
            "turn": int(invocation["turn"]),
            "invocation": int(invocation["invocation"]),
            "trace_file": (
                f"src/traces/{args.task}/turn_{invocation['turn']}_"
                f"inv_{invocation['invocation']}.json"
            ),
            "request_id": request_id,
            "prompt_tokens": len(prompt_tokens),
            "max_tokens": request_max_tokens,
            "reuse": reuse,
        }
        if args.plan_only:
            row["status"] = "planned"
        else:
            payload: dict[str, Any] = {
                "model": args.model,
                "messages": messages,
                "tools": tools,
                "max_tokens": request_max_tokens,
                "temperature": args.temperature,
                "request_id": request_id,
                "chat_template_kwargs": {"enable_thinking": True},
            }
            if reuse is not None:
                cskcache = {key: value for key, value in reuse.items() if key != "occurrence"}
                payload["kv_transfer_params"] = {"cskcache": cskcache}
            started = time.perf_counter()
            try:
                response = request_json(
                    f"{args.base_url.rstrip('/')}/v1/chat/completions",
                    payload,
                    args.api_key,
                    args.request_timeout,
                )
                row.update(
                    status="completed",
                    elapsed_s=round(time.perf_counter() - started, 4),
                    response=response_record(response),
                )
            except Exception as exc:
                row.update(
                    status="failed",
                    elapsed_s=round(time.perf_counter() - started, 4),
                    error=str(exc),
                )
                with partial.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                raise
            with partial.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

        reuse_label = "none"
        if reuse is not None:
            reuse_label = (
                f"{reuse['cache_id']}#{reuse['occurrence']}"
                f"[{reuse['target_start']},{reuse['target_end']})"
            )
        print(
            f"[{row['status']}] task={args.task} inv={invocation_index:03d} "
            f"turn={invocation['turn']} call={invocation['invocation']} "
            f"prompt={len(prompt_tokens)} reuse={reuse_label}",
            flush=True,
        )

    if not args.plan_only:
        partial.replace(output)
        print(f"[done] output={output}", flush=True)


if __name__ == "__main__":
    main()
