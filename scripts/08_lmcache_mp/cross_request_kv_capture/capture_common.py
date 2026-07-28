"""Shared trace and prompt helpers for 08 cross-request KV capture."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def invocation_sort_key(path: Path) -> tuple[int, int]:
    stem = path.stem
    try:
        turn = int(stem.split("turn_", 1)[1].split("_", 1)[0])
        invocation = int(stem.rsplit("_inv_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid trace invocation filename: {path}") from exc
    return turn, invocation


def trace_paths(traces_dir: Path, task: str) -> list[Path]:
    paths = sorted(
        (traces_dir / task).glob("turn_*_inv_*.json"),
        key=invocation_sort_key,
    )
    if not paths:
        raise FileNotFoundError(f"No trace invocations for task={task!r}")
    return paths


def skill_name_from_read_path(value: str) -> str | None:
    normalized = value.replace("\\", "/")
    if not normalized.endswith("/SKILL.md"):
        return None
    components = normalized.split("/")
    try:
        skills_index = components.index("skills")
    except ValueError:
        return None
    if skills_index + 1 >= len(components):
        return None
    return components[skills_index + 1]


def skill_tool_ids(messages: list[dict[str, Any]]) -> dict[str, str]:
    """Map trace tool-use IDs to structurally identified Skill names."""

    result: dict[str, str] = {}
    for message in messages:
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            raise TypeError("Anthropic assistant message content must be a list")
        for block in content:
            if block.get("type") != "tool_use":
                continue
            tool_id = block.get("id")
            tool_name = block.get("name")
            tool_input = block.get("input") or {}
            if not isinstance(tool_id, str) or not isinstance(tool_input, dict):
                raise TypeError("Malformed tool_use block in trace")
            skill: str | None = None
            if tool_name in {"Read", "read_file"}:
                path = tool_input.get("file_path") or tool_input.get("path") or ""
                if isinstance(path, str):
                    skill = skill_name_from_read_path(path)
            elif tool_name == "skill":
                candidate = tool_input.get("skill_name") or tool_input.get("name")
                if isinstance(candidate, str) and candidate and candidate != "list":
                    skill = candidate
            if skill is not None:
                result[tool_id] = skill
    return result


def tool_result_ids(messages: list[dict[str, Any]]) -> set[str]:
    result: set[str] = set()
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            raise TypeError("Anthropic user message content must be a list")
        for block in content:
            if block.get("type") == "tool_result":
                tool_id = block.get("tool_use_id")
                if not isinstance(tool_id, str):
                    raise TypeError("Malformed tool_result block in trace")
                result.add(tool_id)
    return result


def first_skill_request(
    traces_dir: Path, task: str, skill: str
) -> tuple[Path, dict[str, Any], str]:
    """Find the first request snapshot containing a completed Skill read."""

    for path in trace_paths(traces_dir, task):
        invocation = json.loads(path.read_text(encoding="utf-8"))
        messages = invocation.get("messages")
        if not isinstance(messages, list):
            raise TypeError(f"Trace messages are not a list: {path}")
        mapping = skill_tool_ids(messages)
        completed = tool_result_ids(messages)
        matches = sorted(
            tool_id
            for tool_id, skill_name in mapping.items()
            if skill_name == skill and tool_id in completed
        )
        if matches:
            if len(matches) != 1:
                raise ValueError(
                    f"First Skill request must contain one completed {skill!r} read; "
                    f"path={path} matches={matches}"
                )
            return path, invocation, matches[0]
    raise ValueError(f"Task {task!r} never completes a read of Skill {skill!r}")


def convert_tools(raw_tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("input_schema", {"type": "object"}),
            },
        }
        for tool in raw_tools
    ]


def convert_messages(
    *,
    anthropic_messages: list[dict[str, Any]],
    system_prompt: str,
    skills_dir: Path,
    separator: str,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Convert a frozen Anthropic trace and canonicalize every Skill result."""

    mapping = skill_tool_ids(anthropic_messages)
    canonical: dict[str, str] = {}
    for tool_id, skill in mapping.items():
        skill_path = skills_dir / skill / "SKILL.md"
        if not skill_path.is_file():
            raise FileNotFoundError(
                f"Trace references Skill {skill!r}, missing canonical file {skill_path}"
            )
        content = skill_path.read_text(encoding="utf-8")
        if not content.strip():
            raise ValueError(f"Canonical Skill file is empty: {skill_path}")
        canonical[tool_id] = content

    output: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for message in anthropic_messages:
        role = message.get("role")
        blocks = message.get("content")
        if not isinstance(blocks, list):
            raise TypeError("Anthropic message content must be a list")
        if role == "user":
            tool_results = [block for block in blocks if block.get("type") == "tool_result"]
            texts = [block.get("text") for block in blocks if block.get("type") == "text"]
            for block in tool_results:
                tool_id = block.get("tool_use_id")
                if not isinstance(tool_id, str):
                    raise TypeError("Malformed tool_result block")
                body = block.get("content")
                if not isinstance(body, str):
                    raise TypeError(f"Tool result {tool_id!r} content must be text")
                if tool_id in canonical:
                    body = f"{separator}{canonical[tool_id]}{separator}"
                output.append(
                    {"role": "tool", "tool_call_id": tool_id, "content": body}
                )
            clean_texts = [text for text in texts if isinstance(text, str)]
            if len(clean_texts) != len(texts):
                raise TypeError("User text block is missing string text")
            if clean_texts:
                output.append({"role": "user", "content": "\n".join(clean_texts)})
        elif role == "assistant":
            texts = [block.get("text") for block in blocks if block.get("type") == "text"]
            clean_texts = [text for text in texts if isinstance(text, str)]
            if len(clean_texts) != len(texts):
                raise TypeError("Assistant text block is missing string text")
            tool_uses = [block for block in blocks if block.get("type") == "tool_use"]
            converted: dict[str, Any] = {
                "role": "assistant",
                "content": "\n".join(clean_texts),
            }
            if tool_uses:
                converted["tool_calls"] = [
                    {
                        "id": block["id"],
                        "type": "function",
                        "function": {
                            "name": block["name"],
                            "arguments": json.dumps(
                                block.get("input") or {}, ensure_ascii=False
                            ),
                        },
                    }
                    for block in tool_uses
                ]
            output.append(converted)
        else:
            raise ValueError(f"Unsupported Anthropic message role: {role!r}")
    return output, mapping


def subsequence_starts(values: list[int], needle: list[int]) -> list[int]:
    if not needle:
        raise ValueError("Cannot find an empty token subsequence")
    width = len(needle)
    return [
        index
        for index in range(len(values) - width + 1)
        if values[index : index + width] == needle
    ]


def effective_separator_tokens(tokenizer: Any, separator: str) -> list[int]:
    """Mirror SegmentTokenDatabase's tokenizer.encode(separator)[1:]."""

    tokens = list(tokenizer.encode(separator))[1:]
    if not tokens:
        raise ValueError("LMCache effective separator token sequence is empty")
    return tokens


def render_prompt_token_ids(
    tokenizer: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if hasattr(rendered, "input_ids"):
        rendered = rendered.input_ids
    if not isinstance(rendered, list) or not all(isinstance(token, int) for token in rendered):
        raise TypeError("Tokenizer chat template did not return flat token IDs")
    return rendered


def locate_skill_segment(
    *,
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    tool_id_to_skill: dict[str, str],
    target_tool_call_id: str,
    separator: str,
) -> tuple[int, int, list[int], list[int]]:
    ordered_skill_ids = [
        str(message.get("tool_call_id"))
        for message in messages
        if message.get("role") == "tool"
        and message.get("tool_call_id") in tool_id_to_skill
    ]
    if target_tool_call_id not in ordered_skill_ids:
        raise ValueError(f"Target Skill result {target_tool_call_id!r} is absent")
    if len(ordered_skill_ids) != len(set(ordered_skill_ids)):
        raise ValueError("A Skill tool_call_id appears more than once")
    prompt_ids = render_prompt_token_ids(tokenizer, messages, tools)
    separator_ids = effective_separator_tokens(tokenizer, separator)
    occurrences = subsequence_starts(prompt_ids, separator_ids)
    expected = 2 * len(ordered_skill_ids)
    if len(occurrences) != expected:
        raise ValueError(
            f"Separator count mismatch: found={len(occurrences)} expected={expected}"
        )
    rank = ordered_skill_ids.index(target_tool_call_id)
    segment_start = occurrences[2 * rank] + len(separator_ids)
    segment_end = occurrences[2 * rank + 1] + len(separator_ids)
    if not 0 < segment_start < segment_end <= len(prompt_ids):
        raise ValueError(
            f"Invalid Skill span [{segment_start}, {segment_end}) / {len(prompt_ids)}"
        )
    return segment_start, segment_end, prompt_ids, separator_ids
