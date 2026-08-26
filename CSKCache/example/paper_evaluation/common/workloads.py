"""Frozen Skill prompt construction and logical-version mutations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


BLEND_SEPARATOR = " # # "


@dataclass(frozen=True)
class SkillWorkload:
    skill_name: str
    skill_path: Path
    task_id: str
    task_prompt: str
    rules: tuple[Mapping[str, Any], ...] = ()


def read_skill_text(skills_root: Path, skill_name: str) -> str:
    path = skills_root / skill_name / "SKILL.md"
    if not path.is_file():
        raise FileNotFoundError(f"Skill does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Skill is empty: {path}")
    return text


def render_segment(skill_name: str, skill_text: str) -> str:
    from cskcache import render_skill_payload

    return render_skill_payload(skill_name, skill_text)


def blended_segment(skill_name: str, skill_text: str) -> str:
    # CSKCache requires the Context Segment to be the leading Tool payload.
    # The trailing separator is accepted as trailing_text by its parser, while
    # CacheBlend uses it as the right chunk boundary.
    return f"{render_segment(skill_name, skill_text)}\n{BLEND_SEPARATOR}"


def skill_tool(skill_name: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "skill",
                "description": "Load one named Skill.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "enum": [skill_name]}
                    },
                    "required": ["name"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def target_messages(
    *,
    skill_name: str,
    skill_text: str,
    task_prompt: str,
    tool_call_id: str,
) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": f"{task_prompt}\n{BLEND_SEPARATOR}"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": "skill",
                        "arguments": json.dumps(
                            {"name": skill_name}, separators=(",", ":")
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "name": "skill",
            "tool_call_id": tool_call_id,
            "content": blended_segment(skill_name, skill_text),
        },
    ]


def mutate_tokens(
    token_ids: Sequence[int], mutation: str, position_ratio: float
) -> list[int]:
    if not token_ids:
        raise ValueError("cannot mutate an empty token sequence")
    if not 0.0 <= position_ratio <= 1.0:
        raise ValueError("mutation position must be in [0, 1]")
    result = list(token_ids)
    index = min(int(len(result) * position_ratio), len(result) - 1)
    if mutation == "exact":
        return result
    if mutation == "replace":
        original = result[index]
        replacement = next(
            (candidate for candidate in result if candidate != original), None
        )
        if replacement is None:
            raise ValueError("cannot replace a single-token repeated sequence")
        result[index] = replacement
        return result
    if mutation == "append":
        result.append(result[0])
        return result
    raise ValueError(f"unsupported mutation: {mutation}")


def longest_full_chunk_prefix(
    cached: Sequence[int], current: Sequence[int], chunk_tokens: int
) -> int:
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    complete_chunks = min(len(cached), len(current)) // chunk_tokens
    matched = 0
    for chunk_index in range(complete_chunks):
        start = chunk_index * chunk_tokens
        end = start + chunk_tokens
        if tuple(cached[start:end]) != tuple(current[start:end]):
            break
        matched += chunk_tokens
    return matched
