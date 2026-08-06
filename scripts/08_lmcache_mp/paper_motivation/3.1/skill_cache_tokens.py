"""Build the context-segment token object cached for one Skill."""
from __future__ import annotations

from html import escape
from typing import Any


CACHE_SCHEMA_VERSION = 3
CACHE_OBJECT_TYPE = "qwen_context_segment"
CONTEXT_SEGMENT_CLOSE = "</context_segment>"


def context_segment_text(skill_name: str, skill_text: str) -> str:
    """Wrap one SKILL.md body in the canonical online/offline boundary."""
    escaped_name = escape(skill_name, quote=True)
    closing_newline = "" if skill_text.endswith("\n") else "\n"
    return (
        f'<context_segment skill_name="{escaped_name}">\n'
        f"{skill_text}{closing_newline}{CONTEXT_SEGMENT_CLOSE}"
    )


def context_segment_cache_text(skill_name: str, skill_text: str) -> str:
    """Return the exact cached text, including Qwen's boundary newline."""
    return context_segment_text(skill_name, skill_text) + "\n"


def qwen_context_segment_token_ids(
    tokenizer: Any,
    skill_name: str,
    skill_text: str,
) -> list[int]:
    """Return the segment plus Qwen's trailing content-boundary newline."""
    wrapped = context_segment_cache_text(skill_name, skill_text)
    token_ids = tokenizer.encode(wrapped, add_special_tokens=False)
    if not token_ids:
        raise RuntimeError(f"empty context-segment token sequence for {skill_name}")
    return token_ids
