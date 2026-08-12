"""Build the context-segment token object cached for one Skill."""
from __future__ import annotations

from html import escape
from typing import Any


CACHE_SCHEMA_VERSION = 4
CACHE_OBJECT_TYPE = "qwen_context_segment"
LOCATOR_KIND = "context_segment_start_marker_v1"
CONTEXT_SEGMENT_CLOSE = "</context_segment>"


def context_segment_start_marker_text(skill_name: str) -> str:
    """Return the exact opening boundary used to locate one cached Skill."""
    escaped_name = escape(skill_name, quote=True)
    return f'<context_segment skill_name="{escaped_name}">\n'


def context_segment_text(skill_name: str, skill_text: str) -> str:
    """Wrap one SKILL.md body in the canonical online/offline boundary."""
    closing_newline = "" if skill_text.endswith("\n") else "\n"
    return context_segment_start_marker_text(
        skill_name
    ) + f"{skill_text}{closing_newline}{CONTEXT_SEGMENT_CLOSE}"


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


def qwen_context_segment_start_marker_token_ids(
    tokenizer: Any,
    skill_name: str,
) -> list[int]:
    """Tokenize the opening marker once for the offline locator manifest."""
    marker = context_segment_start_marker_text(skill_name)
    token_ids = tokenizer.encode(marker, add_special_tokens=False)
    if not token_ids:
        raise RuntimeError(f"empty context-segment start marker for {skill_name}")
    return token_ids
