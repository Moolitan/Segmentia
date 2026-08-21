"""Canonical Skill Tool-result wire format shared by build and serving."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape, unescape
from typing import Any
import re

from .base import SKILL_PAYLOAD_FORMAT, SkillTokenIdentity
from .fingerprint import fingerprint_token_ids


_OPENING = re.compile(r'^<context_segment skill_name="([^"]+)">\n')
_CLOSING = "</context_segment>"


@dataclass(frozen=True)
class ParsedSkillPayload:
    """One validated Skill payload and any Tool-result suffix after it."""

    skill_name: str
    skill_text: str
    trailing_text: str


def render_skill_payload(skill_name: str, skill_text: str) -> str:
    """Render the exact Context Segment placed in a Skill Tool result."""

    if not isinstance(skill_name, str) or not skill_name.strip():
        raise ValueError("skill_name must be a non-empty string")
    if not isinstance(skill_text, str):
        raise TypeError("skill_text must be a string")
    escaped_name = escape(skill_name, quote=True)
    body_suffix = "" if skill_text.endswith("\n") else "\n"
    return (
        f'<context_segment skill_name="{escaped_name}">\n'
        f"{skill_text}{body_suffix}{_CLOSING}"
    )


def build_skill_token_identity(
    tokenizer: Any,
    skill_name: str,
    skill_text: str,
) -> SkillTokenIdentity:
    """Tokenize the exact Context Segment object used offline and online."""

    observation_text = render_skill_payload(skill_name, skill_text)
    cache_text = observation_text + "\n"
    token_ids = tuple(
        int(token_id)
        for token_id in tokenizer.encode(cache_text, add_special_tokens=False)
    )
    if not token_ids:
        raise RuntimeError(f"empty Context Segment token sequence for {skill_name}")
    opening_end = observation_text.find("\n") + 1
    if opening_end <= 0:
        raise RuntimeError("rendered Context Segment has no opening boundary")
    start_marker_text = observation_text[:opening_end]
    start_marker_token_ids = tuple(
        int(token_id)
        for token_id in tokenizer.encode(
            start_marker_text,
            add_special_tokens=False,
        )
    )
    if not start_marker_token_ids:
        raise RuntimeError(f"empty Context Segment marker for {skill_name}")
    return SkillTokenIdentity(
        payload_format=SKILL_PAYLOAD_FORMAT,
        observation_text=observation_text,
        cache_text=cache_text,
        token_ids=token_ids,
        token_ids_sha256=fingerprint_token_ids(token_ids),
        start_marker_text=start_marker_text,
        start_marker_token_ids=start_marker_token_ids,
        start_marker_token_ids_sha256=fingerprint_token_ids(
            start_marker_token_ids
        ),
    )


def parse_skill_payload(observation: str) -> ParsedSkillPayload | None:
    """Parse the unique leading Context Segment in one Skill Tool result."""

    if not isinstance(observation, str):
        return None
    opening = _OPENING.match(observation)
    if opening is None or len(_OPENING.findall(observation)) != 1:
        return None
    if observation.count(_CLOSING) != 1:
        return None
    closing_start = observation.find(_CLOSING, opening.end())
    if closing_start < 0:
        return None
    closing_end = closing_start + len(_CLOSING)
    return ParsedSkillPayload(
        skill_name=unescape(opening.group(1)),
        skill_text=observation[opening.end() : closing_start],
        trailing_text=observation[closing_end:],
    )
