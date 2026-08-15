"""Canonical Context Segment wire format shared by build and serving paths."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape, unescape
import re


_OPENING = re.compile(r'^<context_segment skill_name="([^"]+)">\n')
_CLOSING = "</context_segment>"


@dataclass(frozen=True)
class ParsedContextSegment:
    """One validated Context Segment and any Tool-result suffix after it."""

    skill_name: str
    skill_text: str
    trailing_text: str


def render_context_segment(skill_name: str, skill_text: str) -> str:
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


def parse_context_segment(observation: str) -> ParsedContextSegment | None:
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
    return ParsedContextSegment(
        skill_name=unescape(opening.group(1)),
        skill_text=observation[opening.end() : closing_start],
        trailing_text=observation[closing_end:],
    )
