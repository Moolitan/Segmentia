"""Prompt marking and stable skill segment extraction helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Iterable, Sequence

from .policy import SkillSegment


@dataclass(frozen=True)
class SkillMarkerConfig:
    """Text markers used by an Agent prompt builder to expose skill boundaries."""

    begin_template: str = '<SKILL id="{skill_id}" type="{skill_type}" version="{version_hash}">'
    end_marker: str = "</SKILL>"

    def begin_marker(
        self,
        *,
        skill_id: str,
        skill_type: str,
        version_hash: str,
    ) -> str:
        return self.begin_template.format(
            skill_id=skill_id,
            skill_type=skill_type,
            version_hash=version_hash,
        )


_SKILL_BLOCK_RE = re.compile(
    r'<SKILL\s+id="(?P<skill_id>[^"]+)"\s+type="(?P<skill_type>[^"]+)"\s+version="(?P<version_hash>[^"]+)">'
    r"(?P<body>.*?)"
    r"</SKILL>",
    flags=re.DOTALL,
)


def stable_text_hash(text: str, *, prefix_len: int = 16) -> str:
    """Return a stable short hash for skill text or versions."""

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[:prefix_len]


def stable_token_hash(token_ids: Sequence[int], *, prefix_len: int = 16) -> str:
    """Return a stable hash for tokenizer ids."""

    raw = ",".join(str(int(tok)) for tok in token_ids)
    return stable_text_hash(raw, prefix_len=prefix_len)


def mark_skill_text(
    *,
    skill_id: str,
    skill_text: str,
    skill_type: str = "generic",
    version_hash: str | None = None,
    marker_config: SkillMarkerConfig | None = None,
) -> str:
    """Wrap skill text with explicit markers for downstream segment lookup."""

    marker_config = marker_config or SkillMarkerConfig()
    version_hash = version_hash or stable_text_hash(skill_text)
    return (
        marker_config.begin_marker(
            skill_id=skill_id,
            skill_type=skill_type,
            version_hash=version_hash,
        )
        + "\n"
        + skill_text
        + "\n"
        + marker_config.end_marker
    )


def extract_marked_skill_segments(
    *,
    prompt_text: str,
    input_ids: Sequence[int],
    offset_mapping: Sequence[tuple[int, int]],
) -> list[SkillSegment]:
    """Extract SkillSegment objects from a tokenized marked prompt.

    The caller should pass offsets from a fast tokenizer with
    ``return_offsets_mapping=True``. Token ranges include only the skill body,
    not the surrounding marker text.
    """

    if len(input_ids) != len(offset_mapping):
        raise ValueError("input_ids and offset_mapping must have the same length")

    segments: list[SkillSegment] = []
    for match in _SKILL_BLOCK_RE.finditer(prompt_text):
        body_start, body_end = match.span("body")
        token_start, token_end = char_span_to_token_span(
            offset_mapping,
            body_start,
            body_end,
        )
        token_hash = stable_token_hash(input_ids[token_start:token_end])
        segments.append(
            SkillSegment(
                skill_id=match.group("skill_id"),
                version_hash=match.group("version_hash"),
                token_hash=token_hash,
                token_range=(token_start, token_end),
                skill_type=match.group("skill_type"),
                length=token_end - token_start,
            )
        )
    return segments


def char_span_to_token_span(
    offset_mapping: Sequence[tuple[int, int]],
    char_start: int,
    char_end: int,
) -> tuple[int, int]:
    """Map a character span to a token span using tokenizer offsets."""

    tok_start: int | None = None
    tok_end: int | None = None
    for idx, (start, end) in enumerate(offset_mapping):
        if end <= char_start:
            continue
        if start >= char_end:
            break
        if tok_start is None:
            tok_start = idx
        tok_end = idx + 1

    if tok_start is None or tok_end is None or tok_end <= tok_start:
        raise ValueError(f"failed to map char span [{char_start}, {char_end})")
    return tok_start, tok_end


def iter_marked_skill_blocks(prompt_text: str) -> Iterable[re.Match[str]]:
    """Expose raw regex matches for debugging and report generation."""

    return _SKILL_BLOCK_RE.finditer(prompt_text)
