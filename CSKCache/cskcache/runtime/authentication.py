"""Exact and logical-chunk prefix authentication for online Skill spans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..metadata.base import CacheObjectMetadata
from ..metadata.fingerprint import fingerprint_token_ids
from .base import SkillMatchMode


@dataclass(frozen=True)
class AuthenticatedSkillPrefix:
    """Newest Skill occurrence and the causally reusable prefix it proves."""

    segment_start: int
    segment_end: int
    match_mode: SkillMatchMode
    matched_chunk_count: int


def locate_authenticated_skill_prefix(
    prompt_token_ids: Sequence[int],
    cache_object: CacheObjectMetadata,
) -> AuthenticatedSkillPrefix | None:
    """Authenticate the newest marker occurrence, exact first then by chunk.

    Chunk digests are independent identities rather than a parent hash chain.
    They are nevertheless compared strictly from chunk zero and matching stops
    at the first difference, because later KV depends on every earlier token.
    """

    start = _find_newest_marker(prompt_token_ids, cache_object.start_marker_token_ids)
    if start is None:
        return None

    exact_end = start + cache_object.token_count
    if exact_end <= len(prompt_token_ids) and (
        fingerprint_token_ids(prompt_token_ids[start:exact_end])
        == cache_object.token_ids_sha256
    ):
        chunk_size = cache_object.chunking.chunk_size_tokens
        chunk_count = (cache_object.token_count + chunk_size - 1) // chunk_size
        return AuthenticatedSkillPrefix(
            segment_start=start,
            segment_end=exact_end,
            match_mode=SkillMatchMode.EXACT,
            matched_chunk_count=chunk_count,
        )

    chunk_size = cache_object.chunking.chunk_size_tokens
    matched_chunk_count = 0
    for chunk_id, expected_digest in enumerate(
        cache_object.chunk_token_ids_sha256
    ):
        chunk_start = start + chunk_id * chunk_size
        chunk_end = chunk_start + chunk_size
        if chunk_end > len(prompt_token_ids):
            break
        if (
            fingerprint_token_ids(prompt_token_ids[chunk_start:chunk_end])
            != expected_digest
        ):
            break
        matched_chunk_count += 1

    if matched_chunk_count == 0:
        return None
    return AuthenticatedSkillPrefix(
        segment_start=start,
        segment_end=start + matched_chunk_count * chunk_size,
        match_mode=SkillMatchMode.PARTIAL_PREFIX,
        matched_chunk_count=matched_chunk_count,
    )


def _find_newest_marker(
    prompt_token_ids: Sequence[int],
    marker: tuple[int, ...],
) -> int | None:
    if len(prompt_token_ids) < len(marker):
        return None
    for start in range(len(prompt_token_ids) - len(marker), -1, -1):
        if tuple(prompt_token_ids[start : start + len(marker)]) == marker:
            return start
    return None
