from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from cskcache.v1.metadata import (
    CSKCacheMode,
    CSKCacheSegment,
    SegmentOccurrence,
)


@dataclass
class SegmentCatalog:
    segments: list[CSKCacheSegment]

    @classmethod
    def from_json_file(cls, path: str | Path) -> "SegmentCatalog":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        raw_segments = payload.get("segments", payload)
        if not isinstance(raw_segments, list):
            raise ValueError("CSKCache catalog must be a list or {'segments': list}")
        segments: list[CSKCacheSegment] = []
        for item in raw_segments:
            token_ids = tuple(int(value) for value in item["token_ids"])
            if not token_ids:
                continue
            segments.append(
                CSKCacheSegment(
                    cache_id=str(item.get("cache_id", item.get("segment_id"))),
                    token_ids=token_ids,
                    mode=CSKCacheMode(item.get("mode", CSKCacheMode.ROPE_REUSE.value)),
                )
            )
        return cls(segments)

    def occurrences(self, token_ids: list[int] | tuple[int, ...]) -> list[SegmentOccurrence]:
        matches: list[SegmentOccurrence] = []
        for segment in self.segments:
            for start in find_subsequence(token_ids, segment.token_ids):
                matches.append(
                    SegmentOccurrence(
                        cache_id=segment.cache_id,
                        start=start,
                        end=start + segment.length,
                        mode=segment.mode,
                    )
                )
        matches.sort(key=lambda item: (item.start, -(item.end - item.start), item.cache_id))
        return matches


def find_subsequence(
    haystack: list[int] | tuple[int, ...],
    needle: list[int] | tuple[int, ...],
) -> Iterable[int]:
    """Yield all exact token-subsequence matches using KMP."""

    if not needle or len(needle) > len(haystack):
        return
    failure = [0] * len(needle)
    j = 0
    for i in range(1, len(needle)):
        while j and needle[i] != needle[j]:
            j = failure[j - 1]
        if needle[i] == needle[j]:
            j += 1
            failure[i] = j

    j = 0
    for i, token in enumerate(haystack):
        while j and token != needle[j]:
            j = failure[j - 1]
        if token == needle[j]:
            j += 1
            if j == len(needle):
                yield i - len(needle) + 1
                j = failure[j - 1]


def find_best_occurrence(
    catalog: SegmentCatalog,
    token_ids: list[int] | tuple[int, ...],
    num_computed_tokens: int,
) -> SegmentOccurrence | None:
    """Pick the next CSK occurrence relevant to the current scheduler state."""

    candidates = catalog.occurrences(token_ids)
    if not candidates:
        return None
    ready = [
        item
        for item in candidates
        if item.start == num_computed_tokens and item.end > num_computed_tokens
    ]
    if ready:
        return max(ready, key=lambda item: (item.length, item.cache_id))
    upcoming = [item for item in candidates if item.start > num_computed_tokens]
    if upcoming:
        return min(upcoming, key=lambda item: (item.start, -item.length, item.cache_id))
    covering = [
        item
        for item in candidates
        if item.start < num_computed_tokens < item.end
    ]
    if covering:
        return max(covering, key=lambda item: (item.end, item.length, item.cache_id))
    return None

