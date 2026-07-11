from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from cskcache.v1.metadata import (
    CSKCacheEntry,
    CSKCacheMode,
    CSKCacheSegment,
    ReuseSpan,
)


@dataclass
class SegmentCatalog:
    """In-memory exact-token index over loaded CSKCache entries.

    This catalog is deliberately simple: it only knows token IDs and cache IDs.
    It does not inspect text, skill names, or agent state. The engine does not
    use this catalog to decide production reuse; it remains a readable
    token-level utility for offline tests and diagnostics.
    """

    segments: list[CSKCacheSegment]

    @classmethod
    def from_entries(cls, entries: Iterable[CSKCacheEntry]) -> "SegmentCatalog":
        """Build matchable segment records from loaded tensor entries.

        The registry owns the actual K/V tensors. The catalog keeps only the
        canonical token sequence for each cache_id so scheduler-side matching
        stays lightweight and serializable.
        """

        segments: list[CSKCacheSegment] = []
        for entry in entries:
            token_ids = tuple(int(value) for value in entry.token_ids)
            if not token_ids:
                continue
            segments.append(
                CSKCacheSegment(
                    cache_id=entry.cache_id,
                    token_ids=token_ids,
                    mode=CSKCacheMode.REUSE,
                )
            )
        return cls(segments)

    def reuse_spans(self, token_ids: list[int] | tuple[int, ...]) -> list[ReuseSpan]:
        """Return every exact catalog segment match in the request tokens.

        The function intentionally performs exact token matching. KV tensors
        are tied to token IDs and position; a semantically similar text span is
        not a match unless it tokenizes to the same sequence.
        """
        matches: list[ReuseSpan] = []
        for segment in self.segments:
            for start in find_subsequence(token_ids, segment.token_ids):
                matches.append(
                    ReuseSpan(
                        cache_id=segment.cache_id,
                        start=start,
                        end=start + segment.length,
                        mode=segment.mode,
                    )
                )
        # Stable ordering makes scheduler decisions deterministic:
        # earliest target span first, longer segment first when spans share a
        # start, then cache_id as a tie breaker.
        matches.sort(key=lambda item: (item.start, -(item.end - item.start), item.cache_id))
        return matches


def find_subsequence(
    haystack: list[int] | tuple[int, ...],
    needle: list[int] | tuple[int, ...],
) -> Iterable[int]:
    """Yield all exact token-subsequence matches using KMP.

    KMP avoids rechecking token prefixes after a mismatch, which matters when a
    long prompt contains repeated skill-like substrings. It still returns every
    reuse span, including overlapping ones, for offline diagnostics.
    """

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


def find_best_reuse(
    catalog: SegmentCatalog,
    token_ids: list[int] | tuple[int, ...],
    num_computed_tokens: int,
) -> ReuseSpan | None:
    """Pick the next reusable span relevant to a scheduler frontier.

    num_computed_tokens is vLLM's current frontier for this request. The
    returned span is interpreted as follows:

    - ready: start == num_computed_tokens, so CSKCache can load immediately.
    - upcoming: start > num_computed_tokens, so scheduler should first prefill
      up to start before loading.
    - covering: start < num_computed_tokens < end, which means the scheduler has
      already entered the span. Current code does not inject from the middle in
      this diagnostic matcher.
    """
    candidates = catalog.reuse_spans(token_ids)
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
