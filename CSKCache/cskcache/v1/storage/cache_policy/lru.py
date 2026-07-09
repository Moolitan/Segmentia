from __future__ import annotations

from collections import OrderedDict


class LRUCachePolicy:
    """Least-recently-used ordering over cache_ids.

    The policy only tracks *order*, never the entries themselves. The storage
    manager records inserts and accesses here, then asks for the coldest
    cache_id when a tier is over budget. Keeping the policy free of tensor state
    makes it trivial to unit test and to swap for LFU/FIFO later.
    """

    def __init__(self) -> None:
        # Maps cache_id -> None, ordered from least- to most-recently used.
        self._order: "OrderedDict[str, None]" = OrderedDict()

    def record_access(self, cache_id: str) -> None:
        """Mark an entry as just used, moving it to the most-recent end."""

        if cache_id in self._order:
            self._order.move_to_end(cache_id)
        else:
            self._order[cache_id] = None

    # Inserting a fresh entry is treated as an access: it is the most recent.
    record_insert = record_access

    def record_remove(self, cache_id: str) -> None:
        """Forget an entry that has left the tier this policy governs."""

        self._order.pop(cache_id, None)

    def evict_candidate(self) -> str | None:
        """Return the least-recently-used cache_id without removing it."""

        for cache_id in self._order:
            return cache_id
        return None

    def __len__(self) -> int:
        return len(self._order)


_POLICIES = {
    "lru": LRUCachePolicy,
}


def get_cache_policy(name: str = "lru") -> LRUCachePolicy:
    """Factory for named eviction policies.

    Only LRU exists today. The factory keeps the call site stable so adding
    LFU/MRU/FIFO later is a one-line registration, matching how LMCache exposes
    its ``cache_policy`` package.
    """

    key = name.strip().lower()
    if key not in _POLICIES:
        raise ValueError(
            f"CSKCache unknown cache policy {name!r}; available: {sorted(_POLICIES)}"
        )
    return _POLICIES[key]()
