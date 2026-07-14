from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from cskcache.profiling import LoadTrace, NullLoadTrace
from cskcache.v1.metadata import CSKCacheEntry


def entry_nbytes(entry: CSKCacheEntry) -> int:
    """Return the KV tensor footprint of an entry in bytes.

    Only the cached K/V tensors are counted; token id lists and small scalars
    are negligible next to the paged KV payload. This is what the storage
    manager uses to enforce a per-tier byte budget, so it must reflect the part
    of the entry that actually competes for memory.
    """

    total = 0
    for key, value in entry.kv_by_layer.values():
        total += key.numel() * key.element_size()
        total += value.numel() * value.element_size()
    return total


class StorageBackendInterface(ABC):
    """Contract for a single storage tier.

    Backends are intentionally "dumb" stores: they only put, get, and report
    size for a ``cache_id``. All tiering and eviction decisions live in
    ``StorageManager`` so a backend can be swapped without changing the policy,
    mirroring the LMCache split between ``StorageManager`` and its backends.
    """

    @abstractmethod
    def contains(self, cache_id: str) -> bool:
        """Whether this tier currently holds the entry."""

    @abstractmethod
    def get(
        self,
        cache_id: str,
        trace: LoadTrace | NullLoadTrace | None = None,
    ) -> CSKCacheEntry | None:
        """Return the entry, or None if this tier does not hold it."""

    @abstractmethod
    def put(self, entry: CSKCacheEntry) -> None:
        """Store (or overwrite) an entry in this tier."""

    @abstractmethod
    def remove(self, cache_id: str) -> bool:
        """Drop an entry from this tier. Returns True if something was removed."""

    @abstractmethod
    def keys(self) -> Iterable[str]:
        """Iterate the cache_ids currently held by this tier."""

    @abstractmethod
    def size_bytes(self) -> int:
        """Total KV footprint held by this tier, in bytes."""

    def __contains__(self, cache_id: str) -> bool:
        return self.contains(cache_id)
