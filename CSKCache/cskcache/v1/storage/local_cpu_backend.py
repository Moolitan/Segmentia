from __future__ import annotations

from typing import Iterable

from cskcache.profiling import LoadTrace, NullLoadTrace
from cskcache.v1.metadata import CSKCacheEntry
from cskcache.v1.storage.abstract_backend import (
    StorageBackendInterface,
    entry_nbytes,
)


class LocalCPUBackend(StorageBackendInterface):
    """In-RAM tier holding fully decoded ``CSKCacheEntry`` objects.

    This is the hot tier and the direct generalization of the original
    ``CSKCacheRegistry._entries`` dict. It keeps a running byte total so the
    storage manager can enforce a CPU budget without re-summing every put.
    """

    def __init__(self) -> None:
        self._entries: dict[str, CSKCacheEntry] = {}
        self._nbytes: dict[str, int] = {}
        self._total_bytes = 0

    def contains(self, cache_id: str) -> bool:
        return cache_id in self._entries

    def get(
        self,
        cache_id: str,
        trace: LoadTrace | NullLoadTrace | None = None,
    ) -> CSKCacheEntry | None:
        return self._entries.get(cache_id)

    def put(self, entry: CSKCacheEntry) -> None:
        # Overwrite semantics: drop the old byte count before adding the new one
        # so re-putting the same cache_id does not double-count.
        if entry.cache_id in self._entries:
            self._total_bytes -= self._nbytes.get(entry.cache_id, 0)
        size = entry_nbytes(entry)
        self._entries[entry.cache_id] = entry
        self._nbytes[entry.cache_id] = size
        self._total_bytes += size

    def remove(self, cache_id: str) -> bool:
        if cache_id not in self._entries:
            return False
        del self._entries[cache_id]
        self._total_bytes -= self._nbytes.pop(cache_id, 0)
        return True

    def keys(self) -> Iterable[str]:
        return tuple(self._entries.keys())

    def size_bytes(self) -> int:
        return self._total_bytes
