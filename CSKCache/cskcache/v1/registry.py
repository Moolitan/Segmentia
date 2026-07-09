from __future__ import annotations

from pathlib import Path
import warnings

import torch

from cskcache.v1.metadata import CSKCacheEntry
from cskcache.v1.storage.local_disk_backend import payload_to_entry
from cskcache.v1.storage.storage_manager import StorageManager


class CSKCacheRegistry:
    """Backward-compatible facade over the tiered storage stack.

    Historically this class *was* the storage: a plain dict of entries. It now
    delegates to a :class:`StorageManager`, so existing callers keep the same
    ``put/get/entries/load_file/load_dir`` API while transparently gaining the
    CPU+disk tiering. Constructed with no arguments it is a pure in-memory store,
    identical in behavior to the original registry.
    """

    def __init__(self, storage: StorageManager | None = None) -> None:
        self._storage = storage if storage is not None else StorageManager()

    @property
    def storage(self) -> StorageManager:
        return self._storage

    def put(self, entry: CSKCacheEntry) -> None:
        self._storage.put(entry)

    def get(self, cache_id: str) -> CSKCacheEntry | None:
        return self._storage.get(cache_id)

    def entries(self) -> tuple[CSKCacheEntry, ...]:
        return self._storage.all_entries()

    def __contains__(self, cache_id: str) -> bool:
        return self._storage.contains(cache_id)

    def load_file(
        self,
        path: str | Path,
        device: torch.device | str | None = None,
    ) -> CSKCacheEntry:
        payload = torch.load(path, map_location=device or "cpu")
        entry = payload_to_entry(payload, device=device)
        self.put(entry)
        return entry

    def load_dir(
        self,
        path: str | Path,
        device: torch.device | str | None = None,
    ) -> list[str]:
        root = Path(path)
        loaded: list[str] = []
        if not root.exists():
            return loaded
        for file in sorted(root.glob("*.pt")):
            try:
                loaded.append(self.load_file(file, device=device).cache_id)
            except Exception as exc:
                warnings.warn(f"CSKCache load skipped {file.name}: {exc}")
        return loaded


_GLOBAL_REGISTRY: CSKCacheRegistry | None = None


def get_global_registry() -> CSKCacheRegistry:
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = CSKCacheRegistry()
    return _GLOBAL_REGISTRY
