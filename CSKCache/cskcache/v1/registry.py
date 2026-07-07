from __future__ import annotations

from pathlib import Path
import warnings

import torch

from cskcache.v1.metadata import CSKCacheEntry


class CSKCacheRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, CSKCacheEntry] = {}

    def put(self, entry: CSKCacheEntry) -> None:
        self._entries[entry.cache_id] = entry

    def get(self, cache_id: str) -> CSKCacheEntry | None:
        return self._entries.get(cache_id)

    def __contains__(self, cache_id: str) -> bool:
        return cache_id in self._entries

    def load_file(
        self,
        path: str | Path,
        device: torch.device | str | None = None,
    ) -> CSKCacheEntry:
        payload = torch.load(path, map_location=device or "cpu")
        entry = CSKCacheEntry(
            cache_id=str(payload["cache_id"]),
            source_start=int(payload["source_start"]),
            source_end=int(payload["source_end"]),
            token_ids=[int(value) for value in payload.get("token_ids", [])],
            kv_by_layer={
                str(layer): (
                    key.to(device) if device else key,
                    value.to(device) if device else value,
                )
                for layer, (key, value) in payload["kv_by_layer"].items()
            },
        )
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

