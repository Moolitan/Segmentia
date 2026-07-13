from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable

import torch

from cskcache.v1.metadata import CSKCacheEntry
from cskcache.v1.storage.abstract_backend import StorageBackendInterface
from cskcache.v1.storage.cache_policy import get_cache_policy
from cskcache.v1.storage.local_cpu_backend import LocalCPUBackend
from cskcache.v1.storage.local_disk_backend import LocalDiskBackend


class StorageManager:
    """Tier router and eviction policy over CPU (hot) and disk (cold).

    Responsibilities kept here (not in the backends):
    - route get/put/contains across tiers,
    - enforce a CPU byte budget by spilling the least-recently-used entry to
      disk, and
    - promote a disk hit back into CPU.

    With ``cpu_max_bytes=None`` (the default) nothing ever spills, so a manager
    with only a CPU tier behaves exactly like the original in-memory registry.
    Give it a CPU budget and a disk tier to let the working set exceed RAM.
    """

    def __init__(
        self,
        cpu_backend: StorageBackendInterface | None = None,
        disk_backend: StorageBackendInterface | None = None,
        cpu_max_bytes: int | None = None,
        policy: str = "lru",
    ) -> None:
        self._cpu = cpu_backend if cpu_backend is not None else LocalCPUBackend()
        self._disk = disk_backend
        self._cpu_max_bytes = cpu_max_bytes
        # The policy tracks recency for the CPU tier only; disk is the overflow.
        self._policy = get_cache_policy(policy)

    # ---- construction helpers -------------------------------------------

    @classmethod
    def with_disk(
        cls,
        disk_dir: str | Path,
        cpu_max_bytes: int | None = None,
        device: torch.device | str | None = None,
        policy: str = "lru",
    ) -> "StorageManager":
        """Build a CPU + disk manager rooted at ``disk_dir``."""

        return cls(
            cpu_backend=LocalCPUBackend(),
            disk_backend=LocalDiskBackend(disk_dir, device=device),
            cpu_max_bytes=cpu_max_bytes,
            policy=policy,
        )

    # ---- core operations -------------------------------------------------

    def contains(self, cache_id: str) -> bool:
        if self._cpu.contains(cache_id):
            return True
        return self._disk is not None and self._disk.contains(cache_id)

    def get(self, cache_id: str) -> CSKCacheEntry | None:
        entry = self._cpu.get(cache_id)
        if entry is not None:
            self._policy.record_access(cache_id)
            return entry
        if self._disk is None:
            return None
        entry = self._disk.get(cache_id)
        if entry is None:
            return None
        # Disk hit: promote into the hot tier so repeated reuse stays fast.
        self._cpu.put(entry)
        self._policy.record_access(cache_id)
        self._enforce_cpu_budget()
        return entry

    def put(self, entry: CSKCacheEntry, *, persist: bool = False) -> None:
        if persist:
            if self._disk is None:
                raise RuntimeError(
                    "CSKCache persist=True requires a configured disk backend"
                )
            self._disk.put(entry)
            if self._cpu_max_bytes == 0:
                return
        self._cpu.put(entry)
        self._policy.record_insert(entry.cache_id)
        self._enforce_cpu_budget()

    def remove(self, cache_id: str) -> bool:
        removed_cpu = self._cpu.remove(cache_id)
        removed_disk = self._disk.remove(cache_id) if self._disk is not None else False
        self._policy.record_remove(cache_id)
        return removed_cpu or removed_disk

    def keys(self) -> tuple[str, ...]:
        ids = set(self._cpu.keys())
        if self._disk is not None:
            ids.update(self._disk.keys())
        return tuple(ids)

    def all_entries(self) -> tuple[CSKCacheEntry, ...]:
        """Materialize every entry across tiers.

        The token matcher needs the token ids of all cached segments to build
        its index. On the default (CPU-only) path this is just the in-memory
        entries; when disk spill is active, cold entries are loaded on demand.
        """

        result: list[CSKCacheEntry] = []
        for cache_id in self._cpu.keys():
            entry = self._cpu.get(cache_id)
            if entry is not None:
                result.append(entry)
        if self._disk is not None:
            seen = set(self._cpu.keys())
            for cache_id in self._disk.keys():
                if cache_id in seen:
                    continue
                entry = self._disk.get(cache_id)
                if entry is not None:
                    result.append(entry)
        return tuple(result)

    def size_bytes(self, tier: str = "cpu") -> int:
        if tier == "cpu":
            return self._cpu.size_bytes()
        if tier == "disk":
            return self._disk.size_bytes() if self._disk is not None else 0
        raise ValueError(f"CSKCache unknown tier {tier!r}; expected 'cpu' or 'disk'")

    # ---- eviction --------------------------------------------------------

    def _enforce_cpu_budget(self) -> None:
        """Spill least-recently-used CPU entries to disk until under budget."""

        if self._cpu_max_bytes is None:
            return
        while self._cpu.size_bytes() > self._cpu_max_bytes:
            victim = self._policy.evict_candidate()
            if victim is None:
                break
            if self._disk is None:
                # No cold tier to spill into: dropping would lose data, so we
                # leave the CPU tier over budget and warn once per breach.
                warnings.warn(
                    "CSKCache CPU tier over budget but no disk backend to spill "
                    "into; keeping entries in memory"
                )
                break
            entry = self._cpu.get(victim)
            if entry is not None:
                self._disk.put(entry)
            self._cpu.remove(victim)
            self._policy.record_remove(victim)
