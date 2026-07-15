from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cskcache.profiling import LoadTrace, NullLoadTrace
    from cskcache.v1.metadata import CSKCacheEntry
    from cskcache.v1.storage.storage_manager import StorageManager

# Bounded so a burst of reuse signals can't spawn unbounded disk threads;
# sized well above the number of skills one request realistically reuses at
# once. Lazily created so importing this module never starts threads.
_MAX_WORKERS = 4
_executor: ThreadPoolExecutor | None = None


def _get_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(
            max_workers=_MAX_WORKERS, thread_name_prefix="cskcache-disk-prefetch"
        )
    return _executor


@dataclass
class PrefetchHandle:
    """A ``StorageManager.get()`` call that may already be running.

    ``result()`` has the same success/exception semantics as calling
    ``storage.get()`` directly and blocking on it -- an exception raised on
    the background thread is re-raised here, not swallowed. The only
    difference from a direct call is that the work may have already
    finished by the time you ask, so the wait is often zero.
    """

    _future: "Future[CSKCacheEntry | None]"

    def result(self, timeout: float | None = None) -> "CSKCacheEntry | None":
        return self._future.result(timeout=timeout)

    def is_ready(self) -> bool:
        return self._future.done()


def submit_disk_prefetch(
    storage: "StorageManager",
    cache_id: str,
    trace: "LoadTrace | NullLoadTrace | None" = None,
) -> PrefetchHandle:
    """Start ``storage.get(cache_id, trace=trace)`` on a background thread now.

    Callers own the ``trace`` object's lifetime: pass one dedicated to this
    call, not a trace another concurrent operation might also be writing to,
    since ``LoadTrace`` itself has no internal locking.
    """
    future = _get_executor().submit(storage.get, cache_id, trace=trace)
    return PrefetchHandle(_future=future)
