from __future__ import annotations

import threading
from typing import Callable, Hashable

from cskcache.v1.async_load.disk_prefetch import PrefetchHandle


class PrefetchRegistry:
    """Thread-safe ``key -> PrefetchHandle`` store with dedup and one-shot pop.

    ``get_or_submit`` is idempotent: calling it twice with the same key while
    a handle is still registered returns the existing handle instead of
    starting a redundant background fetch. ``pop`` consumes a handle exactly
    once -- after that (or if nothing was ever submitted for that key) the
    caller gets ``None`` back and should fall back to its own synchronous
    path, which is always correct, just not accelerated.
    """

    def __init__(self) -> None:
        self._handles: dict[Hashable, PrefetchHandle] = {}
        self._lock = threading.Lock()

    def get_or_submit(
        self, key: Hashable, submit_fn: Callable[[], PrefetchHandle]
    ) -> PrefetchHandle:
        with self._lock:
            existing = self._handles.get(key)
            if existing is not None:
                return existing
            handle = submit_fn()
            self._handles[key] = handle
            return handle

    def pop(self, key: Hashable) -> PrefetchHandle | None:
        with self._lock:
            return self._handles.pop(key, None)
