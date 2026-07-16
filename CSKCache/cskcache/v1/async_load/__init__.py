"""Background prefetch primitives, decoupled from probe/anchor scheduling.

This subpackage only knows how to run an existing blocking call (like
``StorageManager.get()``) on a background thread and hand back a small
handle. It has no knowledge of ``CSKReuseState``, reuse signals, or any other
scheduling concept in ``cskcache.v1.core`` — callers decide when to submit
work and when to block on the result, keeping this module swappable or
removable without touching the engine's own logic.
"""

from cskcache.v1.async_load.disk_prefetch import PrefetchHandle, submit_disk_prefetch
from cskcache.v1.async_load.gpu_prefetch import GpuPrefetchHandle, submit_gpu_prefetch
from cskcache.v1.async_load.prefetch_registry import PrefetchRegistry

__all__ = [
    "PrefetchHandle",
    "submit_disk_prefetch",
    "GpuPrefetchHandle",
    "submit_gpu_prefetch",
    "PrefetchRegistry",
]
