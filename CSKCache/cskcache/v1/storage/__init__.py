"""CSKCache multi-tier storage layer.

This subpackage generalizes the original in-memory ``CSKCacheRegistry`` into a
small, LMCache-style storage stack:

- ``StorageBackendInterface`` — the contract every tier implements.
- ``LocalCPUBackend`` — in-RAM tier holding decoded ``CSKCacheEntry`` objects.
- ``LocalDiskBackend`` — on-disk ``.pt`` tier with lazy load.
- ``StorageManager`` — tier router + eviction policy (CPU hot, disk cold).

The default configuration (``cpu_max_bytes=None``) keeps everything in CPU
memory, which reproduces the previous registry behavior exactly. Setting a CPU
budget turns on spill-to-disk so the number of cached skills can exceed RAM,
which is what enables larger-scale reuse experiments.
"""

from cskcache.v1.storage.abstract_backend import (
    StorageBackendInterface,
    entry_nbytes,
)
from cskcache.v1.storage.local_cpu_backend import LocalCPUBackend
from cskcache.v1.storage.local_disk_backend import LocalDiskBackend
from cskcache.v1.storage.storage_manager import StorageManager

__all__ = [
    "StorageBackendInterface",
    "entry_nbytes",
    "LocalCPUBackend",
    "LocalDiskBackend",
    "StorageManager",
]
