"""CSKCache: a context-skill KV cache middleware for LLM serving.

CSKCache reuses per-skill KV cache across requests. Its distinctive mechanism is
*probe-gated selective recompute*: before trusting an offline skill KV, it lets
the engine recompute a short probe prefix, compares it to the cached KV, and
only then loads the tail (or recomputes a longer anchor first).

The package is layered like a standalone cache middleware and is importable
without vLLM:

- ``core``        engine + config + probe state machine (the brain)
- ``token``       token-sequence -> reusable span diagnostics
- ``kv_transfer`` scatter/gather against paged KV cache
- ``storage``     CPU + disk tiers with LRU eviction
- ``compute``     probe-gate residual + decision (the signature mechanism)
- ``profiling``   optional request-scoped CPU/CUDA timing and reporting

vLLM integration lives in ``cskcache.integration.vllm`` and is the only part
that imports vLLM.
"""

from cskcache.v1.compute import CSKProbeDecision
from cskcache.v1.core import CSKCacheConfig, CSKCacheEngine
from cskcache.v1.kv_transfer import KVConnectorInterface, VLLMPagedGPUConnector
from cskcache.v1.metadata import CSKCacheEntry
from cskcache.v1.registry import CSKCacheRegistry, get_global_registry
from cskcache.v1.storage import (
    LocalCPUBackend,
    LocalDiskBackend,
    StorageManager,
)

__all__ = [
    "CSKCacheConfig",
    "CSKCacheEngine",
    "CSKCacheEntry",
    "CSKCacheRegistry",
    "CSKProbeDecision",
    "KVConnectorInterface",
    "LocalCPUBackend",
    "LocalDiskBackend",
    "StorageManager",
    "VLLMPagedGPUConnector",
    "get_global_registry",
]
