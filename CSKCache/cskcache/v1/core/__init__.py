"""CSKCache core engine layer (vLLM-agnostic).

This subpackage holds the brain of the middleware: configuration, the
probe-gated state machine, and the cache engine that orchestrates lookup, load
planning, reuse, and gating. Nothing here imports vLLM, so the engine can be
constructed and unit-tested as a plain library.
"""

from cskcache.v1.core.cache_engine import CSKCacheEngine
from cskcache.v1.core.config import CSKCacheConfig
from cskcache.v1.core.probe_state import CSKProbePhase, CSKProbeState

__all__ = [
    "CSKCacheConfig",
    "CSKCacheEngine",
    "CSKProbePhase",
    "CSKProbeState",
]
