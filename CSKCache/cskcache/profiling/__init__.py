"""Optional, request-scoped profiling for CSKCache data movement.

The profiling package owns timing, aggregation, and reporting. Core modules
only create a trace and place thin stage markers around existing operations.
When disabled, :class:`NullLoadTrace` turns those markers into no-ops.
"""

from cskcache.profiling.config import ProfileConfig
from cskcache.profiling.reporter import ProfileReporter
from cskcache.profiling.trace import LoadTrace, NullLoadTrace, Profiler

__all__ = [
    "LoadTrace",
    "NullLoadTrace",
    "ProfileConfig",
    "ProfileReporter",
    "Profiler",
]
