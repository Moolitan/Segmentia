"""Token-level reuse matching helpers.

The token matcher now lives in :mod:`cskcache.v1.token.token_database` as part
of the token layer. The engine no longer uses prompt scanning for production
reuse decisions; these helpers are retained for offline diagnostics.
"""

from cskcache.v1.token.token_database import (
    SegmentCatalog,
    find_best_reuse,
    find_subsequence,
)

__all__ = ["SegmentCatalog", "find_best_reuse", "find_subsequence"]
