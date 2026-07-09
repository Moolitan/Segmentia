"""CSKCache token layer: token sequence -> reusable segment occurrence.

Maps request token ids to cached segments. ``SegmentCatalog`` is the exact
(KMP) token-subsequence index used as the fallback when a request carries no
agent directive; the engine's directive path is the preferred discovery route.
"""

from cskcache.v1.token.token_database import (
    SegmentCatalog,
    find_best_occurrence,
    find_subsequence,
)

__all__ = ["SegmentCatalog", "find_best_occurrence", "find_subsequence"]
