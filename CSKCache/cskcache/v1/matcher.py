"""Backward-compatible shim.

The token matcher now lives in :mod:`cskcache.v1.token.token_database` as part
of the token layer. This module re-exports the public names so existing imports
(``from cskcache.v1.matcher import SegmentCatalog``) keep working.
"""

from cskcache.v1.token.token_database import (
    SegmentCatalog,
    find_best_occurrence,
    find_subsequence,
)

__all__ = ["SegmentCatalog", "find_best_occurrence", "find_subsequence"]
