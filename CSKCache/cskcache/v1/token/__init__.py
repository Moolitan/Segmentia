"""CSKCache token layer: token sequence -> reusable span.

Maps request token ids to cached segments for offline diagnostics. Production
reuse decisions are driven by explicit request reuse signals, not prompt scans.
"""

from cskcache.v1.token.token_database import (
    SegmentCatalog,
    find_best_reuse,
    find_subsequence,
)

__all__ = ["SegmentCatalog", "find_best_reuse", "find_subsequence"]
