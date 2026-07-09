from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class CSKCacheMode(str, Enum):
    DISABLED = "disabled"
    REUSE = "reuse"


class CSKCacheDirectivePlacement(str, Enum):
    EXPLICIT_SPAN = "explicit_span"
    SUFFIX_BEFORE_TRAILING = "suffix_before_trailing"


@dataclass(frozen=True)
class CSKCacheRequestDirective:
    enabled: bool
    cache_id: str
    placement: CSKCacheDirectivePlacement = CSKCacheDirectivePlacement.EXPLICIT_SPAN
    target_start: int | None = None
    target_end: int | None = None
    trailing_token_count: int = 0


@dataclass(frozen=True)
class CSKCacheSegment:
    """Canonical segment token sequence known to CSKCache.

    First version uses token matching against segment entries derived from the
    loaded KV registry. TODO(B): allow an upstream prompt builder to pass
    structured segment metadata, then use token matching only as a verification
    step instead of the primary discovery mechanism.
    """

    cache_id: str
    token_ids: tuple[int, ...]
    mode: CSKCacheMode = CSKCacheMode.REUSE

    @property
    def length(self) -> int:
        return len(self.token_ids)


@dataclass(frozen=True)
class SegmentOccurrence:
    cache_id: str
    start: int
    end: int
    mode: CSKCacheMode

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class CSKCacheEntry:
    """Offline KV cache entry for a segment of tokens. 
    
    This is the data structure that is stored in the cache and retrieved when a segment is reused.
    """
    cache_id: str
    source_start: int
    source_end: int
    token_ids: list[int]
    kv_by_layer: dict[str, tuple[torch.Tensor, torch.Tensor]]

    @property
    def length(self) -> int:
        return self.source_end - self.source_start


@dataclass(frozen=True)
class CSKLoadPlan:
    """Final plan for loading a segment from the cache. 

    This is the data structure that is returned by the cache when a segment is requested, 
    and it contains all the information needed to load the segment into the model.
    """
    req_id: str
    cache_id: str
    mode: CSKCacheMode
    start: int
    end: int
    token_ids: tuple[int, ...]
    source_offset: int = 0

    @property
    def length(self) -> int:
        return self.end - self.start
