from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class CSKCacheMode(str, Enum):
    DISABLED = "disabled"
    REUSE = "reuse"


@dataclass(frozen=True)
class CSKCacheSegment:
    """Canonical segment token sequence known to CSKCache.

    First version uses token catalog matching.  TODO(B): allow an upstream
    prompt builder to pass structured segment metadata, then use token matching
    only as a verification step instead of the primary discovery mechanism.
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
    req_id: str
    cache_id: str
    mode: CSKCacheMode
    start: int
    end: int
    token_ids: tuple[int, ...]

    @property
    def length(self) -> int:
        return self.end - self.start

