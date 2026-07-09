from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class CSKCacheMode(str, Enum):
    """High-level policy for a cached segment once it is found in a request."""

    DISABLED = "disabled"
    REUSE = "reuse"


class CSKCacheDirectivePlacement(str, Enum):
    """How an agent-provided request directive locates the current skill span.

    EXPLICIT_SPAN:
        The agent or prompt builder already tokenized the final prompt and
        passes the exact target [start, end) token offsets for the current
        skill. This is the preferred path when the caller can observe token
        boundaries.

    SUFFIX_BEFORE_TRAILING:
        The skill is the final semantic payload added by the agent, but the
        chat template adds a few trailing tokens after it, such as an assistant
        marker, separator, or tool wrapper. The caller passes only how many
        trailing tokens exist; CSKCache derives the span from the loaded entry
        length and then verifies the token slice exactly.
    """

    EXPLICIT_SPAN = "explicit_span"
    SUFFIX_BEFORE_TRAILING = "suffix_before_trailing"


@dataclass(frozen=True)
class CSKCacheRequestDirective:
    """Per-request instruction sent by the agent through vLLM extra args.

    The OpenAI-compatible request carries this under
    kv_transfer_params["cskcache"]. It is intentionally about the *current*
    skill only. Historical skill spans in a multi-turn prompt should be
    inherited through vLLM prefix cache if they were computed in prior requests;
    CSKCache should not scan for and explicitly inject them again.

    enabled=False means "do not use CSKCache for this request". It is not the
    same as omitting the directive: omission falls back to token matching,
    while enabled=False suppresses that fallback.
    """

    enabled: bool
    cache_id: str
    placement: CSKCacheDirectivePlacement = CSKCacheDirectivePlacement.EXPLICIT_SPAN
    target_start: int | None = None
    target_end: int | None = None
    trailing_token_count: int = 0


@dataclass(frozen=True)
class CSKCacheSegment:
    """Canonical segment token sequence known to CSKCache.

    These are lightweight index records derived from loaded CSKCacheEntry
    objects. They do not contain tensors; they only say "cache_id X represents
    this exact token sequence". When a request has no explicit directive,
    SegmentCatalog uses these records to find exact token-subsequence matches.

    With request directives, this catalog is no longer the primary discovery
    source for the current skill, but the same token identity still matters:
    CSKCache only reuses KV when the target request span exactly matches the
    cached entry token IDs.
    """

    cache_id: str
    token_ids: tuple[int, ...]
    mode: CSKCacheMode = CSKCacheMode.REUSE

    @property
    def length(self) -> int:
        return len(self.token_ids)


@dataclass(frozen=True)
class SegmentOccurrence:
    """A concrete occurrence of a cached segment inside one request prompt.

    start/end are target token offsets in the *current request*, not source
    offsets inside the offline cache entry. For a full-entry reuse they usually
    correspond to source_offset=0 in CSKLoadPlan; probe/anchor tail loads may
    use a non-zero source_offset while still targeting request-local offsets.
    """

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
    
    This is the durable payload loaded from disk or inserted into the registry.
    token_ids define the canonical source sequence. kv_by_layer stores the
    cached K/V tensors by vLLM layer name; each layer maps to (key, value)
    tensors whose leading dimension is the token dimension.
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

    The scheduler creates this after it has decided that a request span should
    be satisfied from CSKCache. The worker later consumes the plan in
    start_load_kv(), slices the offline entry at source_offset, applies RoPE
    correction to keys if needed, and scatters the result into the target
    request slots [start, end).

    source_offset is measured in the cached entry. It is zero for full skill
    reuse, and non-zero when probe-gated execution has already recomputed an
    initial prefix and only the remaining tail should be loaded.
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
