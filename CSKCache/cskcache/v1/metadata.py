from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class CSKCacheMode(str, Enum):
    """High-level policy for a cached segment once it is found in a request."""

    DISABLED = "disabled"
    REUSE = "reuse"


@dataclass(frozen=True)
class CSKCacheReuseSignal:
    """Per-request reuse control signal sent through vLLM extra args.

    The OpenAI-compatible request carries this under
    kv_transfer_params["cskcache"]. One request may carry multiple instances of
    this per-entry signal under ``entries``. Historical skill spans in a
    multi-turn prompt should be inherited through vLLM prefix cache if they were
    computed in prior requests; CSKCache should not scan for and explicitly
    inject them again.

    enabled=False means "do not use CSKCache for this request". It is not the
    same as omitting the signal: both cases simply continue normal prefill.
    """

    enabled: bool
    cache_id: str
    target_start: int | None = None
    target_end: int | None = None


@dataclass(frozen=True)
class CSKCacheSegment:
    """Canonical segment token sequence known to CSKCache.

    These are lightweight index records derived from loaded CSKCacheEntry
    objects. They do not contain tensors; they only say "cache_id X represents
    this exact token sequence". The engine no longer scans prompts with this
    catalog to decide reuse; current reuse decisions come from explicit
    per-request reuse signals.
    """

    cache_id: str
    token_ids: tuple[int, ...]
    mode: CSKCacheMode = CSKCacheMode.REUSE

    @property
    def length(self) -> int:
        return len(self.token_ids)


@dataclass(frozen=True)
class ReuseSpan:
    """A concrete request-local span selected for KV reuse.

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


@dataclass(frozen=True)
class CSKReqMeta:
    """Worker instruction for a concrete K/V load.

    The scheduler has already allocated blocks for the target request span.
    ``block_ids`` is the physical slot mapping that lets the worker scatter
    cached K/V into the paged cache before any query depends on those positions.
    This is a plain (vLLM-free) carrier; the integration layer wraps a list of
    these into vLLM's ``KVConnectorMetadata`` envelope for serialization.
    """

    plan: CSKLoadPlan
    block_ids: tuple[list[int], ...]


@dataclass(frozen=True)
class CSKProbeMeta:
    """Worker instruction for gathering a recomputed probe span.

    The span ``[start, end)`` has just been scheduled as normal prefill. During
    the worker save hook, the freshly written K/V is gathered from the same
    slots and compared to the corresponding cached K/V slice to build a gate
    decision. Plain (vLLM-free) carrier, like :class:`CSKReqMeta`.
    """

    req_id: str
    cache_id: str
    start: int
    end: int
    source_offset: int
    block_ids: tuple[list[int], ...]
    tau: float
    gate_metric: str

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class CSKSaveMeta:
    """Worker instruction for persisting a freshly-prefilled token span."""

    req_id: str
    cache_id: str
    start: int
    end: int
    token_ids: tuple[int, ...]
    block_ids: tuple[list[int], ...]
    overwrite: bool = False

    @property
    def length(self) -> int:
        return self.end - self.start
