"""Pinned-memory contracts that consume the shared chunk layout plan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..chunking import SkillChunkPlan
from ..layouts import KVLayout, KVLayoutPlan


@dataclass(frozen=True)
class HostMemorySpec:
    """Host layout selected independently from persistent storage format."""

    layout: KVLayout

    def __post_init__(self) -> None:
        object.__setattr__(self, "layout", KVLayout(self.layout))


@dataclass(frozen=True)
class HostMemoryPlan:
    """Pinned allocation and H2D coordinates for one Skill."""

    spec: HostMemorySpec
    layout_plan: KVLayoutPlan

    def __post_init__(self) -> None:
        if self.spec.layout is not self.layout_plan.layout:
            raise ValueError("host-memory spec differs from its layout plan")


@dataclass(frozen=True)
class ChunkLayerBuffer:
    """Pinned KV for one chunk of one model layer."""

    chunk_id: int
    token_start: int
    token_end: int
    memory_obj: Any


@dataclass(frozen=True)
class SingleLayerChunkBuffers:
    """All chunk buffers that jointly cover one model layer."""

    chunks: tuple[ChunkLayerBuffer, ...]


@dataclass(frozen=True)
class SingleLayerKVBuffer:
    """One physical pinned region plus its authoritative logical chunk plan."""

    layout: KVLayout
    chunk_plan: SkillChunkPlan
    memory_obj: Any

    def __post_init__(self) -> None:
        object.__setattr__(self, "layout", KVLayout(self.layout))
        if self.layout not in (
            KVLayout.CHUNK_SINGLE_LAYER,
            KVLayout.PACKED_CHUNKS_SINGLE_LAYER,
        ):
            raise ValueError("one pinned layer requires a single-layer layout")
