"""Public input contracts for offline Skill KV metadata builders.

Concrete raw-block and LocalDisk builders live in sibling modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from ...chunking import ChunkingSpec
from ...layouts import KVLayout


class OfflineOffsetBackend(Protocol):
    """LMCache slot information used after a direct offline raw save."""

    header_bytes: int

    def entry_offset(self, key: Any) -> int | None: ...


class DirectRawOffsetBackend(OfflineOffsetBackend, Protocol):
    """Physical facts exposed by LMCache after a direct raw-block save."""

    device_path: str
    capacity_bytes: int
    block_align: int


@dataclass(frozen=True)
class LayerBuildInput:
    """Tensor facts needed to publish one persistent layer extent."""

    layer_id: int
    backend_key: str
    lookup_key: Any
    length_bytes: int
    dtype: str
    shape: tuple[int, ...]
    memory_layout: str
    payload_sha256: str


@dataclass(frozen=True)
class CacheObjectBuildInput:
    """Semantic identity and all layers of one offline Skill KV object."""

    object_id: str
    skill_name: str
    skill_version: str
    model_fingerprint: str
    tokenizer_fingerprint: str
    token_count: int
    source_position_start: int
    token_ids_sha256: str
    start_marker_token_ids: tuple[int, ...]
    layers: tuple[LayerBuildInput, ...]
    chunking: ChunkingSpec = ChunkingSpec(256)
    storage_layout: KVLayout = KVLayout.PACKED_CHUNKS_SINGLE_LAYER


@dataclass(frozen=True)
class DirectRawLayerBuildInput:
    """Layer identity and tensor geometry before raw payload verification."""

    layer_id: int
    backend_key: str
    lookup_key: Any
    length_bytes: int
    dtype: str
    shape: tuple[int, ...]
    memory_layout: str


@dataclass(frozen=True)
class DirectRawCacheObjectBuildInput:
    """One exact-save object already written into the shared raw container."""

    object_id: str
    skill_name: str
    skill_version: str
    model_fingerprint: str
    tokenizer_fingerprint: str
    token_count: int
    source_position_start: int
    token_ids_sha256: str
    start_marker_token_ids: tuple[int, ...]
    layers: tuple[DirectRawLayerBuildInput, ...]
    chunking: ChunkingSpec = ChunkingSpec(256)
    storage_layout: KVLayout = KVLayout.PACKED_CHUNKS_SINGLE_LAYER


@dataclass(frozen=True)
class LocalDiskLayerBuildInput:
    """One complete Skill-layer file written by LMCache LocalDisk."""

    layer_id: int
    backend_key: str
    data_path: str
    length_bytes: int
    dtype: str
    shape: tuple[int, ...]
    memory_layout: str


@dataclass(frozen=True)
class LocalDiskCacheObjectBuildInput:
    """Semantic identity and per-layer LocalDisk files for one Skill."""

    object_id: str
    skill_name: str
    skill_version: str
    model_fingerprint: str
    tokenizer_fingerprint: str
    token_count: int
    source_position_start: int
    token_ids_sha256: str
    start_marker_token_ids: tuple[int, ...]
    layers: tuple[LocalDiskLayerBuildInput, ...]
    chunking: ChunkingSpec = ChunkingSpec(256)
    storage_layout: KVLayout = KVLayout.PACKED_CHUNKS_SINGLE_LAYER
