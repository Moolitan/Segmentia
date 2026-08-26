"""Public contracts for loading one complete Skill KV object into host memory.

This module defines data and interfaces only.  Concrete raw-block, LocalDisk,
buffer-pool, threading, and lifecycle behavior belongs to sibling modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from ..metadata.base import (
    CacheObjectMetadata,
    ContainerMetadata,
    LayerExtent,
    StorageBackend,
)
from ..layouts import KVLayout
from .formats import StorageFormat


@dataclass(frozen=True)
class StorageSpec:
    """Persistent layout, encoding, and physical backend selection."""

    layout: KVLayout
    format: StorageFormat
    backend: StorageBackend

    def __post_init__(self) -> None:
        object.__setattr__(self, "layout", KVLayout(self.layout))
        object.__setattr__(self, "format", StorageFormat(self.format))
        object.__setattr__(self, "backend", StorageBackend(self.backend))
        expected = {
            StorageBackend.LOCAL_DISK: StorageFormat.TORCH_PT,
            StorageBackend.RAW_BLOCK: StorageFormat.RAW_CONTAINER,
        }[self.backend]
        if self.format is not expected:
            raise ValueError("storage format is incompatible with its backend")


class ExtentReadBackend(Protocol):
    """Physical reader for offset-addressed raw-block extents.

    The injected implementation owns its sync/io_uring engine, buffered/direct
    open mode, and queue depth.  CSKCache only submits validated physical
    extents through this protocol.
    """

    device_path: str
    capacity_bytes: int
    block_align: int
    header_bytes: int

    def read_extents_into(
        self,
        offsets: Sequence[int],
        lengths: Sequence[int],
        objs: Sequence[Any],
    ) -> list[bool]: ...


class LayerObjectReadBackend(Protocol):
    """Key-addressed reader for one complete group of Skill-layer objects."""

    def read_layer_objects(self, backend_keys: Sequence[str]) -> Sequence[Any]: ...


class HostBufferPool(Protocol):
    """Pinned-memory allocation and layout contract used by storage transfers."""

    def acquire(self, extents: Sequence[LayerExtent]) -> Sequence[Any]: ...

    def acquire_persistent(
        self, extents: Sequence[LayerExtent]
    ) -> Sequence[Any]: ...

    def release(self, memory_objects: Sequence[Any]) -> None: ...

    def arrange_loaded_layers(
        self,
        extents: Sequence[LayerExtent],
        persistent_layer_regions: Sequence[Any],
    ) -> Sequence[Any]: ...


@dataclass(frozen=True)
class CSKReadBatch:
    """One complete, layer-ordered physical read request for a Skill object."""

    cache_object_id: str
    container_id: str | None
    extents: tuple[LayerExtent, ...]

    @classmethod
    def from_metadata(
        cls,
        metadata: CacheObjectMetadata,
        container: ContainerMetadata | None,
        *,
        expected_layers: int,
    ) -> "CSKReadBatch":
        metadata.validate(expected_layers, container)
        return cls(
            cache_object_id=metadata.object_id,
            container_id=None if container is None else container.container_id,
            extents=tuple(sorted(metadata.layers, key=lambda item: item.layer_id)),
        )

    @property
    def layer_ids(self) -> tuple[int, ...]:
        return tuple(extent.layer_id for extent in self.extents)

    @property
    def offsets(self) -> tuple[int, ...]:
        if any(extent.offset_bytes is None for extent in self.extents):
            raise ValueError("this backend does not expose physical offsets")
        return tuple(int(extent.offset_bytes) for extent in self.extents)

    @property
    def lengths(self) -> tuple[int, ...]:
        return tuple(extent.length_bytes for extent in self.extents)


@dataclass(frozen=True)
class CSKReadResult:
    """Per-layer evidence with an explicit whole-object completion verdict."""

    cache_object_id: str
    layer_ids: tuple[int, ...]
    per_layer_success: tuple[bool, ...]

    @classmethod
    def from_backend_results(
        cls,
        batch: CSKReadBatch,
        results: Sequence[bool],
    ) -> "CSKReadResult":
        if len(results) != len(batch.extents):
            raise RuntimeError(
                "physical backend returned a result count different from the batch"
            )
        return cls(
            cache_object_id=batch.cache_object_id,
            layer_ids=batch.layer_ids,
            per_layer_success=tuple(bool(item) for item in results),
        )

    @property
    def complete(self) -> bool:
        return bool(self.per_layer_success) and all(self.per_layer_success)
