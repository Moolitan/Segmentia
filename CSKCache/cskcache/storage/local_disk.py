"""LocalDisk SSD-to-pinned loading for complete Skill-layer groups."""

from __future__ import annotations

from typing import Any

from ..metadata.base import ContainerMetadata, StorageBackend
from .base import CSKReadBatch, HostBufferPool, LayerObjectReadBackend


class LocalDiskLoader:
    """Load per-layer Skill objects through LMCache's LocalDisk interface."""

    storage_backend = StorageBackend.LOCAL_DISK

    def __init__(
        self,
        backend: LayerObjectReadBackend,
        host_buffer_pool: HostBufferPool | None,
    ) -> None:
        self._backend = backend
        self._host_buffer_pool = host_buffer_pool

    def validate_container(self, container: ContainerMetadata | None) -> None:
        """Reject raw-container metadata for a key-addressed LocalDisk object."""

        if container is not None:
            raise ValueError("local_disk storage must not reference a raw container")

    def load(self, batch: CSKReadBatch) -> tuple[Any, ...]:
        """Read all Skill-layer files and arrange the selected host layout."""

        if self._host_buffer_pool is None:
            raise RuntimeError("local_disk loading has no host buffer pool")
        loaded = tuple(
            self._backend.read_layer_objects(
                [extent.backend_key for extent in batch.extents]
            )
        )
        if len(loaded) != len(batch.extents):
            self._host_buffer_pool.release(loaded)
            raise RuntimeError("LocalDisk returned an incomplete layer group")
        return tuple(
            self._host_buffer_pool.arrange_loaded_layers(batch.extents, loaded)
        )
