"""Key-addressed layer-object transfer into the selected host layout."""

from __future__ import annotations

from typing import Any

from ...metadata.base import ContainerMetadata, StorageBackend
from ..base import CSKReadBatch, HostBufferPool, LayerObjectReadBackend


class LayerObjectTransfer:
    """Load per-layer objects through a key-addressed storage backend."""

    storage_backend = StorageBackend.LOCAL_DISK

    def __init__(
        self,
        backend: LayerObjectReadBackend,
        host_buffer_pool: HostBufferPool | None,
    ) -> None:
        self._backend = backend
        self._host_buffer_pool = host_buffer_pool

    def validate_container(self, container: ContainerMetadata | None) -> None:
        """Reject raw-container metadata for a key-addressed object."""

        if container is not None:
            raise ValueError("local_disk storage must not reference a raw container")

    def load(self, batch: CSKReadBatch) -> tuple[Any, ...]:
        """Read every layer object and arrange the selected host layout."""

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

    def load_layer(self, batch: CSKReadBatch, layer_id: int) -> Any:
        """Read and arrange one key-addressed layer object."""

        if self._host_buffer_pool is None:
            raise RuntimeError("local_disk loading has no host buffer pool")
        if not 0 <= layer_id < len(batch.extents):
            raise ValueError("local_disk layer_id is outside the read batch")
        extent = batch.extents[layer_id]
        if extent.layer_id != layer_id:
            raise ValueError("local_disk batch is not in model-layer order")
        loaded = tuple(
            self._backend.read_layer_objects([extent.backend_key])
        )
        if len(loaded) != 1:
            self._host_buffer_pool.release(loaded)
            raise RuntimeError("LocalDisk did not return one layer")
        arranged = tuple(
            self._host_buffer_pool.arrange_loaded_layers((extent,), loaded)
        )
        if len(arranged) != 1:
            self._host_buffer_pool.release(arranged)
            raise RuntimeError("host layout did not return one arranged layer")
        return arranged[0]
