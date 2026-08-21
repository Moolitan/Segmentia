"""LocalDisk SSD-to-pinned loading for complete Skill-layer groups."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from lmcache.utils import parse_cache_key
from lmcache.v1.memory_management import MemoryFormat

from ...metadata.base import CacheObjectMetadata, ContainerMetadata, StorageBackend
from ..base import CSKReadBatch, HostBufferPool, LayerObjectReadBackend


class LMCacheLayerObjectReader:
    """Use LMCache's layerwise LocalDisk API for Catalog-owned layer keys."""

    def __init__(self, storage_manager: Any, *, location: str) -> None:
        if not callable(
            getattr(storage_manager, "layerwise_batched_get", None)
        ) or not callable(getattr(storage_manager, "batched_unpin", None)):
            raise TypeError("LMCache StorageManager lacks layerwise retrieval")
        self._storage_manager = storage_manager
        self._location = location

    def register_catalog_objects(
        self, objects: Sequence[CacheObjectMetadata]
    ) -> None:
        backend = self._storage_manager.storage_backends.get(self._location)
        register = getattr(backend, "register_existing", None)
        validate = getattr(backend, "validate_existing", None)
        if not callable(register) or not callable(validate):
            raise TypeError("LMCache LocalDisk backend cannot register Catalog objects")
        entries = []
        for cache_object in objects:
            if cache_object.storage_backend is not StorageBackend.LOCAL_DISK:
                continue
            positions = torch.arange(
                cache_object.source_position_start,
                cache_object.source_position_start + cache_object.token_count,
                dtype=torch.int64,
            )
            for extent in sorted(
                cache_object.layers, key=lambda item: item.layer_id
            ):
                dtype = getattr(torch, extent.dtype, None)
                if not isinstance(dtype, torch.dtype):
                    raise ValueError(f"unsupported torch dtype: {extent.dtype}")
                try:
                    memory_format = MemoryFormat[extent.memory_layout]
                except KeyError as exc:
                    raise ValueError(
                        f"unsupported LMCache memory format: {extent.memory_layout}"
                    ) from exc
                shape = torch.Size(extent.shape)
                token_dim = (
                    1
                    if memory_format is MemoryFormat.KV_2TD
                    else memory_format.token_dim()
                )
                if (
                    shape.numel() * dtype.itemsize != extent.length_bytes
                    or shape[token_dim] != cache_object.token_count
                ):
                    raise ValueError("Catalog layer geometry is inconsistent")
                entries.append(
                    (
                        parse_cache_key(extent.backend_key),
                        extent.length_bytes,
                        shape,
                        dtype,
                        memory_format,
                        positions,
                    )
                )
        for key, size, _shape, _dtype, _fmt, _positions in entries:
            validate(key, size)
        for key, size, shape, dtype, fmt, positions in entries:
            register(
                key,
                size,
                shape,
                dtype,
                fmt,
                cached_positions=positions,
            )

    def read_layer_objects(self, backend_keys: Sequence[str]) -> Sequence[Any]:
        keys = [parse_cache_key(key) for key in backend_keys]
        tasks = tuple(
            self._storage_manager.layerwise_batched_get(
                [[key] for key in keys],
                location=self._location,
                lookup_id="cskcache-t0",
            )
        )
        results: list[Any] = []
        pinned_keys: list[Any] = []
        first_error: Exception | None = None
        for key, task in zip(keys, tasks, strict=True):
            try:
                layer = task.result()
                if len(layer) != 1 or layer[0] is None:
                    raise RuntimeError("LMCache returned an incomplete layer")
                memory_obj = layer[0]
                memory_obj.unpin()
                results.append(memory_obj)
                pinned_keys.append(key)
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if pinned_keys:
            self._storage_manager.batched_unpin(pinned_keys, [self._location])
        if first_error is not None or len(results) != len(keys):
            for memory_obj in results:
                memory_obj.ref_count_down()
            if first_error is not None:
                raise first_error
            raise RuntimeError("LMCache returned an incomplete layer group")
        return tuple(results)


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
