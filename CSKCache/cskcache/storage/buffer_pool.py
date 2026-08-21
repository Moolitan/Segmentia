"""LMCache-backed pinned host-buffer pool used by CSKCache."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.utils import parse_cache_key

from ..metadata.base import CacheObjectMetadata, LayerExtent, StorageBackend
from .layouts.base import ChunkedLayerBuffer, HostLayout
from .layouts.chunked_layer import acquire_chunked_layers
from .layouts.full_layer import acquire_full_layers


class LMCacheLayerObjectReader:
    """Read one complete Skill layer group through LMCache's layerwise API."""

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
        """Register LocalDisk files described by the authoritative Catalog."""

        backend = self._storage_manager.storage_backends.get(self._location)
        register = getattr(backend, "register_existing", None)
        validate = getattr(backend, "validate_existing", None)
        if not callable(register) or not callable(validate):
            raise TypeError("LMCache LocalDisk backend cannot register Catalog objects")
        entries: list[
            tuple[Any, int, torch.Size, torch.dtype, MemoryFormat, torch.Tensor]
        ] = []
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
            self._storage_manager.batched_unpin(
                pinned_keys,
                [self._location],
            )
        if first_error is not None or len(results) != len(keys):
            self._release_memory_objects(results)
            if first_error is not None:
                raise first_error
            raise RuntimeError("LMCache returned an incomplete layer-object group")
        return tuple(results)

    @staticmethod
    def _release_memory_objects(memory_objects: Sequence[Any]) -> None:
        for memory_obj in memory_objects:
            memory_obj.ref_count_down()


class LMCacheHostBufferPool:
    """Borrow complete layer groups from LMCache's long-lived CPU allocator."""

    def __init__(
        self,
        local_cpu_backend: LocalCPUBackend,
        *,
        layout: str = "full_layer",
        chunk_tokens: int = 256,
    ) -> None:
        if not isinstance(local_cpu_backend, LocalCPUBackend):
            raise TypeError("LMCacheHostBufferPool requires LocalCPUBackend")
        if not local_cpu_backend.use_hot:
            raise ValueError("CSKCache requires the LMCache local CPU hot cache")
        try:
            parsed_layout = HostLayout(layout)
        except ValueError as exc:
            raise ValueError(f"unsupported CSKCache host layout: {layout}") from exc
        if chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be positive")
        self._backend = local_cpu_backend
        self.layout = parsed_layout.value
        self.chunk_tokens = chunk_tokens

    def acquire(self, extents: Sequence[LayerExtent]) -> Sequence[Any]:
        if self.layout == "full_layer":
            return self.acquire_full_layer(extents)
        return self.acquire_chunk_layer(extents)

    def arrange_loaded_layers(
        self,
        extents: Sequence[LayerExtent],
        full_layer_objects: Sequence[Any],
    ) -> Sequence[Any]:
        """Keep full layers or convert them to the configured chunk layout."""

        if len(full_layer_objects) != len(extents):
            raise ValueError("loaded layer count differs from cache metadata")
        if self.layout == "full_layer":
            return tuple(full_layer_objects)

        try:
            chunk_layers = tuple(self.acquire_chunk_layer(extents))
        except Exception:
            self._release_memory_objects(full_layer_objects)
            raise
        try:
            for source, destination in zip(
                full_layer_objects, chunk_layers, strict=True
            ):
                if source.tensor is None:
                    raise ValueError("loaded LMCache layer has no tensor")
                for chunk in destination.chunks:
                    if chunk.memory_obj.tensor is None:
                        raise ValueError("chunk-layer destination has no tensor")
                    chunk.memory_obj.tensor.copy_(
                        source.tensor[:, chunk.token_start : chunk.token_end]
                    )
                    positions = source.metadata.cached_positions
                    if positions is not None:
                        chunk.memory_obj.metadata.cached_positions = positions[
                            chunk.token_start : chunk.token_end
                        ].clone()
        except Exception:
            self.release(chunk_layers)
            raise
        finally:
            self._release_memory_objects(full_layer_objects)
        return chunk_layers

    def acquire_full_layer(
        self, extents: Sequence[LayerExtent]
    ) -> Sequence[Any]:
        first, dtype, memory_format = self._validate_extents(extents)
        objects = acquire_full_layers(
            self._backend,
            shape=torch.Size(first.shape),
            dtype=dtype,
            memory_format=memory_format,
            layer_count=len(extents),
        )
        try:
            for extent, memory_obj in zip(extents, objects, strict=True):
                self._validate_memory_object(memory_obj, extent.shape, dtype)
        except Exception:
            self._release_memory_objects(objects)
            raise
        return tuple(objects)

    def acquire_chunk_layer(
        self,
        extents: Sequence[LayerExtent],
    ) -> Sequence[ChunkedLayerBuffer]:
        first, dtype, memory_format = self._validate_extents(extents)
        if len(first.shape) != 3 or first.shape[0] != 2:
            raise ValueError("chunk-major CSKCache buffers require KV_2TD tensors")

        token_count = first.shape[1]
        hidden_size = first.shape[2]
        return acquire_chunked_layers(
            self._backend,
            layer_count=len(extents),
            token_count=token_count,
            hidden_size=hidden_size,
            chunk_tokens=self.chunk_tokens,
            dtype=dtype,
            memory_format=memory_format,
            validate_memory_object=self._validate_memory_object,
        )

    def _validate_extents(
        self, extents: Sequence[LayerExtent]
    ) -> tuple[LayerExtent, torch.dtype, MemoryFormat]:
        if not extents:
            raise ValueError("a host-buffer group must contain at least one extent")
        first = extents[0]
        for extent in extents:
            if (
                extent.shape != first.shape
                or extent.dtype != first.dtype
                or extent.memory_layout != first.memory_layout
            ):
                raise ValueError("one batched allocation requires homogeneous layers")
        dtype = getattr(torch, first.dtype, None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"unsupported torch dtype: {first.dtype}")
        for extent in extents:
            logical_bytes = torch.Size(extent.shape).numel() * dtype.itemsize
            if logical_bytes != extent.length_bytes:
                raise ValueError(
                    "persisted extent length differs from its shape and dtype"
                )
        try:
            memory_format = MemoryFormat[first.memory_layout]
        except KeyError as exc:
            raise ValueError(
                f"unsupported LMCache memory format: {first.memory_layout}"
            ) from exc
        return first, dtype, memory_format

    @staticmethod
    def _validate_memory_object(
        memory_obj: Any,
        expected_shape: Sequence[int],
        dtype: torch.dtype,
    ) -> None:
        logical_bytes = torch.Size(expected_shape).numel() * dtype.itemsize
        if memory_obj.get_physical_size() < logical_bytes:
            raise MemoryError(
                "LMCache pinned page is smaller than the persisted extent"
            )
        if memory_obj.get_size() != logical_bytes:
            raise ValueError(
                "LMCache memory object has the wrong logical extent size"
            )
        tensor = memory_obj.tensor
        if (
            tensor is None
            or tensor.shape != torch.Size(expected_shape)
            or tensor.dtype is not dtype
        ):
            raise ValueError(
                "LMCache memory object has the wrong logical tensor layout"
            )

    def release(self, memory_objects: Sequence[Any]) -> None:
        for item in memory_objects:
            if isinstance(item, ChunkedLayerBuffer):
                self._release_memory_objects(
                    [chunk.memory_obj for chunk in item.chunks]
                )
            else:
                item.ref_count_down()

    @staticmethod
    def _release_memory_objects(memory_objects: Sequence[Any]) -> None:
        for memory_obj in memory_objects:
            memory_obj.ref_count_down()
