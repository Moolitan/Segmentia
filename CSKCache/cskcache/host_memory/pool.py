"""LMCache-backed pinned host buffers built from one Skill ChunkPlan."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

from ..chunking import ChunkingSpec, build_chunk_plan
from ..layouts import KVLayout
from ..metadata.base import LayerExtent
from .base import ChunkLayerBuffer, SingleLayerChunkBuffers, SingleLayerKVBuffer


class LMCacheHostBufferPool:
    """Borrow complete layer groups from LMCache's long-lived CPU allocator."""

    def __init__(
        self,
        local_cpu_backend: LocalCPUBackend,
        *,
        layout: str = "packed_chunks_single_layer",
        chunk_size_tokens: int = 256,
    ) -> None:
        if not isinstance(local_cpu_backend, LocalCPUBackend):
            raise TypeError("LMCacheHostBufferPool requires LocalCPUBackend")
        if not local_cpu_backend.use_hot:
            raise ValueError("CSKCache requires the LMCache local CPU hot cache")
        try:
            parsed_layout = KVLayout(layout)
        except ValueError as exc:
            raise ValueError(f"unsupported CSKCache host layout: {layout}") from exc
        if parsed_layout not in (
            KVLayout.CHUNK_SINGLE_LAYER,
            KVLayout.PACKED_CHUNKS_SINGLE_LAYER,
        ):
            raise ValueError(
                "the current layerwise LMCache path requires a single-layer "
                "host layout"
            )
        if chunk_size_tokens <= 0:
            raise ValueError("chunk_size_tokens must be positive")
        self._backend = local_cpu_backend
        self.layout = parsed_layout.value
        self.chunk_size_tokens = chunk_size_tokens

    def acquire(self, extents: Sequence[LayerExtent]) -> Sequence[Any]:
        if self.layout == KVLayout.PACKED_CHUNKS_SINGLE_LAYER.value:
            return self.acquire_packed_chunks_single_layer(extents)
        return self.acquire_chunk_single_layer(extents)

    def acquire_persistent(
        self, extents: Sequence[LayerExtent]
    ) -> Sequence[Any]:
        """Allocate the complete per-layer regions encoded by the Catalog."""

        first, dtype, memory_format = self._validate_extents(extents)
        return _acquire_complete_layers(
            self._backend,
            shape=torch.Size(first.shape),
            dtype=dtype,
            memory_format=memory_format,
            layer_count=len(extents),
        )

    def arrange_loaded_layers(
        self,
        extents: Sequence[LayerExtent],
        persistent_layer_regions: Sequence[Any],
    ) -> Sequence[Any]:
        """Keep full layers or convert them to the configured chunk layout."""

        if len(persistent_layer_regions) != len(extents):
            raise ValueError("loaded layer count differs from cache metadata")
        if self.layout == KVLayout.PACKED_CHUNKS_SINGLE_LAYER.value:
            chunk_plan = self._build_chunk_plan(extents[0].shape[1])
            return tuple(
                SingleLayerKVBuffer(
                    layout=KVLayout(self.layout),
                    chunk_plan=chunk_plan,
                    memory_obj=memory_obj,
                )
                for memory_obj in persistent_layer_regions
            )

        try:
            chunk_layers = tuple(self.acquire_chunk_single_layer(extents))
        except Exception:
            self._release_memory_objects(persistent_layer_regions)
            raise
        try:
            for source, destination in zip(
                persistent_layer_regions, chunk_layers, strict=True
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
            self._release_memory_objects(persistent_layer_regions)
        return chunk_layers

    def acquire_packed_chunks_single_layer(
        self, extents: Sequence[LayerExtent]
    ) -> Sequence[Any]:
        objects = tuple(self.acquire_persistent(extents))
        first, dtype, _memory_format = self._validate_extents(extents)
        try:
            for extent, memory_obj in zip(extents, objects, strict=True):
                self._validate_memory_object(memory_obj, extent.shape, dtype)
        except Exception:
            self._release_memory_objects(objects)
            raise
        chunk_plan = self._build_chunk_plan(first.shape[1])
        return tuple(
            SingleLayerKVBuffer(
                layout=KVLayout(self.layout),
                chunk_plan=chunk_plan,
                memory_obj=memory_obj,
            )
            for memory_obj in objects
        )

    def acquire_chunk_single_layer(
        self,
        extents: Sequence[LayerExtent],
    ) -> Sequence[SingleLayerChunkBuffers]:
        first, dtype, memory_format = self._validate_extents(extents)
        if len(first.shape) != 3 or first.shape[0] != 2:
            raise ValueError(
                "chunk-single-layer CSKCache buffers require KV_2TD tensors"
            )

        token_count = first.shape[1]
        hidden_size = first.shape[2]
        chunk_plan = self._build_chunk_plan(token_count)
        return _acquire_chunk_single_layers(
            self._backend,
            layer_count=len(extents),
            chunk_plan=chunk_plan,
            hidden_size=hidden_size,
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
            if isinstance(item, SingleLayerChunkBuffers):
                self._release_memory_objects(
                    [chunk.memory_obj for chunk in item.chunks]
                )
            elif isinstance(item, SingleLayerKVBuffer):
                item.memory_obj.ref_count_down()
            else:
                item.ref_count_down()

    def _build_chunk_plan(self, token_count: int):
        return build_chunk_plan(
            token_count,
            ChunkingSpec(chunk_size_tokens=self.chunk_size_tokens),
        )

    @staticmethod
    def _release_memory_objects(memory_objects: Sequence[Any]) -> None:
        for memory_obj in memory_objects:
            memory_obj.ref_count_down()


def _acquire_complete_layers(
    backend: Any,
    *,
    shape: torch.Size,
    dtype: torch.dtype,
    memory_format: MemoryFormat,
    layer_count: int,
) -> tuple[Any, ...]:
    objects = backend.batched_allocate(
        shape,
        dtype,
        layer_count,
        fmt=memory_format,
        eviction=True,
        busy_loop=False,
    )
    if objects is None or len(objects) != layer_count:
        LMCacheHostBufferPool._release_memory_objects(objects or ())
        raise MemoryError("LMCache pinned pool cannot allocate complete layers")
    return tuple(objects)


def _acquire_chunk_single_layers(
    backend: Any,
    *,
    layer_count: int,
    chunk_plan: Any,
    hidden_size: int,
    dtype: torch.dtype,
    memory_format: MemoryFormat,
    validate_memory_object: Any,
) -> tuple[SingleLayerChunkBuffers, ...]:
    layer_chunks: list[list[ChunkLayerBuffer]] = [
        [] for _ in range(layer_count)
    ]
    allocated: list[Any] = []
    try:
        for chunk in chunk_plan.chunks:
            shape = torch.Size((2, chunk.token_count, hidden_size))
            objects = backend.batched_allocate(
                shape,
                dtype,
                layer_count,
                fmt=memory_format,
                eviction=True,
                busy_loop=False,
            )
            if objects is None or len(objects) != layer_count:
                if objects:
                    allocated.extend(objects)
                raise MemoryError(
                    "LMCache pinned pool cannot allocate chunk-layer buffers"
                )
            allocated.extend(objects)
            for layer_id, memory_obj in enumerate(objects):
                validate_memory_object(memory_obj, shape, dtype)
                layer_chunks[layer_id].append(
                    ChunkLayerBuffer(
                        chunk_id=chunk.chunk_id,
                        token_start=chunk.token_start,
                        token_end=chunk.token_end,
                        memory_obj=memory_obj,
                    )
                )
    except Exception:
        LMCacheHostBufferPool._release_memory_objects(allocated)
        raise
    return tuple(
        SingleLayerChunkBuffers(tuple(chunks)) for chunks in layer_chunks
    )
