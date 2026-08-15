"""LMCache-backed pinned host-buffer pool used by CSKCache."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

from .cache_metadata import LayerExtent


class LMCacheHostBufferPool:
    """Borrow complete layer groups from LMCache's long-lived CPU allocator."""

    def __init__(self, local_cpu_backend: LocalCPUBackend) -> None:
        if not isinstance(local_cpu_backend, LocalCPUBackend):
            raise TypeError("LMCacheHostBufferPool requires LocalCPUBackend")
        if not local_cpu_backend.use_hot:
            raise ValueError("CSKCache requires the LMCache local CPU hot cache")
        self._backend = local_cpu_backend

    def acquire(self, extents: Sequence[LayerExtent]) -> Sequence[Any]:
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
        objects = self._backend.batched_allocate(
            torch.Size(first.shape),
            dtype,
            len(extents),
            fmt=memory_format,
            eviction=True,
            busy_loop=False,
        )
        if objects is None or len(objects) != len(extents):
            if objects:
                self.release(objects)
            raise MemoryError("LMCache pinned CPU pool cannot allocate a full layer group")
        try:
            for extent, memory_obj in zip(extents, objects, strict=True):
                if memory_obj.get_physical_size() < extent.length_bytes:
                    raise MemoryError(
                        "LMCache pinned page is smaller than the persisted extent"
                    )
                if memory_obj.get_size() != extent.length_bytes:
                    raise ValueError(
                        "LMCache memory object has the wrong logical extent size"
                    )
                tensor = memory_obj.tensor
                if (
                    tensor is None
                    or tensor.shape != torch.Size(extent.shape)
                    or tensor.dtype is not dtype
                ):
                    raise ValueError(
                        "LMCache memory object has the wrong logical tensor layout"
                    )
        except Exception:
            self.release(objects)
            raise
        return tuple(objects)

    def release(self, memory_objects: Sequence[Any]) -> None:
        for memory_obj in memory_objects:
            memory_obj.ref_count_down()
