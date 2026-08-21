"""LMCache storage plugin that fills CSKCache extents in pinned memory."""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

from lmcache.utils import CacheEngineKey
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.storage_backend.abstract_backend import StoragePluginInterface
from cskcache import SingleLayerChunkBuffers, SingleLayerKVBuffer


class PinnedMemoryExtentBackend(StoragePluginInterface):
    supports_csk_chunk_layer_buffers = True

    def __init__(
        self,
        config=None,
        metadata=None,
        local_cpu_backend=None,
        loop=None,
        dst_device: str = "cpu",
    ) -> None:
        super().__init__(
            dst_device=dst_device,
            config=config,
            metadata=metadata,
            local_cpu_backend=local_cpu_backend,
            loop=loop,
        )
        extra = config.extra_config or {}
        self.device_path = str(extra["pinned_memory_extent.device_path"])
        self.capacity_bytes = int(extra["pinned_memory_extent.capacity_bytes"])
        self.block_align = int(extra["pinned_memory_extent.block_align"])
        self.header_bytes = int(extra["pinned_memory_extent.header_bytes"])
        self.fill_value = float(extra["pinned_memory_extent.fill_value"])

    def read_extents_into(
        self,
        offsets: Sequence[int],
        lengths: Sequence[int],
        objs: Sequence[MemoryObj],
    ) -> list[bool]:
        for item in objs:
            if isinstance(item, SingleLayerChunkBuffers):
                for chunk in item.chunks:
                    chunk.memory_obj.tensor.fill_(self.fill_value)
            elif isinstance(item, SingleLayerKVBuffer):
                item.memory_obj.tensor.fill_(self.fill_value)
            else:
                item.tensor.fill_(self.fill_value)
        return [True] * len(objs)

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        return False

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        return False

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        objs: list[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        return None

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        return None

    def pin(self, key: CacheEngineKey) -> bool:
        return False

    def unpin(self, key: CacheEngineKey) -> bool:
        return False

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        return False

    def get_allocator_backend(self):
        return self.local_cpu_backend

    def close(self) -> None:
        return None
