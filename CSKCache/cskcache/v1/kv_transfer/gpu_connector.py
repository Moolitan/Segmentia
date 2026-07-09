from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from cskcache.v1.compute.reuse import prepare_reuse_slice
from cskcache.v1.metadata import CSKCacheEntry, CSKLoadPlan
from cskcache.v1.rope import find_rotary_embedding
from cskcache.v1.slot_ops import gather_span, scatter_span


class KVConnectorInterface(ABC):
    """Contract for moving K/V between offline entries and paged KV cache.

    The engine calls this to (a) scatter a reuse plan into the paged cache
    before forward and (b) slice/gather K/V for probe comparison. Keeping the
    interface small means an alternate memory layout only has to reimplement
    these primitives, not the reuse/gate policy.
    """

    @abstractmethod
    def bind_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Register the per-layer paged KV cache tensors for this worker."""

    @abstractmethod
    def set_model(self, model: object | None) -> None:
        """Provide the model so keys can be RoPE-corrected across positions."""

    @abstractmethod
    def to_gpu(
        self,
        entry: CSKCacheEntry,
        plan: CSKLoadPlan,
        block_ids: list[int],
    ) -> None:
        """Scatter every layer of a reuse plan into the paged cache."""

    @abstractmethod
    def reuse_slice(
        self,
        entry: CSKCacheEntry,
        *,
        layer_name: str,
        source_offset: int,
        length: int,
        target_start: int,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the cached K/V slice, RoPE-corrected to the target position."""

    @abstractmethod
    def gather(
        self,
        kv_layer: torch.Tensor,
        block_ids: list[int],
        start: int,
        end: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read the K/V a forward just wrote for one layer's span."""


class VLLMPagedGPUConnector(KVConnectorInterface):
    """KV movement against vLLM's paged KV cache layout.

    Wraps the existing ``slot_ops`` (scatter/gather), ``compute.reuse``
    (slice + key correction), and ``rope`` helpers. Despite the name it imports
    no vLLM at module load; only cross-position key rotation touches vLLM's
    rotary helpers, and only at runtime.
    """

    def __init__(self, block_size: int) -> None:
        self._block_size = block_size
        self._kv_caches: dict[str, torch.Tensor] = {}
        self._rope: object | None = None

    def bind_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self._kv_caches = kv_caches

    @property
    def kv_caches(self) -> dict[str, torch.Tensor]:
        return self._kv_caches

    def set_model(self, model: object | None) -> None:
        self._rope = find_rotary_embedding(model) if model is not None else None

    def reuse_slice(
        self,
        entry: CSKCacheEntry,
        *,
        layer_name: str,
        source_offset: int,
        length: int,
        target_start: int,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return prepare_reuse_slice(
            entry,
            layer_name=layer_name,
            source_offset=source_offset,
            length=length,
            target_start=target_start,
            rope=self._rope,
            device=device,
        )

    def gather(
        self,
        kv_layer: torch.Tensor,
        block_ids: list[int],
        start: int,
        end: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return gather_span(kv_layer, block_ids, start, end, self._block_size)

    def to_gpu(
        self,
        entry: CSKCacheEntry,
        plan: CSKLoadPlan,
        block_ids: list[int],
    ) -> None:
        for layer_name in entry.kv_by_layer:
            target_cache = self._kv_caches.get(layer_name)
            if target_cache is None:
                continue
            key, value = self.reuse_slice(
                entry,
                layer_name=layer_name,
                source_offset=plan.source_offset,
                length=plan.length,
                target_start=plan.start,
                device=target_cache.device,
            )
            scatter_span(
                target_cache,
                block_ids,
                plan.start,
                plan.end,
                self._block_size,
                key,
                value,
            )
