from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from cskcache.profiling import LoadTrace, NullLoadTrace
from cskcache.v1.async_load import GpuPrefetchHandle, submit_gpu_prefetch
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
        trace: LoadTrace | NullLoadTrace | None = None,
        prefetch_stream: torch.cuda.Stream | None = None,
    ) -> tuple[int, int, int]:
        """Scatter every entry layer and return expected/scattered/skipped counts.

        ``prefetch_stream`` is an optional opt-in: when given, layer i+1's
        H2D copy + RoPE correction runs on that stream while layer i's
        scatter runs on the caller's current stream, so the two overlap
        instead of running strictly back-to-back. Omitting it (the default)
        reproduces the exact original sequential behavior.
        """

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
        trace: LoadTrace | NullLoadTrace | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the cached K/V slice, RoPE-corrected to the target position."""

    @abstractmethod
    def gather(
        self,
        kv_layer: torch.Tensor,
        block_ids: list[int],
        start: int,
        end: int,
        trace: LoadTrace | NullLoadTrace | None = None,
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
        self._prefetch_stream: torch.cuda.Stream | None = None

    def get_prefetch_stream(self) -> torch.cuda.Stream:
        """Lazily create and cache one background CUDA stream for the life
        of this connector, used by ``to_gpu()``'s optional pipelined path.
        Not part of ``KVConnectorInterface``: callers that want pipelining
        duck-type-check for this method (``getattr(gpu, "get_prefetch_stream",
        None)``) rather than requiring every connector implementation to
        support it.
        """
        if self._prefetch_stream is None:
            device = next(iter(self._kv_caches.values())).device
            self._prefetch_stream = torch.cuda.Stream(device=device)
        return self._prefetch_stream

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
        trace: LoadTrace | NullLoadTrace | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return prepare_reuse_slice(
            entry,
            layer_name=layer_name,
            source_offset=source_offset,
            length=length,
            target_start=target_start,
            rope=self._rope,
            device=device,
            trace=trace,
        )

    def gather(
        self,
        kv_layer: torch.Tensor,
        block_ids: list[int],
        start: int,
        end: int,
        trace: LoadTrace | NullLoadTrace | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stage_trace = trace if trace is not None else NullLoadTrace()
        with stage_trace.cuda_stage("probe_gather", kv_layer.device):
            return gather_span(kv_layer, block_ids, start, end, self._block_size)

    def to_gpu(
        self,
        entry: CSKCacheEntry,
        plan: CSKLoadPlan,
        block_ids: list[int],
        trace: LoadTrace | NullLoadTrace | None = None,
        prefetch_stream: torch.cuda.Stream | None = None,
    ) -> tuple[int, int, int]:
        expected_layers = set(entry.kv_by_layer)
        available_layers = set(self._kv_caches)
        missing_layers = sorted(expected_layers - available_layers)
        if not expected_layers:
            raise RuntimeError(
                f"CSKCache entry {entry.cache_id} contains no KV layers"
            )
        if missing_layers:
            preview = ",".join(missing_layers[:8])
            raise RuntimeError(
                "CSKCache KV load rejected before scatter: "
                f"cache_id={entry.cache_id} expected_layers={len(expected_layers)} "
                f"available_layers={len(available_layers)} "
                f"missing_layers={len(missing_layers)} missing=[{preview}]"
            )

        stage_trace = trace if trace is not None else NullLoadTrace()
        layer_names = list(entry.kv_by_layer)
        if prefetch_stream is None:
            scattered_layers = self._scatter_sequential(
                entry, plan, block_ids, layer_names, stage_trace
            )
        else:
            scattered_layers = self._scatter_pipelined(
                entry, plan, block_ids, layer_names, stage_trace, prefetch_stream
            )
        return len(expected_layers), scattered_layers, 0

    def _scatter_sequential(
        self,
        entry: CSKCacheEntry,
        plan: CSKLoadPlan,
        block_ids: list[int],
        layer_names: list[str],
        stage_trace: LoadTrace | NullLoadTrace,
    ) -> int:
        """Original behavior: H2D+RoPE and scatter run back-to-back per layer."""
        scattered_layers = 0
        for layer_name in layer_names:
            target_cache = self._kv_caches[layer_name]
            key, value = self.reuse_slice(
                entry,
                layer_name=layer_name,
                source_offset=plan.source_offset,
                length=plan.length,
                target_start=plan.start,
                device=target_cache.device,
                trace=stage_trace,
            )
            with stage_trace.cuda_stage("scatter_span", target_cache.device):
                scatter_span(
                    target_cache,
                    block_ids,
                    plan.start,
                    plan.end,
                    self._block_size,
                    key,
                    value,
                )
            scattered_layers += 1
        return scattered_layers

    def _scatter_pipelined(
        self,
        entry: CSKCacheEntry,
        plan: CSKLoadPlan,
        block_ids: list[int],
        layer_names: list[str],
        stage_trace: LoadTrace | NullLoadTrace,
        prefetch_stream: torch.cuda.Stream,
    ) -> int:
        """Layer i+1's H2D+RoPE (reuse_slice) runs on prefetch_stream while
        layer i's scatter -- the only step that touches the shared paged
        cache -- runs on the caller's current stream. Each layer's source
        tensors and freshly-allocated (key, value) are private to that
        layer, so there is nothing for the two streams to race on; the only
        ordering that matters (layer i's scatter must see layer i's own
        already-ready key/value) is enforced by GpuPrefetchHandle.result()'s
        cross-stream wait.
        """

        def fetch(layer_name: str) -> GpuPrefetchHandle:
            target_cache = self._kv_caches[layer_name]
            return submit_gpu_prefetch(
                lambda: self.reuse_slice(
                    entry,
                    layer_name=layer_name,
                    source_offset=plan.source_offset,
                    length=plan.length,
                    target_start=plan.start,
                    device=target_cache.device,
                    trace=stage_trace,
                ),
                prefetch_stream,
            )

        scattered_layers = 0
        pending = fetch(layer_names[0])
        for index, layer_name in enumerate(layer_names):
            target_cache = self._kv_caches[layer_name]
            key, value = pending.result()
            if index + 1 < len(layer_names):
                pending = fetch(layer_names[index + 1])
            with stage_trace.cuda_stage("scatter_span", target_cache.device):
                scatter_span(
                    target_cache,
                    block_ids,
                    plan.start,
                    plan.end,
                    self._block_size,
                    key,
                    value,
                )
            scattered_layers += 1
        return scattered_layers
