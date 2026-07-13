"""vLLM V1 adapter for CSKCache.

This module is the thin translation layer between vLLM's generic KVConnector
lifecycle and the vLLM-agnostic :class:`CSKCacheEngine`. Every scheduling and
reuse decision lives in the engine; this file only:

- extracts plain data (token ids, computed frontier, ``kv_transfer_params``,
  physical block ids, paged KV tensors) from vLLM ``Request`` /
  ``KVCacheBlocks`` / ``ForwardContext`` objects,
- forwards it to the engine, and
- wraps the engine's plain result carriers (:class:`CSKReqMeta`,
  :class:`CSKProbeMeta`, :class:`CSKProbeDecision`) into vLLM's serializable
  ``KVConnectorMetadata`` / ``KVConnectorWorkerMetadata`` envelopes.

CSKCache identifies the current skill span only from an explicit reuse signal
under ``request.kv_transfer_params["cskcache"]``. Without that signal, the
engine does not scan prompt tokens and lets vLLM perform normal prefill.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)
from vllm.logger import init_logger
from vllm.v1.outputs import KVConnectorOutput

from cskcache.integration.vllm.utils import load_vllm_config
from cskcache.v1.compute import CSKProbeDecision
from cskcache.v1.core.cache_engine import CSKCacheEngine
from cskcache.v1.metadata import CSKProbeMeta, CSKReqMeta, CSKSaveMeta
from cskcache.v1.registry import CSKCacheRegistry, get_global_registry
from cskcache.v1.storage.storage_manager import StorageManager

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request


logger = init_logger(__name__)


@dataclass
class CSKConnectorMetadata(KVConnectorMetadata):
    """Opaque scheduler-to-worker payload carried by vLLM KVConnectorOutput.

    The engine produces plain ``CSKReqMeta`` / ``CSKProbeMeta`` carriers; this
    envelope is the vLLM-serializable container that ships them to the worker.
    """

    requests: list[CSKReqMeta] = field(default_factory=list)
    probes: list[CSKProbeMeta] = field(default_factory=list)
    saves: list[CSKSaveMeta] = field(default_factory=list)


@dataclass
class CSKProbeWorkerMetadata(KVConnectorWorkerMetadata):
    """Worker-to-scheduler payload carrying probe gate decisions."""

    decisions: list[CSKProbeDecision] = field(default_factory=list)

    def aggregate(
        self,
        other: "KVConnectorWorkerMetadata",
    ) -> "CSKProbeWorkerMetadata":
        if not isinstance(other, CSKProbeWorkerMetadata):
            return self
        return CSKProbeWorkerMetadata(decisions=self.decisions + other.decisions)


class CSKCacheConnectorV1Impl:
    """vLLM lifecycle front-end; all logic is delegated to CSKCacheEngine."""

    def __init__(self, vllm_config: "VllmConfig", role: Any, parent: Any) -> None:
        self._parent = parent
        config = load_vllm_config(vllm_config)
        block_size = vllm_config.cache_config.block_size

        # Storage: use the process-global registry by default (CPU-only, matching
        # the historical behavior). If the config asks for disk spill, build a
        # dedicated CPU+disk registry instead so the working set can exceed RAM.
        if config.disk_dir is not None or config.cpu_max_bytes is not None:
            storage = StorageManager.with_disk(
                disk_dir=config.disk_dir if config.disk_dir is not None else config.kv_dir,
                cpu_max_bytes=config.cpu_max_bytes,
            )
            self._registry: CSKCacheRegistry = CSKCacheRegistry(storage)
        else:
            self._registry = get_global_registry()
            storage = self._registry.storage

        if config.kv_dir is not None:
            loaded = self._registry.load_dir(config.kv_dir)
            logger.warning(
                "CSKCache loaded %d KV entries from %s", len(loaded), config.kv_dir
            )

        # The engine builds its matcher catalog from whatever storage now holds.
        self._engine = CSKCacheEngine(config, storage, block_size)
        self._kv_caches_bound = False
        logger.warning(
            "CSKCache connector initialized: role=%s catalog_segments=%d probe_enabled=%s",
            role,
            len(self._engine.catalog.segments),
            config.probe_enabled,
        )

    # ---- worker: cache registration -------------------------------------

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self._engine.register_kv_caches(kv_caches)
        self._kv_caches_bound = True

    # ---- scheduler side --------------------------------------------------

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        return self._engine.get_num_new_matched_tokens(
            request.request_id,
            self._request_token_ids(request),
            num_computed_tokens,
            getattr(request, "kv_transfer_params", None),
        )

    def cap_prefill_before_reuse(
        self,
        request: "Request",
        base_num_computed_tokens: int,
        num_new_tokens: int,
    ) -> int:
        return self._engine.cap_prefill_before_reuse(
            request.request_id,
            self._request_token_ids(request),
            base_num_computed_tokens,
            num_new_tokens,
            getattr(request, "kv_transfer_params", None),
        )

    def get_boundary_reuse_load_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> int:
        return self._engine.get_boundary_reuse_load_tokens(
            request.request_id,
            self._request_token_ids(request),
            num_computed_tokens,
        )

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        block_ids = blocks.get_block_ids(allow_none=True)
        self._engine.update_reuse_after_alloc(
            request.request_id,
            block_ids,
            num_external_tokens,
        )
        self._engine.update_save_after_alloc(
            request.request_id,
            self._request_token_ids(request),
            request.num_computed_tokens,
            block_ids,
            getattr(request, "kv_transfer_params", None),
        )

    def build_connector_meta(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> KVConnectorMetadata:
        requests, probes, saves = self._engine.build_meta(
            scheduler_output.num_scheduled_tokens
        )
        return CSKConnectorMetadata(requests=requests, probes=probes, saves=saves)

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        worker_meta = connector_output.kv_connector_worker_meta
        if isinstance(worker_meta, CSKProbeWorkerMetadata):
            self._engine.on_worker_decisions(worker_meta.decisions)

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        self._engine.on_finished(finished_req_ids)
        return None, None

    # ---- worker side -----------------------------------------------------

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        if not self._kv_caches_bound:
            caches = self._extract_kv_caches(forward_context)
            if caches:
                self._engine.register_kv_caches(caches)
                self._kv_caches_bound = True
        metadata = self._parent._get_connector_metadata()
        assert isinstance(metadata, CSKConnectorMetadata)
        model = getattr(forward_context, "model", None)
        self._engine.load(metadata.requests, model=model)

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        metadata = self._parent._get_connector_metadata()
        assert isinstance(metadata, CSKConnectorMetadata)
        if metadata.probes:
            self._engine.capture_probes(metadata.probes, layer_name, kv_layer)
        if metadata.saves:
            self._engine.capture_saves(metadata.saves, layer_name, kv_layer)

    def wait_for_save(self) -> None:
        self._engine.finalize_saves()

    def build_connector_worker_meta(self) -> CSKProbeWorkerMetadata | None:
        decisions = self._engine.decide_probes()
        return CSKProbeWorkerMetadata(decisions=decisions) if decisions else None

    def shutdown(self) -> None:
        return None

    # ---- vLLM extraction helpers ----------------------------------------

    @staticmethod
    def _request_token_ids(request: "Request") -> list[int]:
        """Materialize request tokens from vLLM's read-only token views."""

        return list(
            getattr(request, "all_token_ids", None) or request.prompt_token_ids or []
        )

    @staticmethod
    def _extract_kv_caches(
        forward_context: "ForwardContext",
    ) -> dict[str, torch.Tensor]:
        """Discover per-layer paged KV tensors from the active forward context."""

        caches: dict[str, torch.Tensor] = {}
        for layer_name, layer in getattr(
            forward_context, "no_compile_layers", {}
        ).items():
            kv_cache = getattr(layer, "kv_cache", None)
            if kv_cache is not None:
                caches[layer_name] = kv_cache[0]
        return caches
