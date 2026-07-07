from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.logger import init_logger

from cskcache.integration.vllm.utils import load_vllm_config
from cskcache.v1.matcher import SegmentCatalog, find_best_occurrence
from cskcache.v1.metadata import CSKCacheMode, CSKLoadPlan, SegmentOccurrence
from cskcache.v1.registry import CSKCacheRegistry, get_global_registry
from cskcache.v1.rope import find_rotary_embedding, rerotate_k_for_target_positions
from cskcache.v1.slot_ops import scatter_span

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request


logger = init_logger(__name__)


@dataclass(frozen=True)
class CSKReqMeta:
    plan: CSKLoadPlan
    block_ids: tuple[list[int], ...]


@dataclass
class CSKConnectorMetadata(KVConnectorMetadata):
    requests: list[CSKReqMeta] = field(default_factory=list)


class CSKCacheConnectorV1Impl:
    def __init__(self, vllm_config: "VllmConfig", role: Any, parent: Any) -> None:
        self._vllm_config = vllm_config
        self._role = role
        self._parent = parent
        self._block_size = vllm_config.cache_config.block_size
        self._config = load_vllm_config(vllm_config)
        self._registry: CSKCacheRegistry = get_global_registry()
        self._plans: dict[str, CSKLoadPlan] = {}
        self._allocated_blocks: dict[str, tuple[list[int], ...]] = {}
        self._kv_caches: dict[str, torch.Tensor] = {}
        self._pending_boundaries: dict[str, SegmentOccurrence] = {}
        if self._config.kv_dir is not None:
            loaded = self._registry.load_dir(self._config.kv_dir)
            logger.info("CSKCache loaded %d KV entries from %s", len(loaded), self._config.kv_dir)
        self._catalog: SegmentCatalog = SegmentCatalog.from_entries(
            self._registry.entries()
        )
        logger.info(
            "CSKCache connector initialized: role=%s catalog_segments=%d",
            role,
            len(self._catalog.segments),
        )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self._kv_caches = kv_caches

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        token_ids = list(getattr(request, "all_token_ids", None) or request.prompt_token_ids or [])
        occurrence = find_best_occurrence(
            self._catalog,
            token_ids,
            num_computed_tokens,
        )
        req_id = request.request_id
        self._plans.pop(req_id, None)
        self._pending_boundaries.pop(req_id, None)
        if occurrence is None:
            return 0, False
        if occurrence.end <= num_computed_tokens:
            return 0, False
        if occurrence.start > num_computed_tokens:
            # TODO(Boundary hook): expose this boundary to vLLM scheduler so
            # chunked prefill can stop at occurrence.start.  Until that hook
            # exists, returning 0 is conservative and avoids crossing into an
            # external-load region silently.
            self._pending_boundaries[req_id] = occurrence
            logger.debug(
                "CSKCache occurrence for request %s starts at %d after computed=%d; "
                "scheduler boundary hook not enabled yet",
                req_id,
                occurrence.start,
                num_computed_tokens,
            )
            return 0, False
        if occurrence.start < num_computed_tokens:
            logger.debug(
                "CSKCache occurrence for request %s was partially crossed: "
                "occurrence=[%d,%d), computed=%d",
                req_id,
                occurrence.start,
                occurrence.end,
                num_computed_tokens,
            )
            return 0, False
        entry = self._registry.get(occurrence.cache_id)
        if entry is None:
            logger.warning("CSKCache cache_id=%s matched but no KV entry is loaded", occurrence.cache_id)
            return 0, False
        target_token_ids = tuple(token_ids[occurrence.start : occurrence.end])
        if tuple(entry.token_ids) != target_token_ids:
            logger.warning("CSKCache token mismatch for cache_id=%s; skip load", occurrence.cache_id)
            return 0, False
        self._plans[req_id] = CSKLoadPlan(
            req_id=req_id,
            cache_id=occurrence.cache_id,
            mode=occurrence.mode,
            start=occurrence.start,
            end=occurrence.end,
            token_ids=target_token_ids,
        )
        return occurrence.length, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        req_id = request.request_id
        if num_external_tokens <= 0:
            self._plans.pop(req_id, None)
            self._allocated_blocks.pop(req_id, None)
            return
        plan = self._plans.get(req_id)
        if plan is None:
            raise RuntimeError(
                f"CSKCache allocated external tokens for {req_id} without a load plan"
            )
        if plan.length != num_external_tokens:
            raise RuntimeError(
                f"CSKCache plan length mismatch for {req_id}: "
                f"plan={plan.length}, allocated={num_external_tokens}"
            )
        self._allocated_blocks[req_id] = blocks.get_block_ids(allow_none=True)

    def build_connector_meta(self, scheduler_output: "SchedulerOutput") -> CSKConnectorMetadata:
        meta = CSKConnectorMetadata()
        for req_id in scheduler_output.num_scheduled_tokens:
            plan = self._plans.pop(req_id, None)
            blocks = self._allocated_blocks.pop(req_id, None)
            if plan is None or blocks is None:
                continue
            meta.requests.append(CSKReqMeta(plan=plan, block_ids=blocks))
        return meta

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        if not self._kv_caches:
            self._init_kv_caches_from_forward_context(forward_context)
        metadata = self._parent._get_connector_metadata()
        assert isinstance(metadata, CSKConnectorMetadata)
        if not metadata.requests:
            return
        rope = None
        model = getattr(forward_context, "model", None)
        for request in metadata.requests:
            plan = request.plan
            entry = self._registry.get(plan.cache_id)
            if entry is None:
                raise RuntimeError(f"CSKCache cache_id={plan.cache_id} is not loaded")
            if entry.length != plan.length:
                raise RuntimeError(
                    f"CSKCache length mismatch for {plan.cache_id}: "
                    f"entry={entry.length}, plan={plan.length}"
                )
            if tuple(entry.token_ids) != plan.token_ids:
                raise RuntimeError(f"CSKCache token mismatch for {plan.cache_id}")
            if not request.block_ids or request.block_ids[0] is None:
                raise RuntimeError(f"CSKCache load plan for {plan.req_id} has no blocks")
            for layer_name, (source_key, source_value) in entry.kv_by_layer.items():
                target_cache = self._kv_caches.get(layer_name)
                if target_cache is None:
                    continue
                key = source_key.to(target_cache.device)
                value = source_value.to(target_cache.device)
                if plan.mode == CSKCacheMode.ROPE_REUSE:
                    if entry.source_start != plan.start:
                        if rope is None and model is not None:
                            rope = find_rotary_embedding(model)
                        key = rerotate_k_for_target_positions(
                            key,
                            source_start=entry.source_start,
                            target_start=plan.start,
                            rope=rope,
                        )
                scatter_span(
                    target_cache,
                    request.block_ids[0],
                    plan.start,
                    plan.end,
                    self._block_size,
                    key,
                    value,
                )
            logger.info(
                "CSKCache loaded cache_id=%s request=%s target=[%d,%d) tokens=%d",
                plan.cache_id,
                plan.req_id,
                plan.start,
                plan.end,
                plan.length,
            )

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: Any, **kwargs: Any) -> None:
        # Save path is intentionally deferred. First version only validates
        # registry-derived load. TODO(B): when prompt metadata is available,
        # use it to save canonical segment KV without relying on oracle span xargs.
        return

    def wait_for_save(self) -> None:
        return

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        return None, None

    def shutdown(self) -> None:
        return None

    def _init_kv_caches_from_forward_context(self, forward_context: "ForwardContext") -> None:
        for layer_name, layer in getattr(forward_context, "no_compile_layers", {}).items():
            kv_cache = getattr(layer, "kv_cache", None)
            if kv_cache is not None:
                self._kv_caches[layer_name] = kv_cache[0]
