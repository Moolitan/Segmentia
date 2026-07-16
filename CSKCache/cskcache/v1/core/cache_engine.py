"""CSKCache engine: the vLLM-agnostic brain of the middleware.

This module owns every decision that used to live inside the vLLM adapter:
reuse-signal parsing, reuse span scheduling, the probe-gated state machine,
load-plan construction, KV scatter/gather, and gate aggregation. It operates
purely on plain Python data (token id lists, ints, mappings) and torch tensors,
so it can be constructed and tested without importing vLLM.

The integration layer (``integration/vllm/v1_adapter.py``) is a thin translator:
it extracts these plain values from vLLM ``Request`` / ``forward_context`` /
``KVCacheBlocks`` objects, calls the engine, and wraps the engine's plain result
carriers into vLLM's ``KVConnectorMetadata`` envelopes.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Iterable

import torch

from cskcache.v1.async_load import PrefetchRegistry, submit_disk_prefetch
from cskcache.v1.compute import CSKProbeAccumulator, CSKProbeDecision
from cskcache.logging import init_logger
from cskcache.profiling import LoadTrace, NullLoadTrace, Profiler
from cskcache.v1.core.config import CSKCacheConfig
from cskcache.v1.core.probe_state import CSKReuseStage, CSKReuseState
from cskcache.v1.kv_transfer.gpu_connector import (
    KVConnectorInterface,
    VLLMPagedGPUConnector,
)
from cskcache.v1.token.token_database import SegmentCatalog
from cskcache.v1.metadata import (
    CSKCacheEntry,
    CSKCacheMode,
    CSKCacheReuseSignal,
    CSKLoadPlan,
    CSKPrefetchHint,
    CSKProbeMeta,
    CSKReqMeta,
    CSKSaveMeta,
    ReuseSpan,
)
from cskcache.v1.storage import entry_nbytes
from cskcache.v1.storage.storage_manager import StorageManager

logger = init_logger(__name__)


@dataclass
class _PendingSave:
    cache_id: str
    start: int
    end: int
    token_ids: tuple[int, ...]
    overwrite: bool
    base_num_computed_tokens: int = 0
    block_ids: tuple[list[int], ...] | None = None


class CSKCacheEngine:
    """Orchestrates lookup, probe-gated reuse, load, and gating.

    Scheduler-side methods (``get_num_new_matched_tokens``,
    ``cap_prefill_before_reuse``, ``get_boundary_reuse_load_tokens``,
    ``update_reuse_after_alloc``, ``update_save_after_alloc``, ``build_meta``,
    ``on_worker_decisions``) run in the scheduler process. Worker-side methods
    (``register_kv_caches``, ``load``, ``capture_probes``, ``capture_saves``,
    ``finalize_saves``, ``decide_probes``) run around model forward. Both sides
    share only plain per-request state held here.
    """

    def __init__(
        self,
        config: CSKCacheConfig,
        storage: StorageManager,
        block_size: int,
        catalog: SegmentCatalog | None = None,
        gpu_connector: KVConnectorInterface | None = None,
        profiler: Profiler | None = None,
    ) -> None:
        self._config = config
        self._storage = storage
        self._block_size = block_size
        self._catalog = (
            catalog
            if catalog is not None
            else (
                SegmentCatalog([])
                if config.capture_only
                else SegmentCatalog.from_entries(storage.all_entries())
            )
        )
        # Device-side KV movement is delegated to the connector so the engine
        # stays free of paged-memory details.
        self._gpu: KVConnectorInterface = (
            gpu_connector
            if gpu_connector is not None
            else VLLMPagedGPUConnector(block_size)
        )
        self._profiler = profiler if profiler is not None else Profiler()
        # Scheduler-side one-shot / per-request state.
        self._plans: dict[str, CSKLoadPlan] = {}
        self._allocated_blocks: dict[str, tuple[list[int], ...]] = {}
        self._reuse_spans: dict[str, tuple[ReuseSpan, ...]] = {}
        self._pending_reuses: dict[str, ReuseSpan] = {}
        self._reuse_states: dict[str, CSKReuseState] = {}
        self._pending_saves: dict[str, _PendingSave] = {}
        self._request_prompt_lengths: dict[str, int] = {}
        self._request_initial_frontiers: dict[str, int] = {}
        # Worker-side gate accumulation (the star mechanism) stays in the engine.
        self._probe_accumulators: dict[str, CSKProbeAccumulator] = {}
        self._save_accumulators: dict[
            str, tuple[CSKSaveMeta, dict[str, tuple[torch.Tensor, torch.Tensor]]]
        ] = {}
        # Worker-side background disk prefetch, keyed by (req_id, cache_id).
        # Purely additive: nothing reads this unless capture_probes() finds a
        # handle waiting, and everything falls back to a synchronous
        # storage.get() if it doesn't.
        self._prefetch_registry = PrefetchRegistry()

    # ---- introspection ---------------------------------------------------

    @property
    def config(self) -> CSKCacheConfig:
        return self._config

    @property
    def catalog(self) -> SegmentCatalog:
        return self._catalog

    def refresh_catalog(self) -> None:
        """Rebuild the matcher index from the current storage contents."""

        self._catalog = SegmentCatalog.from_entries(self._storage.all_entries())

    # ---- scheduler side --------------------------------------------------

    def get_num_new_matched_tokens(
        self,
        req_id: str,
        token_ids: list[int],
        num_computed_tokens: int,
        kv_transfer_params: Mapping | None = None,
    ) -> tuple[int, bool]:
        """Prefix-style lookup for non-probe direct reuse.

        Returns (num_external_tokens, load_async). If the reusable span starts
        exactly at the frontier, returns its length so vLLM can treat it as
        externally computed. Otherwise records a boundary and returns 0.
        """
        if self._config.probe_enabled:
            return 0, False

        self._ensure_reuse_spans(
            req_id, token_ids, kv_transfer_params, num_computed_tokens
        )
        reuse = self._next_reuse(req_id, num_computed_tokens)
        self._plans.pop(req_id, None)
        self._pending_reuses.pop(req_id, None)
        if reuse is None:
            return 0, False
        if reuse.end <= num_computed_tokens:
            return 0, False
        if reuse.start > num_computed_tokens:
            self._pending_reuses[req_id] = reuse
            self._log_gap_scheduled(req_id, reuse, num_computed_tokens)
            return 0, False
        if reuse.start < num_computed_tokens:
            return 0, False
        self._plans[req_id] = self._make_load_plan(req_id, reuse, token_ids)
        self._log_load_plan(self._plans[req_id], "frontier")
        return reuse.length, False

    def cap_prefill_before_reuse(
        self,
        req_id: str,
        token_ids: list[int],
        base_num_computed_tokens: int,
        num_new_tokens: int,
        kv_transfer_params: Mapping | None = None,
    ) -> int:
        """Cap normal prefill so it stops at the next reuse boundary."""

        if num_new_tokens <= 0:
            return num_new_tokens

        if not self._config.probe_enabled:
            if req_id not in self._reuse_spans:
                self._ensure_reuse_spans(
                    req_id, token_ids, kv_transfer_params, base_num_computed_tokens
                )
            boundary = self._pending_reuses.get(req_id)
            base = base_num_computed_tokens
            if boundary is None or boundary.start < base:
                self._pending_reuses.pop(req_id, None)
                boundary = self._next_reuse(req_id, base)
                if boundary is None:
                    return num_new_tokens
                self._pending_reuses[req_id] = boundary
                self._log_gap_scheduled(req_id, boundary, base)
            if base < boundary.start:
                return min(num_new_tokens, boundary.start - base)
            if base < boundary.end:
                return 0
            return num_new_tokens

        state = self._get_or_create_reuse_state(
            req_id, token_ids, base_num_computed_tokens, kv_transfer_params
        )
        if state is None:
            return num_new_tokens

        base = base_num_computed_tokens
        if base < state.start:
            return min(num_new_tokens, state.start - base)
        if base > state.end:
            state.stage = CSKReuseStage.DONE
            return num_new_tokens

        if state.stage == CSKReuseStage.LOADING:
            # The bulk preload is dispatched from get_boundary_reuse_load_tokens,
            # which the scheduler checks before this method each step. Nothing
            # to really compute yet; wait for that to flip the stage forward.
            return 0

        if state.stage == CSKReuseStage.PROBING:
            if base == state.start and not state.gap_completed_logged:
                logger.info(
                    "gap prefill completed req_id=%s cache_id=%s frontier=%d",
                    req_id,
                    state.cache_id,
                    base,
                )
                state.gap_completed_logged = True
                self._profiler.mark_timeline(
                    req_id=req_id,
                    cache_id=state.cache_id,
                    target_start=state.start,
                    event="gap_completed",
                )
            if base >= state.probe_end:
                state.stage = CSKReuseStage.GATING
                return 0
            capped = min(num_new_tokens, state.probe_end - base)
            if not state.probe_scheduled_logged:
                logger.info(
                    "probe scheduled req_id=%s cache_id=%s skill=[%d,%d) "
                    "probe=[%d,%d) probe_tokens=%d recompute_end=%d "
                    "tokens_after_skill=%d",
                    req_id,
                    state.cache_id,
                    state.start,
                    state.end,
                    state.start,
                    state.probe_end,
                    state.probe_len,
                    state.anchor_end,
                    self._tokens_after_skill(req_id, state.end),
                )
                state.probe_scheduled_logged = True
                self._profiler.mark_timeline(
                    req_id=req_id,
                    cache_id=state.cache_id,
                    target_start=state.start,
                    event="probe_scheduled",
                    metadata={"probe_tokens": state.probe_len},
                )
            if base + capped >= state.probe_end:
                state.pending_capture = True
            return capped

        if state.stage == CSKReuseStage.GATING:
            return 0

        if state.stage == CSKReuseStage.RECOMPUTING:
            if base >= state.anchor_end:
                state.stage = CSKReuseStage.READY
                return 0
            capped = min(num_new_tokens, state.anchor_end - base)
            if not state.recompute_scheduled_logged:
                logger.info(
                    "recompute scheduled req_id=%s cache_id=%s "
                    "recompute=[%d,%d) recompute_tokens=%d",
                    req_id,
                    state.cache_id,
                    base,
                    state.anchor_end,
                    state.anchor_end - base,
                )
                state.recompute_scheduled_logged = True
                self._profiler.mark_timeline(
                    req_id=req_id,
                    cache_id=state.cache_id,
                    target_start=state.start,
                    event="recompute_scheduled",
                    metadata={"recompute_tokens": state.anchor_end - base},
                )
            if base + capped >= state.anchor_end:
                # Mirrors the PROBING branch: as soon as this step commits to
                # scheduling the last recompute chunk, it is safe to consider
                # recompute done -- vLLM guarantees scheduled tokens get
                # computed. build_meta() advances the stage the moment it
                # sees this, without waiting for a later confirming call here.
                state.pending_capture = True
            return capped

        if state.stage == CSKReuseStage.READY:
            return 0

        return num_new_tokens

    def get_boundary_reuse_load_tokens(
        self,
        req_id: str,
        token_ids: list[int],
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        """Return (num_tokens, advance_frontier) for a load dispatched
        without a real forward pass on the scheduler side this step.

        advance_frontier tells the caller whether to immediately bump
        request.num_computed_tokens by num_tokens. It is True everywhere
        except the probe-gated bulk preload (CSKReuseStage.LOADING): that
        call reserves blocks and scatters the *entire* span in one shot, but
        the frontier must stay put so vLLM still runs a real forward pass
        over the probe (and, if the gate fails, recompute) prefix -- that
        real forward naturally overwrites whatever was just scattered at
        those positions. Once the gate resolves (CSKReuseStage.READY), a
        second call here only confirms the frontier the rest of the way to
        state.end; nothing needs loading again because it is already there.
        """

        boundary = self._pending_reuses.get(req_id)
        if boundary is not None and num_computed_tokens == boundary.start:
            self._plans[req_id] = self._make_load_plan(req_id, boundary, token_ids)
            self._log_load_plan(self._plans[req_id], "boundary")
            logger.info(
                "gap prefill completed req_id=%s cache_id=%s frontier=%d",
                req_id,
                boundary.cache_id,
                num_computed_tokens,
            )
            self._profiler.mark_timeline(
                req_id=req_id,
                cache_id=boundary.cache_id,
                target_start=boundary.start,
                event="gap_completed",
            )
            self._pending_reuses.pop(req_id, None)
            return boundary.length, True

        if not self._config.probe_enabled:
            return 0, True
        state = self._reuse_states.get(req_id)
        if state is None:
            return 0, True

        if (
            state.stage == CSKReuseStage.LOADING
            and num_computed_tokens == state.start
        ):
            target_token_ids = tuple(token_ids[state.start : state.end])
            self._plans[req_id] = CSKLoadPlan(
                req_id=req_id,
                cache_id=state.cache_id,
                mode=CSKCacheMode.REUSE,
                start=state.start,
                end=state.end,
                token_ids=target_token_ids,
                source_offset=0,
                requires_scatter=True,
            )
            state.stage = CSKReuseStage.PROBING
            logger.info(
                "bulk preload dispatched req_id=%s cache_id=%s target=[%d,%d) "
                "tokens=%d probe_tokens=%d recompute_tokens=%d",
                req_id,
                state.cache_id,
                state.start,
                state.end,
                state.length,
                state.probe_len,
                state.anchor_len,
            )
            self._profiler.mark_timeline(
                req_id=req_id,
                cache_id=state.cache_id,
                target_start=state.start,
                event="bulk_preload_dispatched",
                metadata={"load_tokens": state.length},
            )
            return state.length, False

        if state.stage == CSKReuseStage.READY:
            length = state.end - num_computed_tokens
            if length <= 0:
                state.stage = CSKReuseStage.DONE
                return 0, True
            target_token_ids = tuple(token_ids[num_computed_tokens : state.end])
            self._plans[req_id] = CSKLoadPlan(
                req_id=req_id,
                cache_id=state.cache_id,
                mode=CSKCacheMode.REUSE,
                start=num_computed_tokens,
                end=state.end,
                token_ids=target_token_ids,
                source_offset=num_computed_tokens - state.start,
                requires_scatter=False,
            )
            logger.info(
                "reuse confirmed req_id=%s cache_id=%s frontier=%d "
                "recomputed_skill_tokens=%d target=[%d,%d) tokens=%d "
                "tokens_after_skill=%d",
                req_id,
                state.cache_id,
                num_computed_tokens,
                num_computed_tokens - state.start,
                num_computed_tokens,
                state.end,
                length,
                self._tokens_after_skill(req_id, state.end),
            )
            self._profiler.mark_timeline(
                req_id=req_id,
                cache_id=state.cache_id,
                target_start=state.start,
                event="reuse_confirmed",
            )
            self._log_load_plan(self._plans[req_id], "confirm")
            return length, True

        return 0, True

    def update_reuse_after_alloc(
        self,
        req_id: str,
        block_ids: tuple[list[int], ...],
        num_external_tokens: int,
    ) -> None:
        """Record physical block allocation for reuse and probe metadata."""
        self._allocated_blocks[req_id] = block_ids
        if num_external_tokens <= 0:
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

    def update_save_after_alloc(
        self,
        req_id: str,
        token_ids: list[int],
        num_computed_tokens: int,
        block_ids: tuple[list[int], ...],
        kv_transfer_params: Mapping | None,
    ) -> None:
        """Register or advance an offline save independently of reuse state."""

        pending = self._pending_saves.get(req_id)
        if pending is None:
            pending = self._parse_save_signal(token_ids, kv_transfer_params)
            if pending is None:
                return
            self._pending_saves[req_id] = pending
        pending.base_num_computed_tokens = num_computed_tokens
        pending.block_ids = block_ids

    def build_meta(
        self,
        num_scheduled_tokens: Mapping[str, int],
    ) -> tuple[
        list[CSKReqMeta], list[CSKProbeMeta], list[CSKSaveMeta], list[CSKPrefetchHint]
    ]:
        """Package scheduler decisions into plain worker carriers.

        Consumes one-shot ``_plans`` / ``_allocated_blocks`` so a plan is never
        replayed, and advances probe states that just finished a probe/anchor
        chunk. Also emits one ``CSKPrefetchHint`` per request the first time
        its probe state is seen here -- typically during gap prefill, well
        before the probe/anchor span is actually scheduled -- so the worker
        can start warming that cache_id's entry in the background instead of
        only discovering it needs it mid-forward-pass. The integration layer
        wraps the returned lists into vLLM's serializable
        ``KVConnectorMetadata``.
        """
        requests: list[CSKReqMeta] = []
        probes: list[CSKProbeMeta] = []
        saves: list[CSKSaveMeta] = []
        prefetch_hints: list[CSKPrefetchHint] = []
        for req_id, scheduled in num_scheduled_tokens.items():
            plan = self._plans.pop(req_id, None)
            blocks = self._allocated_blocks.pop(req_id, None)
            if plan is not None:
                if blocks is None:
                    raise RuntimeError(f"CSKCache load plan for {req_id} has no blocks")
                requests.append(CSKReqMeta(plan=plan, block_ids=blocks))
                logger.info(
                    "load dispatched req_id=%s cache_id=%s target=[%d,%d) "
                    "source_offset=%d tokens=%d blocks=%d "
                    "frontier_after_load=%d tokens_after_skill=%d",
                    req_id,
                    plan.cache_id,
                    plan.start,
                    plan.end,
                    plan.source_offset,
                    plan.length,
                    len(blocks[0]),
                    plan.end,
                    self._tokens_after_skill(req_id, self._skill_end(req_id, plan)),
                )
                self._profiler.mark_timeline(
                    req_id=req_id,
                    cache_id=plan.cache_id,
                    target_start=self._skill_start(req_id, plan),
                    event="load_dispatched",
                    metadata={
                        "load_tokens": plan.length,
                        "source_offset": plan.source_offset,
                    },
                )
                # Only the confirm plan (issued once the gate resolved) means
                # this reuse span is actually finished. The bulk-preload plan
                # also flows through here, but its state.stage is already
                # PROBING (set by get_boundary_reuse_load_tokens) and must
                # stay that way -- vLLM still owes it a real forward pass.
                state = self._reuse_states.get(req_id)
                if state is not None and state.stage == CSKReuseStage.READY:
                    state.stage = CSKReuseStage.DONE
                continue

            pending_save = self._pending_saves.get(req_id)
            if pending_save is not None:
                computed_after_step = (
                    pending_save.base_num_computed_tokens + scheduled
                )
                if computed_after_step >= pending_save.end:
                    save_blocks = pending_save.block_ids
                    if (
                        save_blocks is None
                        or not save_blocks
                        or save_blocks[0] is None
                    ):
                        raise RuntimeError(
                            f"CSKCache save plan for {req_id} has no blocks"
                        )
                    saves.append(
                        CSKSaveMeta(
                            req_id=req_id,
                            cache_id=pending_save.cache_id,
                            start=pending_save.start,
                            end=pending_save.end,
                            token_ids=pending_save.token_ids,
                            block_ids=save_blocks,
                            overwrite=pending_save.overwrite,
                        )
                    )
                    self._pending_saves.pop(req_id, None)
                continue

            state = self._reuse_states.get(req_id)
            if state is None:
                continue
            if not state.prefetch_hint_sent:
                prefetch_hints.append(
                    CSKPrefetchHint(req_id=req_id, cache_id=state.cache_id)
                )
                state.prefetch_hint_sent = True
            if scheduled <= 0 or blocks is None:
                continue
            if not state.pending_capture:
                continue
            if state.stage == CSKReuseStage.PROBING:
                probes.append(
                    CSKProbeMeta(
                        req_id=req_id,
                        cache_id=state.cache_id,
                        start=state.start,
                        end=state.probe_end,
                        source_offset=0,
                        block_ids=blocks,
                        tau=state.tau,
                        gate_metric=state.gate_metric,
                    )
                )
                state.pending_capture = False
                state.stage = CSKReuseStage.GATING
                self._profiler.mark_timeline(
                    req_id=req_id,
                    cache_id=state.cache_id,
                    target_start=state.start,
                    event="probe_dispatched",
                )
            elif state.stage == CSKReuseStage.RECOMPUTING:
                # The last recompute chunk was just committed to this step's
                # schedule; vLLM guarantees it will really run, so it is safe
                # to consider the gate resolved without waiting for a later
                # confirming call to cap_prefill_before_reuse.
                state.pending_capture = False
                state.stage = CSKReuseStage.READY
                self._profiler.mark_timeline(
                    req_id=req_id,
                    cache_id=state.cache_id,
                    target_start=state.start,
                    event="recompute_completed",
                )
        return requests, probes, saves, prefetch_hints

    def on_worker_decisions(self, decisions: Iterable[CSKProbeDecision]) -> None:
        """Advance reuse states from worker gate decisions (was update_connector_output)."""

        for decision in decisions:
            state = self._reuse_states.get(decision.req_id)
            if state is None or state.stage != CSKReuseStage.GATING:
                continue
            state.decision = decision
            self._profiler.mark_timeline(
                req_id=decision.req_id,
                cache_id=decision.cache_id,
                target_start=state.start,
                event="probe_decision_received",
                metadata={"passed": decision.passed},
            )
            if decision.passed:
                # Cached KV was already fresh; the bulk-preloaded tail from
                # probe_end onward needs nothing further.
                state.stage = CSKReuseStage.READY
            else:
                # Extend the real forward pass out to anchor_end; it will
                # overwrite the bulk-preloaded [probe_end, anchor_end) slice.
                state.stage = CSKReuseStage.RECOMPUTING
            logger.info(
                "probe decision req_id=%s cache_id=%s passed=%s "
                "gate=%.6f tau=%.6f metric=%s layers=%d "
                "k_mean=%.6f v_mean=%.6f kv_mean=%.6f "
                "next_stage=%s",
                decision.req_id,
                decision.cache_id,
                decision.passed,
                decision.metrics.gate_value,
                decision.tau,
                decision.metrics.gate_metric,
                decision.metrics.num_layers,
                decision.metrics.k_mean,
                decision.metrics.v_mean,
                decision.metrics.kv_mean,
                state.stage.value,
            )

    def on_finished(self, req_ids: Iterable[str]) -> None:
        for req_id in req_ids:
            spans = self._reuse_spans.get(req_id, ())
            state = self._reuse_states.get(req_id)
            final_stage = state.stage.value if state is not None else "none"
            had_state = any(
                req_id in mapping
                for mapping in (
                    self._reuse_states,
                    self._plans,
                    self._allocated_blocks,
                    self._probe_accumulators,
                    self._reuse_spans,
                    self._pending_reuses,
                    self._pending_saves,
                    self._save_accumulators,
                )
            )
            self._profiler.finish_request_timelines(req_id)
            self._reuse_states.pop(req_id, None)
            self._plans.pop(req_id, None)
            self._allocated_blocks.pop(req_id, None)
            self._probe_accumulators.pop(req_id, None)
            self._reuse_spans.pop(req_id, None)
            self._pending_reuses.pop(req_id, None)
            self._pending_saves.pop(req_id, None)
            self._save_accumulators.pop(req_id, None)
            self._request_prompt_lengths.pop(req_id, None)
            self._request_initial_frontiers.pop(req_id, None)
            if had_state:
                remaining = sum(
                    req_id in mapping
                    for mapping in (
                        self._reuse_states,
                        self._plans,
                        self._allocated_blocks,
                        self._probe_accumulators,
                        self._reuse_spans,
                        self._pending_reuses,
                        self._pending_saves,
                        self._save_accumulators,
                        self._request_prompt_lengths,
                        self._request_initial_frontiers,
                    )
                )
                logger.info(
                    "request finished req_id=%s reuse_entries=%d "
                    "final_reuse_stage=%s scheduler_state_remaining=%d",
                    req_id,
                    len(spans),
                    final_stage,
                    remaining,
                )

    # ---- worker side -----------------------------------------------------

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self._gpu.bind_kv_caches(kv_caches)
        logger.info("worker KV caches bound available_layers=%d", len(kv_caches))

    def register_model(self, model: object) -> None:
        """Register model state used to correct keys across positions."""

        self._gpu.set_model(model)
        logger.info("worker model registered model_type=%s", type(model).__name__)

    def load(
        self,
        requests: Iterable[CSKReqMeta],
        model: object | None = None,
    ) -> None:
        """Scatter cached K/V into the paged cache before forward.

        Validates cache availability, source slice bounds, and block assignment,
        then delegates the actual per-layer scatter to the KV connector.
        ``model`` is only used to locate the rotary embedding for
        cross-position keys.
        """

        # KV-only loads run outside model forward and receive no model in their
        # ForwardContext. Do not erase the RoPE object bound during worker init.
        if model is not None:
            self._gpu.set_model(model)
        for request in requests:
            plan = request.plan
            if not plan.requires_scatter:
                # The confirm plan issued once a probe-gated span's gate
                # resolved (CSKReuseStage.READY): [plan.start, plan.end) was
                # already scattered by the earlier bulk-preload plan and, for
                # whatever prefix vLLM's own forward pass touched, freshly
                # overwritten by that forward. Nothing to move; the scheduler
                # side already advanced the frontier.
                logger.info(
                    "KV already resident, confirming frontier only "
                    "req_id=%s cache_id=%s target=[%d,%d) tokens=%d",
                    plan.req_id,
                    plan.cache_id,
                    plan.start,
                    plan.end,
                    plan.length,
                )
                continue
            trace = self._profiler.start_worker_load(
                req_id=plan.req_id,
                cache_id=plan.cache_id,
                target_start=plan.start,
                target_end=plan.end,
                tokens=plan.length,
                source_offset=plan.source_offset,
            )
            entry = self._storage.get(plan.cache_id, trace=trace)
            if entry is None:
                raise RuntimeError(f"CSKCache cache_id={plan.cache_id} is not loaded")
            if plan.source_offset + plan.length > entry.length:
                raise RuntimeError(
                    f"CSKCache source slice mismatch for {plan.cache_id}: "
                    f"offset={plan.source_offset}, length={plan.length}, "
                    f"entry={entry.length}"
                )
            if not request.block_ids or request.block_ids[0] is None:
                raise RuntimeError(f"CSKCache load plan for {plan.req_id} has no blocks")
            if trace.enabled:
                full_entry_bytes = entry_nbytes(entry)
                trace.set(
                    bytes=full_entry_bytes * plan.length // entry.length,
                    entry_bytes=full_entry_bytes,
                    entry_tokens=entry.length,
                    expected_layers=len(entry.kv_by_layer),
                )
            prefetch_stream = None
            if self._config.gpu_prefetch_enabled:
                get_prefetch_stream = getattr(self._gpu, "get_prefetch_stream", None)
                if get_prefetch_stream is not None:
                    prefetch_stream = get_prefetch_stream()
            started = time.perf_counter()
            expected_layers, scattered_layers, skipped_layers = self._gpu.to_gpu(
                entry,
                plan,
                request.block_ids[0],
                trace=trace,
                prefetch_stream=prefetch_stream,
            )
            if scattered_layers != expected_layers or skipped_layers != 0:
                raise RuntimeError(
                    "CSKCache KV load layer accounting mismatch: "
                    f"req_id={plan.req_id} cache_id={plan.cache_id} "
                    f"expected_layers={expected_layers} "
                    f"scattered_layers={scattered_layers} "
                    f"skipped_layers={skipped_layers}"
                )
            elapsed_ms = (time.perf_counter() - started) * 1000
            source_start = entry.source_start + plan.source_offset
            source_end = source_start + plan.length
            logger.info(
                "KV load completed req_id=%s cache_id=%s source=[%d,%d) "
                "target=[%d,%d) rope_delta=%d tokens=%d "
                "expected_layers=%d scattered_layers=%d skipped_layers=%d "
                "elapsed_ms=%.3f",
                plan.req_id,
                plan.cache_id,
                source_start,
                source_end,
                plan.start,
                plan.end,
                plan.start - source_start,
                plan.length,
                expected_layers,
                scattered_layers,
                skipped_layers,
                elapsed_ms,
            )
            trace.set(
                scattered_layers=scattered_layers,
                skipped_layers=skipped_layers,
            )
            self._profiler.finish(trace)

    def submit_prefetch(self, req_id: str, cache_id: str) -> None:
        """Best-effort: start warming ``cache_id``'s entry in the background.

        Safe to call more than once for the same (req_id, cache_id) pair
        (deduplicated by ``PrefetchRegistry``), and safe for the handle to
        never be consumed (e.g. the request is aborted before its probe
        runs) -- an unclaimed handle is just dropped, no worse than today's
        behavior of never having started the read at all.
        """
        self._prefetch_registry.get_or_submit(
            (req_id, cache_id),
            lambda: submit_disk_prefetch(self._storage, cache_id),
        )

    def capture_probes(
        self,
        probes: Iterable[CSKProbeMeta],
        layer_name: str,
        kv_layer: torch.Tensor,
    ) -> None:
        """Gather recomputed probe K/V and accumulate residual vs cached K/V."""

        for probe in probes:
            trace = self._profiler.get_or_start_probe_capture(
                req_id=probe.req_id,
                cache_id=probe.cache_id,
                target_start=probe.start,
                target_end=probe.end,
            )
            handle = self._prefetch_registry.pop((probe.req_id, probe.cache_id))
            if handle is not None:
                with trace.cpu_stage("prefetch_wait"):
                    entry = handle.result()
            else:
                entry = self._storage.get(probe.cache_id, trace=trace)
            if entry is None or layer_name not in entry.kv_by_layer:
                continue
            if not probe.block_ids or probe.block_ids[0] is None:
                continue
            reuse_key, reuse_value = self._gpu.reuse_slice(
                entry,
                layer_name=layer_name,
                source_offset=probe.source_offset,
                length=probe.length,
                target_start=probe.start,
                device=kv_layer.device,
                trace=trace,
            )
            recompute_key, recompute_value = self._gpu.gather(
                kv_layer,
                probe.block_ids[0],
                probe.start,
                probe.end,
                trace=trace,
            )
            if trace.enabled:
                full_entry_bytes = entry_nbytes(entry)
                trace.set(
                    bytes=full_entry_bytes * probe.length // entry.length,
                    entry_bytes=full_entry_bytes,
                    entry_tokens=entry.length,
                )
            accumulator = self._probe_accumulators.get(probe.req_id)
            if accumulator is None:
                accumulator = CSKProbeAccumulator(
                    req_id=probe.req_id,
                    cache_id=probe.cache_id,
                    tau=probe.tau,
                    gate_metric=probe.gate_metric,
                )
                self._probe_accumulators[probe.req_id] = accumulator
            with trace.cuda_stage("residual", kv_layer.device):
                accumulator.add_layer(
                    layer_name,
                    reuse_key=reuse_key,
                    reuse_value=reuse_value,
                    recompute_key=recompute_key,
                    recompute_value=recompute_value,
                )

    def capture_saves(
        self,
        saves: Iterable[CSKSaveMeta],
        layer_name: str,
        kv_layer: torch.Tensor,
    ) -> None:
        """Gather one freshly-prefilled K/V layer for each pending save."""

        for save in saves:
            if not save.block_ids or save.block_ids[0] is None:
                raise RuntimeError(
                    f"CSKCache save plan for {save.req_id} has no block ids"
                )
            current = self._save_accumulators.get(save.req_id)
            if current is None:
                layers: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
                self._save_accumulators[save.req_id] = (save, layers)
            else:
                saved_meta, layers = current
                if saved_meta != save:
                    raise RuntimeError(
                        f"CSKCache conflicting save metadata for {save.req_id}"
                    )
            key, value = self._gpu.gather(
                kv_layer,
                save.block_ids[0],
                save.start,
                save.end,
            )
            layers[layer_name] = (
                key.detach().to(device="cpu"),
                value.detach().to(device="cpu"),
            )

    def finalize_saves(self) -> list[str]:
        """Persist all captured save entries and clear their worker state."""

        saved: list[str] = []
        for req_id, (meta, kv_by_layer) in list(self._save_accumulators.items()):
            if not kv_by_layer:
                raise RuntimeError(f"CSKCache save for {req_id} captured no KV layers")
            entry = CSKCacheEntry(
                cache_id=meta.cache_id,
                source_start=meta.start,
                source_end=meta.end,
                token_ids=list(meta.token_ids),
                kv_by_layer=kv_by_layer,
            )
            self._storage.put(entry, persist=True)
            saved.append(meta.cache_id)
            logger.info(
                "saved cache_id=%s source=[%d,%d) tokens=%d layers=%d",
                meta.cache_id,
                meta.start,
                meta.end,
                meta.length,
                len(kv_by_layer),
            )
            self._save_accumulators.pop(req_id, None)
        if saved:
            self.refresh_catalog()
        return saved

    def decide_probes(self) -> list[CSKProbeDecision]:
        """Turn accumulated probe residuals into gate decisions (worker->scheduler)."""

        if not self._probe_accumulators:
            return []
        decisions: list[CSKProbeDecision] = []
        for req_id, accumulator in list(self._probe_accumulators.items()):
            expected_count = 0
            captured_count = len(accumulator.layer_names)
            try:
                entry = self._storage.get(accumulator.cache_id)
                if entry is None:
                    raise RuntimeError(
                        f"CSK probe cache_id={accumulator.cache_id} is not loaded"
                    )
                expected = set(entry.kv_by_layer)
                expected_count = len(expected)
                captured_names = accumulator.layer_names
                captured = set(captured_names)
                missing = sorted(expected - captured)
                unexpected = sorted(captured - expected)
                duplicates = len(captured_names) - len(captured)
                if duplicates or missing or unexpected:
                    missing_preview = ",".join(missing[:8])
                    unexpected_preview = ",".join(unexpected[:8])
                    raise RuntimeError(
                        "CSK probe layer coverage mismatch: "
                        f"req_id={req_id} expected_layers={len(expected)} "
                        f"captured_layers={len(captured_names)} "
                        f"missing_layers={len(missing)} "
                        f"unexpected_layers={len(unexpected)} "
                        f"duplicate_layers={duplicates} "
                        f"missing=[{missing_preview}] "
                        f"unexpected=[{unexpected_preview}]"
                    )
                logger.info(
                    "probe captured req_id=%s cache_id=%s expected_layers=%d "
                    "captured_layers=%d missing_layers=0 "
                    "unexpected_layers=0 duplicate_layers=0",
                    req_id,
                    accumulator.cache_id,
                    len(expected),
                    len(captured_names),
                )
                decisions.append(accumulator.decide())
                self._profiler.finish_probe_capture(
                    req_id=req_id,
                    cache_id=accumulator.cache_id,
                    expected_layers=expected_count,
                    captured_layers=captured_count,
                )
            except Exception as exc:
                self._profiler.finish_probe_capture(
                    req_id=req_id,
                    cache_id=accumulator.cache_id,
                    expected_layers=expected_count,
                    captured_layers=captured_count,
                    status="error",
                    error=str(exc),
                )
                raise
            finally:
                self._profiler.discard_probe_capture(
                    req_id=req_id, cache_id=accumulator.cache_id
                )
                self._probe_accumulators.pop(req_id, None)
        return decisions

    # ---- internals -------------------------------------------------------

    def _ensure_reuse_spans(
        self,
        req_id: str,
        token_ids: list[int],
        kv_transfer_params: Mapping | None,
        num_computed_tokens: int | None = None,
    ) -> tuple[ReuseSpan, ...]:
        spans = self._reuse_spans.get(req_id)
        if spans is not None:
            return spans
        spans = self._resolve_reuses(req_id, token_ids, kv_transfer_params)
        self._reuse_spans[req_id] = spans
        if spans:
            frontier = 0 if num_computed_tokens is None else num_computed_tokens
            self._request_prompt_lengths[req_id] = len(token_ids)
            self._request_initial_frontiers[req_id] = frontier
            rendered = ", ".join(
                f"{span.cache_id}:[{span.start},{span.end})" for span in spans
            )
            logger.info(
                "reuse signal accepted req_id=%s prompt_tokens=%d "
                "prefix_frontier=%d entries=%d [%s]",
                req_id,
                len(token_ids),
                frontier,
                len(spans),
                rendered,
            )
            previous_end = frontier
            for index, span in enumerate(spans, start=1):
                self._profiler.register_timeline(
                    req_id=req_id,
                    cache_id=span.cache_id,
                    target_start=span.start,
                    target_end=span.end,
                    metadata={
                        "prompt_tokens": len(token_ids),
                        "prefix_frontier": frontier,
                        "tokens_after_skill": len(token_ids) - span.end,
                    },
                )
                logger.info(
                    "reuse entry layout req_id=%s entry=%d/%d cache_id=%s "
                    "target=[%d,%d) skill_tokens=%d prefix_frontier=%d "
                    "gap_from_frontier=%d gap_from_previous_entry=%d "
                    "tokens_after_skill=%d",
                    req_id,
                    index,
                    len(spans),
                    span.cache_id,
                    span.start,
                    span.end,
                    span.length,
                    frontier,
                    span.start - frontier,
                    span.start - previous_end,
                    len(token_ids) - span.end,
                )
                previous_end = span.end
        return spans

    def _parse_save_signal(
        self,
        token_ids: list[int],
        kv_transfer_params: Mapping | None,
    ) -> _PendingSave | None:
        """Parse one explicit offline-save operation into scheduler state."""

        if not isinstance(kv_transfer_params, Mapping):
            return None
        raw = kv_transfer_params.get("cskcache")
        if not isinstance(raw, Mapping) or raw.get("operation", "reuse") != "save":
            return None
        if raw.get("enabled", True) is False:
            return None

        cache_id = raw.get("cache_id")
        if not isinstance(cache_id, str) or not cache_id:
            raise ValueError("CSKCache save signal requires a non-empty cache_id")
        start = raw.get("source_start")
        end = raw.get("source_end")
        if isinstance(start, bool) or not isinstance(start, int):
            raise ValueError("CSKCache save signal source_start must be an int")
        if isinstance(end, bool) or not isinstance(end, int):
            raise ValueError("CSKCache save signal source_end must be an int")
        if start < 0 or start >= end or end > len(token_ids):
            raise ValueError(
                f"CSKCache invalid save span [{start},{end}) for "
                f"token_count={len(token_ids)}"
            )
        overwrite = raw.get("overwrite", False)
        if not isinstance(overwrite, bool):
            raise ValueError("CSKCache save signal overwrite must be a bool")
        if self._storage.contains(cache_id) and not overwrite:
            logger.info("save skipped existing cache_id=%s", cache_id)
            return None

        return _PendingSave(
            cache_id=cache_id,
            start=start,
            end=end,
            token_ids=tuple(token_ids[start:end]),
            overwrite=overwrite,
        )

    def _resolve_reuses(
        self,
        req_id: str,
        token_ids: list[int],
        kv_transfer_params: Mapping | None,
    ) -> tuple[ReuseSpan, ...]:
        """Resolve and order request-local spans from explicit signals only."""

        signals = self._parse_reuse_signals(kv_transfer_params)
        resolved: list[ReuseSpan] = []
        for signal in signals:
            if not signal.enabled:
                continue
            trace = self._profiler.start_scheduler_lookup(
                req_id=req_id,
                cache_id=signal.cache_id,
                target_start=signal.target_start,
                target_end=signal.target_end,
            )
            span = self._reuse_from_signal(token_ids, signal, trace=trace)
            self._profiler.finish(trace)
            resolved.append(span)
        spans = sorted(resolved, key=lambda span: (span.start, span.end, span.cache_id))
        for previous, current in zip(spans, spans[1:]):
            if current.start < previous.end:
                raise ValueError(
                    "CSKCache reuse spans must not overlap: "
                    f"{previous.cache_id}=[{previous.start},{previous.end}) and "
                    f"{current.cache_id}=[{current.start},{current.end})"
                )
        return tuple(spans)

    def _parse_reuse_signals(
        self,
        kv_transfer_params: Mapping | None,
    ) -> tuple[CSKCacheReuseSignal, ...]:
        if not isinstance(kv_transfer_params, Mapping):
            return ()
        raw = kv_transfer_params.get("cskcache")
        if raw is None:
            return ()
        if not isinstance(raw, Mapping):
            raise ValueError("CSKCache reuse signal must be a mapping")
        if raw.get("operation", "reuse") == "save":
            return ()

        enabled_raw = raw.get("enabled", True)
        if not isinstance(enabled_raw, bool):
            raise ValueError("CSKCache reuse signal enabled must be a bool")
        if not enabled_raw:
            return ()

        has_entries = "entries" in raw
        has_single = "cache_id" in raw
        if has_entries and has_single:
            raise ValueError(
                "CSKCache reuse signal cannot contain both cache_id and entries"
            )
        if has_entries:
            entries = raw.get("entries")
            if not isinstance(entries, list) or not entries:
                raise ValueError(
                    "CSKCache reuse signal entries must be a non-empty list"
                )
            signals: list[CSKCacheReuseSignal] = []
            for index, entry in enumerate(entries):
                if not isinstance(entry, Mapping):
                    raise ValueError(
                        f"CSKCache reuse signal entries[{index}] must be a mapping"
                    )
                signals.append(self._parse_reuse_entry(entry, f"entries[{index}]"))
            return tuple(signals)

        return (self._parse_reuse_entry(raw, "reuse signal"),)

    def _parse_reuse_entry(
        self,
        raw: Mapping,
        label: str,
    ) -> CSKCacheReuseSignal:
        cache_id = raw.get("cache_id")
        if not isinstance(cache_id, str) or not cache_id:
            raise ValueError(f"CSKCache {label} requires a non-empty cache_id")

        return CSKCacheReuseSignal(
            enabled=True,
            cache_id=cache_id,
            target_start=self._parse_optional_int(
                raw.get("target_start"), f"{label}.target_start"
            ),
            target_end=self._parse_optional_int(
                raw.get("target_end"), f"{label}.target_end"
            ),
        )

    @staticmethod
    def _parse_optional_int(value: object, name: str) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"CSKCache reuse signal {name} must be an int")
        return value

    def _reuse_from_signal(
        self,
        token_ids: list[int],
        signal: CSKCacheReuseSignal,
        trace: LoadTrace | NullLoadTrace | None = None,
    ) -> ReuseSpan:
        # The scheduler only needs the entry's length (to validate the reuse
        # span) and its byte size (for trace metadata) — never the KV
        # tensors themselves. get_metadata() answers this from the on-disk
        # sidecar/CPU-tier entry without deserializing the full payload,
        # unlike a plain get(). The worker's own load() call further down
        # the pipeline still does a real get() when it actually needs the
        # tensors.
        metadata = self._storage.get_metadata(signal.cache_id, trace=trace)
        if metadata is None:
            raise RuntimeError(
                f"CSKCache reuse signal cache_id={signal.cache_id} is not loaded"
            )
        entry_length, entry_bytes = metadata

        if signal.target_start is None or signal.target_end is None:
            raise ValueError(
                "CSKCache reuse signal requires target_start and target_end"
            )
        target_start = signal.target_start
        target_end = signal.target_end

        if (
            target_start < 0
            or target_start >= target_end
            or target_end > len(token_ids)
        ):
            raise RuntimeError(
                f"CSKCache reuse signal provides invalid target_start and target_end "
                f"target=[{target_start},{target_end}), token_count={len(token_ids)}"
            )
        if target_end - target_start != entry_length:
            raise RuntimeError(
                f"CSKCache reuse span length mismatch for {signal.cache_id}: "
                f"target_length={target_end - target_start}, entry_length={entry_length}"
            )
        if trace is not None and trace.enabled:
            trace.set(
                bytes=entry_bytes,
                entry_bytes=entry_bytes,
                entry_tokens=entry_length,
            )

        return ReuseSpan(
            cache_id=signal.cache_id,
            start=target_start,
            end=target_end,
            mode=CSKCacheMode.REUSE,
        )

    def _next_reuse(
        self,
        req_id: str,
        num_computed_tokens: int,
    ) -> ReuseSpan | None:
        for reuse in self._reuse_spans.get(req_id, ()):
            if reuse.start >= num_computed_tokens:
                return reuse
        return None

    @staticmethod
    def _make_load_plan(
        req_id: str,
        reuse: ReuseSpan,
        token_ids: list[int],
        source_offset: int = 0,
    ) -> CSKLoadPlan:
        return CSKLoadPlan(
            req_id=req_id,
            cache_id=reuse.cache_id,
            mode=reuse.mode,
            start=reuse.start,
            end=reuse.end,
            token_ids=tuple(token_ids[reuse.start : reuse.end]),
            source_offset=source_offset,
        )

    def _log_load_plan(self, plan: CSKLoadPlan, trigger: str) -> None:
        spans = self._reuse_spans.get(plan.req_id, ())
        entry_index = next(
            (
                index
                for index, span in enumerate(spans, start=1)
                if span.cache_id == plan.cache_id
                and span.start <= plan.start
                and span.end == plan.end
            ),
            0,
        )
        logger.info(
            "reuse boundary selected req_id=%s cache_id=%s entry=%d/%d trigger=%s "
            "target=[%d,%d) source_offset=%d tokens=%d "
            "recomputed_skill_tokens=%d frontier_after_load=%d "
            "tokens_after_skill=%d",
            plan.req_id,
            plan.cache_id,
            entry_index,
            len(spans),
            trigger,
            plan.start,
            plan.end,
            plan.source_offset,
            plan.length,
            plan.start - (spans[entry_index - 1].start if entry_index else plan.start),
            plan.end,
            self._tokens_after_skill(plan.req_id, self._skill_end(plan.req_id, plan)),
        )
        self._profiler.mark_timeline(
            req_id=plan.req_id,
            cache_id=plan.cache_id,
            target_start=self._skill_start(plan.req_id, plan),
            event="load_planned",
            metadata={
                "trigger": trigger,
                "load_tokens": plan.length,
                "source_offset": plan.source_offset,
            },
        )

    def _get_or_create_reuse_state(
        self,
        req_id: str,
        token_ids: list[int],
        num_computed_tokens: int,
        kv_transfer_params: Mapping | None,
    ) -> CSKReuseState | None:
        state = self._reuse_states.get(req_id)
        if state is not None and state.stage != CSKReuseStage.DONE:
            return state

        if req_id not in self._reuse_spans:
            self._ensure_reuse_spans(
                req_id, token_ids, kv_transfer_params, num_computed_tokens
            )
        reuse = self._next_reuse(req_id, num_computed_tokens)
        if reuse is None:
            return None

        length = reuse.length
        state = CSKReuseState(
            req_id=req_id,
            cache_id=reuse.cache_id,
            start=reuse.start,
            end=reuse.end,
            probe_len=min(self._config.probe_tokens, length),
            anchor_len=min(self._config.anchor_tokens, length),
            tau=self._config.probe_tau,
            gate_metric=self._config.gate_metric,
        )
        self._reuse_states[req_id] = state
        self._log_gap_scheduled(req_id, reuse, num_computed_tokens)
        return state

    def _log_gap_scheduled(
        self,
        req_id: str,
        reuse: ReuseSpan,
        frontier: int,
    ) -> None:
        gap = max(reuse.start - frontier, 0)
        if gap <= 0:
            return
        logger.info(
            "gap prefill scheduled req_id=%s cache_id=%s frontier=[%d,%d) "
            "gap_tokens=%d next_skill_start=%d",
            req_id,
            reuse.cache_id,
            frontier,
            reuse.start,
            gap,
            reuse.start,
        )
        self._profiler.mark_timeline(
            req_id=req_id,
            cache_id=reuse.cache_id,
            target_start=reuse.start,
            event="gap_scheduled",
            metadata={"gap_tokens": gap, "frontier": frontier},
        )

    def _tokens_after_skill(self, req_id: str, skill_end: int) -> int:
        return max(self._request_prompt_lengths.get(req_id, skill_end) - skill_end, 0)

    def _skill_end(self, req_id: str, plan: CSKLoadPlan) -> int:
        for span in self._reuse_spans.get(req_id, ()):
            if span.cache_id == plan.cache_id and span.start <= plan.start < span.end:
                return span.end
        return plan.end

    def _skill_start(self, req_id: str, plan: CSKLoadPlan) -> int:
        for span in self._reuse_spans.get(req_id, ()):
            if span.cache_id == plan.cache_id and span.start <= plan.start < span.end:
                return span.start
        return plan.start
