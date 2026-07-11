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

import logging
from collections.abc import Mapping
from typing import Iterable

import torch

from cskcache.v1.compute import CSKProbeAccumulator, CSKProbeDecision
from cskcache.v1.core.config import CSKCacheConfig
from cskcache.v1.core.probe_state import CSKProbePhase, CSKProbeState
from cskcache.v1.kv_transfer.gpu_connector import (
    KVConnectorInterface,
    VLLMPagedGPUConnector,
)
from cskcache.v1.token.token_database import SegmentCatalog
from cskcache.v1.metadata import (
    CSKCacheMode,
    CSKCacheReuseSignal,
    CSKLoadPlan,
    CSKProbeMeta,
    CSKReqMeta,
    ReuseSpan,
)
from cskcache.v1.storage.storage_manager import StorageManager

# Standard logging keeps the engine importable without vLLM.
logger = logging.getLogger(__name__)


class CSKCacheEngine:
    """Orchestrates lookup, probe-gated reuse, load, and gating.

    Scheduler-side methods (``get_num_new_matched_tokens``,
    ``cap_prefill_before_reuse``, ``get_boundary_reuse_load_tokens``,
    ``update_state_after_alloc``, ``build_meta``, ``on_worker_decisions``) run
    in the scheduler process. Worker-side methods
    (``register_kv_caches``, ``load``, ``capture_probes``, ``decide_probes``) run
    around model forward. Both sides share only plain per-request state held here.
    """

    def __init__(
        self,
        config: CSKCacheConfig,
        storage: StorageManager,
        block_size: int,
        catalog: SegmentCatalog | None = None,
        gpu_connector: KVConnectorInterface | None = None,
    ) -> None:
        self._config = config
        self._storage = storage
        self._block_size = block_size
        self._catalog = (
            catalog
            if catalog is not None
            else SegmentCatalog.from_entries(storage.all_entries())
        )
        # Device-side KV movement is delegated to the connector so the engine
        # stays free of paged-memory details.
        self._gpu: KVConnectorInterface = (
            gpu_connector
            if gpu_connector is not None
            else VLLMPagedGPUConnector(block_size)
        )
        # Scheduler-side one-shot / per-request state.
        self._plans: dict[str, CSKLoadPlan] = {}
        self._allocated_blocks: dict[str, tuple[list[int], ...]] = {}
        self._pending_reuses: dict[str, ReuseSpan] = {}
        self._probe_states: dict[str, CSKProbeState] = {}
        # Worker-side gate accumulation (the star mechanism) stays in the engine.
        self._probe_accumulators: dict[str, CSKProbeAccumulator] = {}

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

        reuse = self._resolve_reuse(token_ids, kv_transfer_params)
        self._plans.pop(req_id, None)
        self._pending_reuses.pop(req_id, None)
        if reuse is None:
            return 0, False
        if reuse.end <= num_computed_tokens:
            return 0, False
        if reuse.start > num_computed_tokens:
            self._pending_reuses[req_id] = reuse
            return 0, False
        if reuse.start < num_computed_tokens:
            return 0, False
        self._plans[req_id] = self._make_load_plan(req_id, reuse, token_ids)
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
            boundary = self._pending_reuses.get(req_id)
            if boundary is not None:
                base = base_num_computed_tokens
                if base < boundary.start:
                    return min(num_new_tokens, boundary.start - base)
                if base < boundary.end:
                    return 0
                self._pending_reuses.pop(req_id, None)
            return num_new_tokens

        state = self._get_or_create_probe_state(
            req_id, token_ids, base_num_computed_tokens, kv_transfer_params
        )
        if state is None:
            return num_new_tokens

        base = base_num_computed_tokens
        if base < state.start:
            return min(num_new_tokens, state.start - base)
        if base > state.end:
            state.phase = CSKProbePhase.DONE
            return num_new_tokens

        if state.phase == CSKProbePhase.NEED_PROBE:
            if base >= state.probe_end:
                state.phase = CSKProbePhase.WAIT_PROBE
                return 0
            capped = min(num_new_tokens, state.probe_end - base)
            if base + capped >= state.probe_end:
                state.pending_capture = "probe"
            return capped

        if state.phase == CSKProbePhase.WAIT_PROBE:
            return 0

        if state.phase == CSKProbePhase.NEED_ANCHOR:
            if base >= state.anchor_end:
                state.phase = CSKProbePhase.NEED_LOAD
                state.load_start = state.anchor_end
                return 0
            capped = min(num_new_tokens, state.anchor_end - base)
            if base + capped >= state.anchor_end:
                state.pending_capture = "anchor"
            return capped

        if state.phase == CSKProbePhase.NEED_LOAD:
            return 0

        return num_new_tokens

    def get_boundary_reuse_load_tokens(
        self,
        req_id: str,
        token_ids: list[int],
        num_computed_tokens: int,
    ) -> int:
        """Return how many boundary tokens should be loaded without forward."""

        boundary = self._pending_reuses.get(req_id)
        if boundary is not None and num_computed_tokens == boundary.start:
            self._plans[req_id] = self._make_load_plan(req_id, boundary, token_ids)
            self._pending_reuses.pop(req_id, None)
            return boundary.length

        if not self._config.probe_enabled:
            return 0
        state = self._probe_states.get(req_id)
        if state is None or state.phase != CSKProbePhase.NEED_LOAD:
            return 0
        load_start = state.load_start
        if load_start is None or num_computed_tokens != load_start:
            return 0
        length = state.end - load_start
        if length <= 0:
            state.phase = CSKProbePhase.DONE
            return 0
        target_token_ids = tuple(token_ids[load_start : state.end])
        self._plans[req_id] = CSKLoadPlan(
            req_id=req_id,
            cache_id=state.cache_id,
            mode=CSKCacheMode.REUSE,
            start=load_start,
            end=state.end,
            token_ids=target_token_ids,
            source_offset=load_start - state.start,
        )
        return length

    def update_state_after_alloc(
        self,
        req_id: str,
        block_ids: tuple[list[int], ...],
        num_external_tokens: int,
    ) -> None:
        """Record vLLM's physical block allocation for later worker metadata."""
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

    def build_meta(
        self,
        num_scheduled_tokens: Mapping[str, int],
    ) -> tuple[list[CSKReqMeta], list[CSKProbeMeta]]:
        """Package scheduler decisions into plain worker carriers.

        Consumes one-shot ``_plans`` / ``_allocated_blocks`` so a plan is never
        replayed, and advances probe states that just finished a probe/anchor
        chunk. The integration layer wraps the returned lists into vLLM's
        serializable ``KVConnectorMetadata``.
        """
        requests: list[CSKReqMeta] = []
        probes: list[CSKProbeMeta] = []
        for req_id, scheduled in num_scheduled_tokens.items():
            plan = self._plans.pop(req_id, None)
            blocks = self._allocated_blocks.pop(req_id, None)
            if plan is not None:
                if blocks is None:
                    raise RuntimeError(f"CSKCache load plan for {req_id} has no blocks")
                requests.append(CSKReqMeta(plan=plan, block_ids=blocks))
                state = self._probe_states.get(req_id)
                if state is not None:
                    state.phase = CSKProbePhase.DONE
                continue

            state = self._probe_states.get(req_id)
            if state is None:
                continue
            if scheduled <= 0 or blocks is None:
                continue
            if state.pending_capture == "probe":
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
                state.pending_capture = None
                state.phase = CSKProbePhase.WAIT_PROBE
            elif state.pending_capture == "anchor":
                state.pending_capture = None
                state.phase = CSKProbePhase.NEED_LOAD
                state.load_start = state.anchor_end
        return requests, probes

    def on_worker_decisions(self, decisions: Iterable[CSKProbeDecision]) -> None:
        """Advance probe states from worker gate decisions (was update_connector_output)."""

        for decision in decisions:
            state = self._probe_states.get(decision.req_id)
            if state is None or state.phase != CSKProbePhase.WAIT_PROBE:
                continue
            state.decision = decision
            if decision.passed:
                state.phase = CSKProbePhase.NEED_LOAD
                state.load_start = state.probe_end
            else:
                state.phase = CSKProbePhase.NEED_ANCHOR
            logger.info(
                "CSKCache probe decision request=%s cache_id=%s passed=%s "
                "gate=%.6f tau=%.6f metric=%s layers=%d",
                decision.req_id,
                decision.cache_id,
                decision.passed,
                decision.metrics.gate_value,
                decision.tau,
                decision.metrics.gate_metric,
                decision.metrics.num_layers,
            )

    def on_finished(self, req_ids: Iterable[str]) -> None:
        for req_id in req_ids:
            self._probe_states.pop(req_id, None)
            self._plans.pop(req_id, None)
            self._allocated_blocks.pop(req_id, None)
            self._probe_accumulators.pop(req_id, None)
            self._pending_reuses.pop(req_id, None)

    # ---- worker side -----------------------------------------------------

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self._gpu.bind_kv_caches(kv_caches)

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

        self._gpu.set_model(model)
        for request in requests:
            plan = request.plan
            entry = self._storage.get(plan.cache_id)
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
            self._gpu.to_gpu(entry, plan, request.block_ids[0])

    def capture_probes(
        self,
        probes: Iterable[CSKProbeMeta],
        layer_name: str,
        kv_layer: torch.Tensor,
    ) -> None:
        """Gather recomputed probe K/V and accumulate residual vs cached K/V."""

        for probe in probes:
            entry = self._storage.get(probe.cache_id)
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
            )
            recompute_key, recompute_value = self._gpu.gather(
                kv_layer,
                probe.block_ids[0],
                probe.start,
                probe.end,
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
            accumulator.add_layer(
                layer_name,
                reuse_key=reuse_key,
                reuse_value=reuse_value,
                recompute_key=recompute_key,
                recompute_value=recompute_value,
            )

    def decide_probes(self) -> list[CSKProbeDecision]:
        """Turn accumulated probe residuals into gate decisions (worker->scheduler)."""

        if not self._probe_accumulators:
            return []
        decisions: list[CSKProbeDecision] = []
        for req_id, accumulator in list(self._probe_accumulators.items()):
            try:
                decisions.append(accumulator.decide())
            except RuntimeError as exc:
                logger.warning("CSKCache probe decision skipped for %s: %s", req_id, exc)
            finally:
                self._probe_accumulators.pop(req_id, None)
        return decisions

    # ---- internals -------------------------------------------------------

    def _resolve_reuse(
        self,
        token_ids: list[int],
        kv_transfer_params: Mapping | None,
    ) -> ReuseSpan | None:
        """Resolve the request-local span from an explicit reuse signal only."""

        signal = self._parse_reuse_signal(kv_transfer_params)
        if signal is None or not signal.enabled:
            return None
        return self._reuse_from_signal(token_ids, signal)

    def _parse_reuse_signal(
        self,
        kv_transfer_params: Mapping | None,
    ) -> CSKCacheReuseSignal | None:
        if not isinstance(kv_transfer_params, Mapping):
            return None
        raw = kv_transfer_params.get("cskcache")
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise ValueError("CSKCache reuse signal must be a mapping")

        enabled_raw = raw.get("enabled", True)
        if not isinstance(enabled_raw, bool):
            raise ValueError("CSKCache reuse signal enabled must be a bool")
        if not enabled_raw:
            return CSKCacheReuseSignal(enabled=False, cache_id="")

        cache_id = raw.get("cache_id")
        if not isinstance(cache_id, str) or not cache_id:
            raise ValueError("CSKCache reuse signal requires a non-empty cache_id")

        return CSKCacheReuseSignal(
            enabled=True,
            cache_id=cache_id,
            target_start=self._parse_optional_int(raw.get("target_start"), "target_start"),
            target_end=self._parse_optional_int(raw.get("target_end"), "target_end"),
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
    ) -> ReuseSpan:
        entry = self._storage.get(signal.cache_id)
        if entry is None:
            raise RuntimeError(
                f"CSKCache reuse signal cache_id={signal.cache_id} is not loaded"
            )

        if signal.target_start is None or signal.target_end is None:
            raise ValueError(
                "CSKCache reuse signal requires target_start and target_end"
            )
        target_start = signal.target_start
        target_end = signal.target_end

        if target_start < 0 or target_start >= target_end:
            raise RuntimeError(
                f"CSKCache reuse signal provides invalid target_start and target_end "
                f"target=[{target_start},{target_end}), token_count={len(token_ids)}"
            )
        if target_end - target_start != entry.length:
            raise RuntimeError(
                f"CSKCache reuse span length mismatch for {signal.cache_id}: "
                f"target_length={target_end - target_start}, entry_length={entry.length}"
            )

        return ReuseSpan(
            cache_id=signal.cache_id,
            start=target_start,
            end=target_end,
            mode=CSKCacheMode.REUSE,
        )

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

    def _get_or_create_probe_state(
        self,
        req_id: str,
        token_ids: list[int],
        num_computed_tokens: int,
        kv_transfer_params: Mapping | None,
    ) -> CSKProbeState | None:
        state = self._probe_states.get(req_id)
        if state is not None and state.phase != CSKProbePhase.DONE:
            return state

        reuse = self._resolve_reuse(token_ids, kv_transfer_params)
        if reuse is None:
            return None
        if reuse.end <= num_computed_tokens:
            return None
        if reuse.start < num_computed_tokens:
            return None

        length = reuse.length
        state = CSKProbeState(
            req_id=req_id,
            cache_id=reuse.cache_id,
            start=reuse.start,
            end=reuse.end,
            probe_len=min(self._config.probe_tokens, length),
            anchor_len=min(self._config.anchor_tokens, length),
            tau=self._config.probe_tau,
            gate_metric=self._config.gate_metric,
        )
        self._probe_states[req_id] = state
        return state
