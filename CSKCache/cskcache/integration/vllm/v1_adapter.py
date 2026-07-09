from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
import os
import sys
from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)
from vllm.logger import init_logger
from vllm.v1.outputs import KVConnectorOutput

from cskcache.integration.vllm.utils import load_vllm_config
from cskcache.v1.compute import CSKProbeAccumulator, CSKProbeDecision
from cskcache.v1.compute.reuse import prepare_reuse_slice
from cskcache.v1.matcher import SegmentCatalog, find_best_occurrence
from cskcache.v1.metadata import (
    CSKCacheDirectivePlacement,
    CSKCacheMode,
    CSKCacheRequestDirective,
    CSKLoadPlan,
    SegmentOccurrence,
)
from cskcache.v1.registry import CSKCacheRegistry, get_global_registry
from cskcache.v1.rope import find_rotary_embedding
from cskcache.v1.slot_ops import gather_span, scatter_span

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.request import Request


logger = init_logger(__name__)


def _trace(message: str, *args: object) -> None:
    if args:
        message = message % args
    print(f"CSKCacheTRACE pid={os.getpid()} {message}", file=sys.stderr, flush=True)


class CSKProbePhase(str, Enum):
    NEED_PROBE = "need_probe"
    WAIT_PROBE = "wait_probe"
    NEED_ANCHOR = "need_anchor"
    NEED_LOAD = "need_load"
    DONE = "done"


@dataclass
class CSKProbeState:
    req_id: str
    cache_id: str
    start: int
    end: int
    probe_len: int
    anchor_len: int
    tau: float
    gate_metric: str
    phase: CSKProbePhase = CSKProbePhase.NEED_PROBE
    pending_capture: str | None = None
    load_start: int | None = None
    decision: CSKProbeDecision | None = None

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def probe_end(self) -> int:
        return self.start + self.probe_len

    @property
    def anchor_end(self) -> int:
        return self.start + self.anchor_len


@dataclass(frozen=True)
class CSKReqMeta:
    plan: CSKLoadPlan
    block_ids: tuple[list[int], ...]


@dataclass(frozen=True)
class CSKProbeMeta:
    req_id: str
    cache_id: str
    start: int
    end: int
    source_offset: int
    block_ids: tuple[list[int], ...]
    tau: float
    gate_metric: str

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class CSKConnectorMetadata(KVConnectorMetadata):
    requests: list[CSKReqMeta] = field(default_factory=list)
    probes: list[CSKProbeMeta] = field(default_factory=list)


@dataclass
class CSKProbeWorkerMetadata(KVConnectorWorkerMetadata):
    decisions: list[CSKProbeDecision] = field(default_factory=list)

    def aggregate(
        self,
        other: "KVConnectorWorkerMetadata",
    ) -> "CSKProbeWorkerMetadata":
        if not isinstance(other, CSKProbeWorkerMetadata):
            return self
        return CSKProbeWorkerMetadata(decisions=self.decisions + other.decisions)


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
        self._directive_boundaries: set[str] = set()
        self._probe_states: dict[str, CSKProbeState] = {}
        self._probe_accumulators: dict[str, CSKProbeAccumulator] = {}
        self._probe_warned_no_accumulator: set[str] = set()
        self._probe_warned_worker_meta_type: set[str] = set()
        self._probe_skip_logs_remaining = 8
        self._probe_no_match_logs_remaining = 8
        self._probe_cap_logs_remaining = 20
        self._current_rope: object | None = None
        if self._config.kv_dir is not None:
            loaded = self._registry.load_dir(self._config.kv_dir)
            logger.warning("CSKCache loaded %d KV entries from %s", len(loaded), self._config.kv_dir)
            _trace("loaded %d KV entries from %s", len(loaded), self._config.kv_dir)
        self._catalog: SegmentCatalog = SegmentCatalog.from_entries(
            self._registry.entries()
        )
        logger.warning(
            "CSKCache connector initialized: role=%s catalog_segments=%d probe_enabled=%s",
            role,
            len(self._catalog.segments),
            self._config.probe_enabled,
        )
        _trace(
            "connector initialized role=%s catalog_segments=%d probe_enabled=%s",
            role,
            len(self._catalog.segments),
            self._config.probe_enabled,
        )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self._kv_caches = kv_caches

    def cap_num_new_tokens(
        self,
        request: "Request",
        base_num_computed_tokens: int,
        num_new_tokens: int,
    ) -> int:
        if num_new_tokens <= 0:
            return num_new_tokens

        if not self._config.probe_enabled:
            boundary = self._pending_boundaries.get(request.request_id)
            if (
                boundary is not None
                and request.request_id in self._directive_boundaries
            ):
                base = base_num_computed_tokens
                if base < boundary.start:
                    return min(num_new_tokens, boundary.start - base)
                if base < boundary.end:
                    return 0
                self._pending_boundaries.pop(request.request_id, None)
                self._directive_boundaries.discard(request.request_id)
            return num_new_tokens

        state = self._get_or_create_probe_state(request, base_num_computed_tokens)
        if state is None:
            return num_new_tokens

        base = base_num_computed_tokens
        if self._probe_cap_logs_remaining > 0:
            _trace(
                "cap request=%s base=%d num_new=%d phase=%s span=[%d,%d) probe_end=%d anchor_end=%d",
                request.request_id,
                base,
                num_new_tokens,
                state.phase.value,
                state.start,
                state.end,
                state.probe_end,
                state.anchor_end,
            )
            self._probe_cap_logs_remaining -= 1
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

    def get_inprocess_load_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> int:
        req_id = request.request_id
        boundary = self._pending_boundaries.get(req_id)
        if (
            boundary is not None
            and req_id in self._directive_boundaries
            and num_computed_tokens == boundary.start
        ):
            token_ids = self._request_token_ids(request)
            self._plans[req_id] = self._make_load_plan(
                req_id=req_id,
                occurrence=boundary,
                token_ids=token_ids,
            )
            self._pending_boundaries.pop(req_id, None)
            self._directive_boundaries.discard(req_id)
            _trace(
                "directive in-process load requested request=%s cache_id=%s target=[%d,%d) tokens=%d",
                req_id,
                boundary.cache_id,
                boundary.start,
                boundary.end,
                boundary.length,
            )
            return boundary.length

        if not self._config.probe_enabled:
            return 0
        state = self._probe_states.get(req_id)
        if state is None or state.phase != CSKProbePhase.NEED_LOAD:
            return 0
        load_start = state.load_start
        if load_start is None:
            return 0
        if num_computed_tokens != load_start:
            return 0
        length = state.end - load_start
        if length <= 0:
            state.phase = CSKProbePhase.DONE
            return 0

        token_ids = self._request_token_ids(request)
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
        logger.warning(
            "CSKCache in-process load requested request=%s cache_id=%s "
            "target=[%d,%d) source_offset=%d tokens=%d",
            req_id,
            state.cache_id,
            load_start,
            state.end,
            load_start - state.start,
            length,
        )
        _trace(
            "in-process load requested request=%s cache_id=%s target=[%d,%d) source_offset=%d tokens=%d",
            req_id,
            state.cache_id,
            load_start,
            state.end,
            load_start - state.start,
            length,
        )
        return length

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        if self._config.probe_enabled:
            return 0, False

        req_id = request.request_id
        token_ids = self._request_token_ids(request)
        occurrence, directive_seen = self._select_occurrence(
            request,
            token_ids,
            num_computed_tokens,
        )
        self._plans.pop(req_id, None)
        self._pending_boundaries.pop(req_id, None)
        self._directive_boundaries.discard(req_id)
        if occurrence is None:
            return 0, False
        if occurrence.end <= num_computed_tokens:
            return 0, False
        if occurrence.start > num_computed_tokens:
            self._pending_boundaries[req_id] = occurrence
            if directive_seen:
                self._directive_boundaries.add(req_id)
            logger.debug(
                "CSKCache occurrence for request %s starts at %d after computed=%d; "
                "waiting for scheduler boundary",
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
        self._plans[req_id] = self._make_load_plan(
            req_id=req_id,
            occurrence=occurrence,
            token_ids=token_ids,
        )
        return occurrence.length, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        req_id = request.request_id
        self._allocated_blocks[req_id] = blocks.get_block_ids(allow_none=True)
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

    def build_connector_meta(self, scheduler_output: "SchedulerOutput") -> CSKConnectorMetadata:
        meta = CSKConnectorMetadata()
        for req_id, num_scheduled_tokens in scheduler_output.num_scheduled_tokens.items():
            plan = self._plans.pop(req_id, None)
            blocks = self._allocated_blocks.pop(req_id, None)
            if plan is not None:
                if blocks is None:
                    raise RuntimeError(f"CSKCache load plan for {req_id} has no blocks")
                meta.requests.append(CSKReqMeta(plan=plan, block_ids=blocks))
                state = self._probe_states.get(req_id)
                if state is not None:
                    state.phase = CSKProbePhase.DONE
                continue

            state = self._probe_states.get(req_id)
            if state is None:
                continue
            if num_scheduled_tokens <= 0 or blocks is None:
                continue
            if state.pending_capture == "probe":
                meta.probes.append(
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
                logger.warning(
                    "CSKCache probe capture scheduled request=%s cache_id=%s "
                    "target=[%d,%d)",
                    req_id,
                    state.cache_id,
                    state.start,
                    state.probe_end,
                )
                _trace(
                    "probe capture scheduled request=%s cache_id=%s target=[%d,%d)",
                    req_id,
                    state.cache_id,
                    state.start,
                    state.probe_end,
                )
                state.pending_capture = None
                state.phase = CSKProbePhase.WAIT_PROBE
            elif state.pending_capture == "anchor":
                state.pending_capture = None
                state.phase = CSKProbePhase.NEED_LOAD
                state.load_start = state.anchor_end
                logger.warning(
                    "CSKCache anchor completed request=%s cache_id=%s "
                    "anchor_end=%d load_start=%d",
                    req_id,
                    state.cache_id,
                    state.anchor_end,
                    state.load_start,
                )
                _trace(
                    "anchor completed request=%s cache_id=%s anchor_end=%d load_start=%d",
                    req_id,
                    state.cache_id,
                    state.anchor_end,
                    state.load_start,
                )
        return meta

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        if not self._kv_caches:
            self._init_kv_caches_from_forward_context(forward_context)
        metadata = self._parent._get_connector_metadata()
        assert isinstance(metadata, CSKConnectorMetadata)

        self._current_rope = None
        model = getattr(forward_context, "model", None)
        if model is not None:
            self._current_rope = find_rotary_embedding(model)
        if not metadata.requests:
            return

        for request in metadata.requests:
            plan = request.plan
            entry = self._registry.get(plan.cache_id)
            if entry is None:
                raise RuntimeError(f"CSKCache cache_id={plan.cache_id} is not loaded")
            if plan.source_offset + plan.length > entry.length:
                raise RuntimeError(
                    f"CSKCache source slice mismatch for {plan.cache_id}: "
                    f"offset={plan.source_offset}, length={plan.length}, "
                    f"entry={entry.length}"
                )
            expected_tokens = tuple(
                int(value)
                for value in entry.token_ids[
                    plan.source_offset : plan.source_offset + plan.length
                ]
            )
            if expected_tokens != plan.token_ids:
                raise RuntimeError(f"CSKCache token mismatch for {plan.cache_id}")
            if not request.block_ids or request.block_ids[0] is None:
                raise RuntimeError(f"CSKCache load plan for {plan.req_id} has no blocks")
            for layer_name in entry.kv_by_layer:
                target_cache = self._kv_caches.get(layer_name)
                if target_cache is None:
                    continue
                key, value = prepare_reuse_slice(
                    entry,
                    layer_name=layer_name,
                    source_offset=plan.source_offset,
                    length=plan.length,
                    target_start=plan.start,
                    rope=self._current_rope,
                    device=target_cache.device,
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
                "CSKCache loaded cache_id=%s request=%s target=[%d,%d) "
                "source_offset=%d tokens=%d",
                plan.cache_id,
                plan.req_id,
                plan.start,
                plan.end,
                plan.source_offset,
                plan.length,
            )

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
        if not metadata.probes:
            return

        for probe in metadata.probes:
            entry = self._registry.get(probe.cache_id)
            if entry is None or layer_name not in entry.kv_by_layer:
                if self._probe_skip_logs_remaining > 0:
                    sample_layers = []
                    if entry is not None:
                        sample_layers = list(entry.kv_by_layer.keys())[:3]
                    logger.warning(
                        "CSKCache probe skip layer request=%s cache_id=%s "
                        "layer_name=%s entry_found=%s sample_entry_layers=%s",
                        probe.req_id,
                        probe.cache_id,
                        layer_name,
                        entry is not None,
                        sample_layers,
                    )
                    _trace(
                        "probe skip layer request=%s cache_id=%s layer_name=%s entry_found=%s sample_entry_layers=%s",
                        probe.req_id,
                        probe.cache_id,
                        layer_name,
                        entry is not None,
                        sample_layers,
                    )
                    self._probe_skip_logs_remaining -= 1
                continue
            if not probe.block_ids or probe.block_ids[0] is None:
                logger.warning(
                    "CSKCache probe skip layer request=%s cache_id=%s "
                    "layer_name=%s reason=no_block_ids",
                    probe.req_id,
                    probe.cache_id,
                    layer_name,
                )
                _trace(
                    "probe skip layer request=%s cache_id=%s layer_name=%s reason=no_block_ids",
                    probe.req_id,
                    probe.cache_id,
                    layer_name,
                )
                continue
            reuse_key, reuse_value = prepare_reuse_slice(
                entry,
                layer_name=layer_name,
                source_offset=probe.source_offset,
                length=probe.length,
                target_start=probe.start,
                rope=self._current_rope,
                device=kv_layer.device,
            )
            recompute_key, recompute_value = gather_span(
                kv_layer,
                probe.block_ids[0],
                probe.start,
                probe.end,
                self._block_size,
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
                logger.warning(
                    "CSKCache probe accumulator created request=%s cache_id=%s "
                    "first_layer=%s",
                    probe.req_id,
                    probe.cache_id,
                    layer_name,
                )
                _trace(
                    "probe accumulator created request=%s cache_id=%s first_layer=%s",
                    probe.req_id,
                    probe.cache_id,
                    layer_name,
                )
            accumulator.add_layer(
                layer_name,
                reuse_key=reuse_key,
                reuse_value=reuse_value,
                recompute_key=recompute_key,
                recompute_value=recompute_value,
            )

    def wait_for_save(self) -> None:
        return

    def build_connector_worker_meta(self) -> CSKProbeWorkerMetadata | None:
        if not self._probe_accumulators:
            metadata = self._parent._get_connector_metadata()
            if isinstance(metadata, CSKConnectorMetadata):
                for probe in metadata.probes:
                    if probe.req_id not in self._probe_warned_no_accumulator:
                        logger.warning(
                            "CSKCache probe metadata produced no accumulator "
                            "request=%s cache_id=%s target=[%d,%d)",
                            probe.req_id,
                            probe.cache_id,
                            probe.start,
                            probe.end,
                        )
                        _trace(
                            "probe metadata produced no accumulator request=%s cache_id=%s target=[%d,%d)",
                            probe.req_id,
                            probe.cache_id,
                            probe.start,
                            probe.end,
                        )
                        self._probe_warned_no_accumulator.add(probe.req_id)
            return None
        decisions: list[CSKProbeDecision] = []
        for req_id, accumulator in list(self._probe_accumulators.items()):
            try:
                decisions.append(accumulator.decide())
            except RuntimeError as exc:
                logger.warning("CSKCache probe decision skipped for %s: %s", req_id, exc)
            finally:
                self._probe_accumulators.pop(req_id, None)
        if decisions:
            logger.warning("CSKCache built %d probe worker decision(s)", len(decisions))
            _trace("built %d probe worker decision(s)", len(decisions))
        return CSKProbeWorkerMetadata(decisions=decisions) if decisions else None

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        worker_meta = connector_output.kv_connector_worker_meta
        if not isinstance(worker_meta, CSKProbeWorkerMetadata):
            for req_id, state in self._probe_states.items():
                if state.phase == CSKProbePhase.WAIT_PROBE and req_id not in self._probe_warned_worker_meta_type:
                    logger.warning(
                        "CSKCache waiting for probe decision but worker meta is %s "
                        "request=%s cache_id=%s",
                        type(worker_meta).__name__,
                        req_id,
                        state.cache_id,
                    )
                    _trace(
                        "waiting for probe decision but worker meta is %s request=%s cache_id=%s",
                        type(worker_meta).__name__,
                        req_id,
                        state.cache_id,
                    )
                    self._probe_warned_worker_meta_type.add(req_id)
            return
        for decision in worker_meta.decisions:
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
            _trace(
                "probe decision request=%s cache_id=%s passed=%s gate=%.6f tau=%.6f metric=%s layers=%d",
                decision.req_id,
                decision.cache_id,
                decision.passed,
                decision.metrics.gate_value,
                decision.tau,
                decision.metrics.gate_metric,
                decision.metrics.num_layers,
            )

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        for req_id in finished_req_ids:
            self._probe_states.pop(req_id, None)
            self._plans.pop(req_id, None)
            self._allocated_blocks.pop(req_id, None)
            self._probe_accumulators.pop(req_id, None)
            self._pending_boundaries.pop(req_id, None)
            self._directive_boundaries.discard(req_id)
        return None, None

    def shutdown(self) -> None:
        return None

    def _init_kv_caches_from_forward_context(self, forward_context: "ForwardContext") -> None:
        for layer_name, layer in getattr(forward_context, "no_compile_layers", {}).items():
            kv_cache = getattr(layer, "kv_cache", None)
            if kv_cache is not None:
                self._kv_caches[layer_name] = kv_cache[0]

    @staticmethod
    def _request_token_ids(request: "Request") -> list[int]:
        return list(getattr(request, "all_token_ids", None) or request.prompt_token_ids or [])

    def _select_occurrence(
        self,
        request: "Request",
        token_ids: list[int],
        num_computed_tokens: int,
    ) -> tuple[SegmentOccurrence | None, bool]:
        directive = self._parse_request_directive(request)
        if directive is not None:
            if not directive.enabled:
                return None, True
            return self._occurrence_from_directive(request, token_ids, directive), True
        return (
            find_best_occurrence(self._catalog, token_ids, num_computed_tokens),
            False,
        )

    def _parse_request_directive(
        self,
        request: "Request",
    ) -> CSKCacheRequestDirective | None:
        params = getattr(request, "kv_transfer_params", None)
        if not isinstance(params, Mapping):
            return None
        raw = params.get("cskcache")
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise ValueError("CSKCache directive must be a mapping")

        enabled_raw = raw.get("enabled", True)
        if not isinstance(enabled_raw, bool):
            raise ValueError("CSKCache directive enabled must be a bool")
        if not enabled_raw:
            return CSKCacheRequestDirective(enabled=False, cache_id="")

        cache_id = raw.get("cache_id")
        if not isinstance(cache_id, str) or not cache_id:
            raise ValueError("CSKCache directive requires a non-empty cache_id")

        placement_raw = raw.get(
            "placement",
            CSKCacheDirectivePlacement.EXPLICIT_SPAN.value,
        )
        try:
            placement = CSKCacheDirectivePlacement(str(placement_raw))
        except ValueError as exc:
            raise ValueError(
                f"Unsupported CSKCache directive placement: {placement_raw!r}"
            ) from exc

        trailing_token_count = self._parse_nonnegative_int(
            raw.get("trailing_token_count", 0),
            "trailing_token_count",
        )
        return CSKCacheRequestDirective(
            enabled=True,
            cache_id=cache_id,
            placement=placement,
            target_start=self._parse_optional_int(raw.get("target_start"), "target_start"),
            target_end=self._parse_optional_int(raw.get("target_end"), "target_end"),
            trailing_token_count=trailing_token_count,
        )

    @staticmethod
    def _parse_optional_int(value: object, name: str) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"CSKCache directive {name} must be an int")
        return value

    @staticmethod
    def _parse_nonnegative_int(value: object, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"CSKCache directive {name} must be an int")
        if value < 0:
            raise ValueError(f"CSKCache directive {name} must be non-negative")
        return value

    def _occurrence_from_directive(
        self,
        request: "Request",
        token_ids: list[int],
        directive: CSKCacheRequestDirective,
    ) -> SegmentOccurrence:
        entry = self._registry.get(directive.cache_id)
        if entry is None:
            raise RuntimeError(
                f"CSKCache directive cache_id={directive.cache_id} is not loaded"
            )

        if directive.placement == CSKCacheDirectivePlacement.EXPLICIT_SPAN:
            if directive.target_start is None or directive.target_end is None:
                raise ValueError(
                    "CSKCache explicit_span directive requires target_start and target_end"
                )
            target_start = directive.target_start
            target_end = directive.target_end
        elif directive.placement == CSKCacheDirectivePlacement.SUFFIX_BEFORE_TRAILING:
            target_end = len(token_ids) - directive.trailing_token_count
            target_start = target_end - entry.length
        else:
            raise ValueError(f"Unsupported CSKCache placement: {directive.placement}")

        if target_start < 0 or target_end > len(token_ids) or target_start >= target_end:
            raise RuntimeError(
                f"CSKCache directive span out of bounds for {directive.cache_id}: "
                f"target=[{target_start},{target_end}), token_count={len(token_ids)}"
            )
        if target_end - target_start != entry.length:
            raise RuntimeError(
                f"CSKCache directive span length mismatch for {directive.cache_id}: "
                f"target_length={target_end - target_start}, entry_length={entry.length}"
            )

        target_token_ids = tuple(token_ids[target_start:target_end])
        entry_token_ids = tuple(int(value) for value in entry.token_ids)
        if target_token_ids != entry_token_ids:
            raise RuntimeError(
                f"CSKCache directive token mismatch for {directive.cache_id}: "
                f"target=[{target_start},{target_end})"
            )

        _trace(
            "directive resolved request=%s cache_id=%s placement=%s target=[%d,%d) trailing=%d",
            request.request_id,
            directive.cache_id,
            directive.placement.value,
            target_start,
            target_end,
            directive.trailing_token_count,
        )
        return SegmentOccurrence(
            cache_id=directive.cache_id,
            start=target_start,
            end=target_end,
            mode=CSKCacheMode.REUSE,
        )

    @staticmethod
    def _make_load_plan(
        req_id: str,
        occurrence: SegmentOccurrence,
        token_ids: list[int],
        source_offset: int = 0,
    ) -> CSKLoadPlan:
        return CSKLoadPlan(
            req_id=req_id,
            cache_id=occurrence.cache_id,
            mode=occurrence.mode,
            start=occurrence.start,
            end=occurrence.end,
            token_ids=tuple(token_ids[occurrence.start : occurrence.end]),
            source_offset=source_offset,
        )

    def _get_or_create_probe_state(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> CSKProbeState | None:
        req_id = request.request_id
        state = self._probe_states.get(req_id)
        if state is not None and state.phase != CSKProbePhase.DONE:
            return state

        token_ids = self._request_token_ids(request)
        occurrence, _ = self._select_occurrence(
            request,
            token_ids,
            num_computed_tokens,
        )
        if occurrence is None:
            if self._probe_no_match_logs_remaining > 0:
                _trace(
                    "no probe occurrence request=%s computed=%d token_count=%d catalog_segments=%d",
                    req_id,
                    num_computed_tokens,
                    len(token_ids),
                    len(self._catalog.segments),
                )
                self._probe_no_match_logs_remaining -= 1
            return None
        if occurrence.end <= num_computed_tokens:
            return None
        if occurrence.start < num_computed_tokens:
            return None
        entry = self._registry.get(occurrence.cache_id)
        if entry is None:
            logger.warning("CSKCache cache_id=%s matched but no KV entry is loaded", occurrence.cache_id)
            return None
        target_token_ids = tuple(token_ids[occurrence.start : occurrence.end])
        if tuple(entry.token_ids) != target_token_ids:
            logger.warning("CSKCache token mismatch for cache_id=%s; skip probe", occurrence.cache_id)
            return None

        length = occurrence.length
        state = CSKProbeState(
            req_id=req_id,
            cache_id=occurrence.cache_id,
            start=occurrence.start,
            end=occurrence.end,
            probe_len=min(self._config.probe_tokens, length),
            anchor_len=min(self._config.anchor_tokens, length),
            tau=self._config.probe_tau,
            gate_metric=self._config.gate_metric,
        )
        self._probe_states[req_id] = state
        logger.warning(
            "CSKCache probe state request=%s cache_id=%s target=[%d,%d) "
            "probe_len=%d anchor_len=%d",
            req_id,
            occurrence.cache_id,
            occurrence.start,
            occurrence.end,
            state.probe_len,
            state.anchor_len,
        )
        _trace(
            "probe state request=%s cache_id=%s target=[%d,%d) probe_len=%d anchor_len=%d token_count=%d",
            req_id,
            occurrence.cache_id,
            occurrence.start,
            occurrence.end,
            state.probe_len,
            state.anchor_len,
            len(token_ids),
        )
        return state
