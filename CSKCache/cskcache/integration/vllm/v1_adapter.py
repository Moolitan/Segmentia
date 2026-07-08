from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
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
from cskcache.v1.metadata import CSKCacheMode, CSKLoadPlan, SegmentOccurrence
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
        self._probe_states: dict[str, CSKProbeState] = {}
        self._probe_accumulators: dict[str, CSKProbeAccumulator] = {}
        self._current_rope: object | None = None
        if self._config.kv_dir is not None:
            loaded = self._registry.load_dir(self._config.kv_dir)
            logger.info("CSKCache loaded %d KV entries from %s", len(loaded), self._config.kv_dir)
        self._catalog: SegmentCatalog = SegmentCatalog.from_entries(
            self._registry.entries()
        )
        logger.info(
            "CSKCache connector initialized: role=%s catalog_segments=%d probe_enabled=%s",
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
        if not self._config.probe_enabled or num_new_tokens <= 0:
            return num_new_tokens
        state = self._get_or_create_probe_state(request, base_num_computed_tokens)
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

    def get_inprocess_load_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> int:
        if not self._config.probe_enabled:
            return 0
        state = self._probe_states.get(request.request_id)
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

        token_ids = list(getattr(request, "all_token_ids", None) or request.prompt_token_ids or [])
        target_token_ids = tuple(token_ids[load_start : state.end])
        self._plans[request.request_id] = CSKLoadPlan(
            req_id=request.request_id,
            cache_id=state.cache_id,
            mode=CSKCacheMode.REUSE,
            start=load_start,
            end=state.end,
            token_ids=target_token_ids,
            source_offset=load_start - state.start,
        )
        return length

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        if self._config.probe_enabled:
            return 0, False

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
                state.pending_capture = None
                state.phase = CSKProbePhase.WAIT_PROBE
            elif state.pending_capture == "anchor":
                state.pending_capture = None
                state.phase = CSKProbePhase.NEED_LOAD
                state.load_start = state.anchor_end
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
                continue
            if not probe.block_ids or probe.block_ids[0] is None:
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
            return None
        decisions: list[CSKProbeDecision] = []
        for req_id, accumulator in list(self._probe_accumulators.items()):
            try:
                decisions.append(accumulator.decide())
            except RuntimeError as exc:
                logger.warning("CSKCache probe decision skipped for %s: %s", req_id, exc)
            finally:
                self._probe_accumulators.pop(req_id, None)
        return CSKProbeWorkerMetadata(decisions=decisions) if decisions else None

    def update_connector_output(self, connector_output: KVConnectorOutput) -> None:
        worker_meta = connector_output.kv_connector_worker_meta
        if not isinstance(worker_meta, CSKProbeWorkerMetadata):
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

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str] | None, set[str] | None]:
        for req_id in finished_req_ids:
            self._probe_states.pop(req_id, None)
            self._plans.pop(req_id, None)
            self._allocated_blocks.pop(req_id, None)
            self._probe_accumulators.pop(req_id, None)
        return None, None

    def shutdown(self) -> None:
        return None

    def _init_kv_caches_from_forward_context(self, forward_context: "ForwardContext") -> None:
        for layer_name, layer in getattr(forward_context, "no_compile_layers", {}).items():
            kv_cache = getattr(layer, "kv_cache", None)
            if kv_cache is not None:
                self._kv_caches[layer_name] = kv_cache[0]

    def _get_or_create_probe_state(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> CSKProbeState | None:
        req_id = request.request_id
        state = self._probe_states.get(req_id)
        if state is not None and state.phase != CSKProbePhase.DONE:
            return state

        token_ids = list(getattr(request, "all_token_ids", None) or request.prompt_token_ids or [])
        occurrence = find_best_occurrence(
            self._catalog,
            token_ids,
            num_computed_tokens,
        )
        if occurrence is None or occurrence.end <= num_computed_tokens:
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
        return state
