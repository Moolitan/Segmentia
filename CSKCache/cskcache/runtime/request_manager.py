"""T0 Skill selection and request-independent reuse lifecycle coordination."""

from __future__ import annotations

import threading
import time
from typing import Sequence

from ..metadata.skill_format import parse_skill_payload
from ..metadata.manager import MetadataManager
from .authentication import locate_authenticated_skill_prefix
from .base import (
    BindingState,
    CorrectionStrategy,
    HostLoadState,
    ReusePlan,
    ReusePolicy,
    ReuseReadiness,
    ReuseReadinessResult,
    RuntimeReuseState,
    VerifiedRequestBinding,
)
from ..storage.manager import StorageManager


class RequestManager:
    """Own Skill selection, Tool-result checks, and request authentication.

    ``ticket`` is the structured tool-call ID.  The manager does not depend on
    HTTP or vLLM message classes: callers pass the three standard Tool-result
    fields and vLLM's final token IDs.  GPU scheduling remains a later stage.
    """

    def __init__(
        self,
        metadata_manager: MetadataManager,
        storage_manager: StorageManager,
        *,
        model_fingerprint: str,
        tokenizer_fingerprint: str,
        ticket_ttl_seconds: float = 60.0,
    ) -> None:
        if not model_fingerprint or not tokenizer_fingerprint:
            raise ValueError("deployment fingerprints must be non-empty")
        if ticket_ttl_seconds <= 0:
            raise ValueError("ticket_ttl_seconds must be > 0")
        self._metadata_manager = metadata_manager
        self._storage_manager = storage_manager
        self._model_fingerprint = model_fingerprint
        self._tokenizer_fingerprint = tokenizer_fingerprint
        self._ticket_ttl_ns = int(ticket_ttl_seconds * 1_000_000_000)
        self._lock = threading.RLock()
        self._closed = False

    def select_skill(self, ticket: str, skill_name: str) -> bool:
        """Accept one structured SkillAction without waiting for storage I/O.

        The RPC is at-least-once safe: repeating the same ticket and object is
        accepted without creating another lease or physical read.  Reusing a
        ticket for another Skill fails closed.
        """

        ticket = ticket.strip()
        skill_name = skill_name.strip()
        if not ticket or not skill_name:
            return False
        with self._lock:
            if self._closed:
                return False
            self._expire_locked()
            try:
                cache_object = self._metadata_manager.resolve_object(
                    skill_name=skill_name,
                    model_fingerprint=self._model_fingerprint,
                    tokenizer_fingerprint=self._tokenizer_fingerprint,
                )
            except (KeyError, ValueError):
                return False

            try:
                existing = self._metadata_manager.get_runtime(ticket)
            except KeyError:
                existing = None
            if existing is not None:
                return self._duplicate_is_live(existing, cache_object.object_id)

            now_ns = time.time_ns()
            self._metadata_manager.create_ticket(
                ticket,
                cache_object.object_id,
                now_ns=now_ns,
                deadline_ns=now_ns + self._ticket_ttl_ns,
            )
            try:
                self._storage_manager.submit_host_load(ticket, cache_object.object_id)
            except Exception as exc:
                try:
                    self._metadata_manager.mark_host_failed(
                        ticket, f"host load submission failed: {type(exc).__name__}"
                    )
                except ValueError:
                    pass
                return False
            return True

    def poll(self, ticket: str) -> RuntimeReuseState:
        """Return the current ticket state after applying deadline cleanup."""

        with self._lock:
            self._expire_locked()
            return self._metadata_manager.get_runtime(ticket)

    def inspect_tool_observation(
        self,
        ticket: str,
        tool_name: str,
        content: str,
    ) -> bool:
        """Check request B's Tool result while vLLM tokenizes in parallel.

        This is intentionally only an early eligibility check.  It proves
        that OpenHands returned the expected successful Skill result and that
        the T0 transaction is still live; it does not authenticate KV tokens.
        """

        ticket = ticket.strip()
        tool_name = tool_name.strip()
        if not ticket:
            return False
        with self._lock:
            if self._closed:
                return False
            self._expire_locked()
            try:
                state = self._metadata_manager.get_runtime(ticket)
            except KeyError:
                return False
            if state.binding_state is BindingState.OBSERVED:
                return self._observation_matches_object(state, tool_name, content)
            if state.binding_state is not BindingState.UNBOUND:
                return False
            if not self._observation_matches_object(state, tool_name, content):
                self.cancel(ticket, "invalid_skill_observation")
                return False
            try:
                self._metadata_manager.mark_observation_verified(ticket)
            except ValueError:
                return False
            return True

    def authenticate_and_bind(
        self,
        ticket: str,
        request_id: str,
        prompt_token_ids: Sequence[int],
    ) -> VerifiedRequestBinding | None:
        """Authenticate the newest Skill's longest unchanged chunk prefix.

        The final prompt tokens come from vLLM's normal tokenizer.  CSKCache
        searches with its persistent marker metadata and authenticates the
        newest occurrence.  Exact identity remains the fast path; otherwise
        independent logical chunks are checked in order until the first
        difference.
        """

        ticket = ticket.strip()
        request_id = request_id.strip()
        if not ticket or not request_id:
            return None
        with self._lock:
            if self._closed:
                return None
            self._expire_locked()
            try:
                state = self._metadata_manager.get_runtime(ticket)
            except KeyError:
                return None
            if state.binding_state is not BindingState.OBSERVED:
                return None
            cache_object = self._metadata_manager.get_object(state.cache_object_id)
            authenticated = locate_authenticated_skill_prefix(
                prompt_token_ids, cache_object
            )
            if authenticated is None:
                self.cancel(ticket, "skill_token_authentication_failed")
                return None
            start = authenticated.segment_start
            end = authenticated.segment_end
            try:
                bound = self._metadata_manager.bind_request(
                    ticket,
                    request_id=request_id,
                    verified_cache_object_id=cache_object.object_id,
                    segment_start=start,
                    segment_end=end,
                    match_mode=authenticated.match_mode,
                    matched_chunk_count=authenticated.matched_chunk_count,
                )
            except ValueError:
                self.cancel(ticket, "request_binding_failed")
                return None
            return VerifiedRequestBinding(
                ticket=ticket,
                cache_object_id=bound.cache_object_id,
                request_id=request_id,
                segment_start=start,
                segment_end=end,
                match_mode=authenticated.match_mode,
                matched_chunk_count=authenticated.matched_chunk_count,
            )

    def cancel(self, ticket: str, reason: str = "request_cancelled") -> None:
        """Cancel one ticket-owned Host load while preserving audit state."""

        with self._lock:
            try:
                self._storage_manager.cancel_host_load(ticket, reason)
            except KeyError:
                try:
                    state = self._metadata_manager.get_runtime(ticket)
                except KeyError:
                    return
                if state.binding_state not in (
                    BindingState.FALLBACK,
                    BindingState.RELEASED,
                ):
                    self._metadata_manager.fallback(ticket, reason)

    def prepare_reuse(
        self,
        ticket: str,
        request_id: str,
        *,
        block_alignment: int,
        policy: ReusePolicy | None = None,
    ) -> ReusePlan | None:
        """Create the aligned online-recompute and reuse ranges.

        This method only plans token ranges.  It neither waits for SSD I/O nor
        moves KV to the GPU.  A range that cannot retain the minimum reusable
        suffix fails closed and releases the speculative host load.
        """

        selected_policy = policy or ReusePolicy()
        if block_alignment <= 0:
            return None
        with self._lock:
            if self._closed:
                return None
            self._expire_locked()
            try:
                state = self._metadata_manager.get_runtime(ticket)
            except KeyError:
                return None
            if (
                state.binding_state is not BindingState.VERIFIED
                or state.request_id != request_id
                or state.segment_start is None
                or state.segment_end is None
            ):
                return None
            cache_object = self._metadata_manager.get_object(state.cache_object_id)
            matched_tokens = state.segment_end - state.segment_start
            if matched_tokens > cache_object.token_count:
                self.cancel(ticket, "verified_prefix_exceeds_source_object")
                return None

            calibration_tokens = selected_policy.resolve_calibration_tokens(
                matched_tokens
            )
            nominal_start = (
                state.segment_start
                + selected_policy.minimum_full_recompute_tokens
                + calibration_tokens
            )
            reuse_start = _round_up(nominal_start, block_alignment)
            reuse_end = _round_down(state.segment_end, block_alignment)
            if reuse_end - reuse_start < selected_policy.minimum_reuse_tokens:
                self.cancel(ticket, "reusable_suffix_too_short")
                return None

            relative_start = reuse_start - state.segment_start
            relative_end = reuse_end - state.segment_start
            source_reuse_start = (
                cache_object.source_position_start + relative_start
            )
            source_reuse_end = cache_object.source_position_start + relative_end
            calibration_end = reuse_start
            calibration_start = reuse_start - calibration_tokens
            plan = ReusePlan(
                ticket=ticket,
                cache_object_id=state.cache_object_id,
                request_id=request_id,
                segment_start=state.segment_start,
                segment_end=state.segment_end,
                reuse_start=reuse_start,
                reuse_end=reuse_end,
                source_reuse_start=source_reuse_start,
                source_reuse_end=source_reuse_end,
                calibration_start=calibration_start,
                calibration_end=calibration_end,
                correction_alpha=selected_policy.correction_alpha,
                block_alignment=block_alignment,
                source_token_count=cache_object.token_count,
                correction_strategy=CorrectionStrategy(
                    selected_policy.correction_strategy
                ),
                deviation_recompute_ratio=(
                    selected_policy.deviation_recompute_ratio
                ),
                deviation_check_layer=selected_policy.deviation_check_layer,
            )
            if state.reuse_start is not None:
                existing = self._plan_from_state(state)
                return existing if existing == plan else None
            try:
                self._metadata_manager.set_reuse_plan(ticket, plan)
            except ValueError:
                self.cancel(ticket, "reuse_plan_registration_failed")
                return None
            return plan

    def query_reuse_readiness(
        self, ticket: str, request_id: str
    ) -> ReuseReadinessResult:
        """Return host readiness without activating GPU reuse."""

        with self._lock:
            if self._closed:
                return ReuseReadinessResult(
                    ReuseReadiness.FALLBACK, reason="request_manager_closed"
                )
            self._expire_locked()
            try:
                self._storage_manager.poll_host_load(ticket)
                state = self._metadata_manager.get_runtime(ticket)
            except KeyError:
                return ReuseReadinessResult(
                    ReuseReadiness.FALLBACK, reason="unknown_ticket"
                )
            if state.request_id != request_id:
                return ReuseReadinessResult(
                    ReuseReadiness.FALLBACK, reason="request_binding_mismatch"
                )
            if state.binding_state in (
                BindingState.FALLBACK,
                BindingState.RELEASED,
            ) or state.host_load_state is HostLoadState.FAILED:
                return ReuseReadinessResult(
                    ReuseReadiness.FALLBACK,
                    reason=state.fallback_reason or state.binding_state.value,
                )
            if state.reuse_start is None:
                return ReuseReadinessResult(
                    ReuseReadiness.FALLBACK, reason="reuse_plan_missing"
                )
            plan = self._plan_from_state(state)
            if state.host_load_state is HostLoadState.READY:
                return ReuseReadinessResult(ReuseReadiness.READY, plan=plan)
            return ReuseReadinessResult(ReuseReadiness.LOADING, plan=plan)

    def activate_reuse(self, ticket: str, request_id: str) -> ReusePlan | None:
        """Bind one ready host-buffer group to its authenticated request.

        Activation is deliberately separate from readiness polling.  The
        scheduler calls it only after it has successfully reserved the GPU KV
        blocks for ``[reuse_start, reuse_end)``.  Repeating the same activation
        is idempotent; another request cannot take over the ticket.
        """

        with self._lock:
            if self._closed:
                return None
            self._expire_locked()
            try:
                self._storage_manager.poll_host_load(ticket)
                state = self._metadata_manager.get_runtime(ticket)
            except KeyError:
                return None
            if state.request_id != request_id or state.reuse_start is None:
                return None
            if state.binding_state is BindingState.ACTIVE:
                return self._plan_from_state(state)
            if (
                state.binding_state is not BindingState.VERIFIED
                or state.host_load_state is not HostLoadState.READY
            ):
                return None
            try:
                active = self._metadata_manager.activate(ticket)
            except ValueError:
                return None
            return self._plan_from_state(active)

    def get_active_layer_buffers(
        self, ticket: str, request_id: str
    ) -> tuple[object, ...]:
        """Return the local rank's complete, read-only pinned layer group.

        This is a process-local worker API.  ``MemoryObj`` instances must not
        cross the scheduler RPC boundary; every LMCache worker resolves the
        same ticket against the buffers that its own T0 handler loaded.
        """

        with self._lock:
            if self._closed:
                raise RuntimeError("request manager is closed")
            state = self._metadata_manager.get_runtime(ticket)
            if state.request_id != request_id:
                raise ValueError("request does not own the active ticket")
            if state.binding_state is not BindingState.ACTIVE:
                raise RuntimeError("reuse ticket is not active")
            buffers = self._storage_manager.get_ready_buffers(ticket)
            if len(buffers) != self._metadata_manager.expected_layers:
                raise RuntimeError("active host buffer group is incomplete")
            return tuple(buffers)

    def mark_layer_loaded(
        self, ticket: str, request_id: str, layer_id: int
    ) -> RuntimeReuseState:
        """Record one in-order host-to-GPU layer transfer."""

        with self._lock:
            self._require_active_request(ticket, request_id)
            return self._metadata_manager.mark_layer_loaded(ticket, layer_id)

    def mark_layer_corrected(
        self, ticket: str, request_id: str, layer_id: int
    ) -> RuntimeReuseState:
        """Record one in-order K-correction completion."""

        with self._lock:
            self._require_active_request(ticket, request_id)
            return self._metadata_manager.mark_layer_corrected(ticket, layer_id)

    def release(self, ticket: str) -> RuntimeReuseState:
        """End one ticket and return its Host buffers."""

        with self._lock:
            return self._storage_manager.release_host_load(ticket)

    def close(self) -> None:
        """Stop new selections and release all storage resources."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._storage_manager.close()

    def _expire_locked(self) -> None:
        for state in self._metadata_manager.expire():
            try:
                self._storage_manager.cancel_host_load(
                    state.ticket, "deadline_expired"
                )
            except (KeyError, ValueError):
                pass

    def _require_active_request(
        self, ticket: str, request_id: str
    ) -> RuntimeReuseState:
        state = self._metadata_manager.get_runtime(ticket)
        if state.request_id != request_id:
            raise ValueError("request does not own the active ticket")
        if state.binding_state is not BindingState.ACTIVE:
            raise RuntimeError("reuse ticket is not active")
        return state

    def _observation_matches_object(
        self,
        state: RuntimeReuseState,
        tool_name: str,
        content: str,
    ) -> bool:
        if tool_name != "skill" or not isinstance(content, str):
            return False
        payload = parse_skill_payload(content)
        if payload is None:
            return False
        cache_object = self._metadata_manager.get_object(state.cache_object_id)
        return payload.skill_name == cache_object.skill_name

    @staticmethod
    def _duplicate_is_live(state: RuntimeReuseState, object_id: str) -> bool:
        return (
            state.cache_object_id == object_id
            and state.host_load_state not in (
                HostLoadState.FAILED,
            )
            and state.binding_state not in (
                BindingState.FALLBACK,
                BindingState.RELEASED,
            )
        )

    @staticmethod
    def _plan_from_state(state: RuntimeReuseState) -> ReusePlan:
        required = (
            state.request_id,
            state.segment_start,
            state.segment_end,
            state.source_token_count,
            state.reuse_start,
            state.reuse_end,
            state.source_reuse_start,
            state.source_reuse_end,
            state.calibration_start,
            state.calibration_end,
            state.correction_alpha,
            state.correction_strategy,
            state.deviation_recompute_ratio,
            state.deviation_check_layer,
            state.block_alignment,
        )
        if any(value is None for value in required):
            raise ValueError("runtime reuse plan is incomplete")
        return ReusePlan(
            ticket=state.ticket,
            cache_object_id=state.cache_object_id,
            request_id=str(state.request_id),
            segment_start=int(state.segment_start),
            segment_end=int(state.segment_end),
            reuse_start=int(state.reuse_start),
            reuse_end=int(state.reuse_end),
            source_reuse_start=int(state.source_reuse_start),
            source_reuse_end=int(state.source_reuse_end),
            calibration_start=int(state.calibration_start),
            calibration_end=int(state.calibration_end),
            correction_alpha=float(state.correction_alpha),
            block_alignment=int(state.block_alignment),
            source_token_count=int(state.source_token_count),
            correction_strategy=CorrectionStrategy(state.correction_strategy),
            deviation_recompute_ratio=float(state.deviation_recompute_ratio),
            deviation_check_layer=int(state.deviation_check_layer),
        )

def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _round_down(value: int, alignment: int) -> int:
    return (value // alignment) * alignment
