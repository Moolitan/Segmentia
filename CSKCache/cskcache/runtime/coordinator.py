"""CSKCache-owned scheduler lifecycle and transition decisions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from ..profile import profile_event
from .base import (
    ActivationDirective,
    FailedReuseRange,
    LeaseOwner,
    ReusePlan,
    SchedulerControlPort,
    SchedulerReusePhase,
    SchedulerReuseState,
)


class SchedulerReuseCoordinator:
    """Own every request-local CSKCache scheduler transition.

    Serving runtimes provide request snapshots and execute the returned
    directives.  They do not interpret ReusePlan ranges or mutate CSKCache
    phases and lease ownership directly.
    """

    def __init__(self) -> None:
        self._states: dict[str, SchedulerReuseState] = {}

    def register(
        self,
        verified: object,
        *,
        request_id: str,
        prompt_tokens: int,
        block_alignment: int,
        async_scheduling: bool,
        control: SchedulerControlPort | None,
    ) -> bool:
        if not isinstance(verified, Mapping) or control is None:
            return False
        ticket = verified.get("ticket")
        if not isinstance(ticket, str) or verified.get("request_id") != request_id:
            return False
        if async_scheduling:
            control.cancel_csk_prefetch(ticket, "unsupported_scheduler_mode")
            return False
        plan = control.prepare_csk_reuse(ticket, request_id, block_alignment)
        if not self._matches_verified_request(
            plan,
            verified,
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            block_alignment=block_alignment,
        ):
            control.cancel_csk_prefetch(ticket, "invalid_reuse_plan")
            return False
        assert plan is not None
        self._states[request_id] = SchedulerReuseState(ticket=ticket, plan=plan)
        profile_event(
            "csk_reuse_registered",
            request_id,
            ticket=ticket,
            reuse_start=plan.reuse_start,
            reuse_end=plan.reuse_end,
        )
        return True

    def limit_prefill(
        self,
        request_id: str,
        *,
        num_computed_tokens: int,
        num_new_tokens: int,
        control: SchedulerControlPort | None,
    ) -> int:
        state = self._states.get(request_id)
        if state is None or state.phase is not SchedulerReusePhase.INITIAL:
            return num_new_tokens
        boundary = state.plan.calibration_start
        if num_computed_tokens >= boundary:
            state.phase = SchedulerReusePhase.FALLBACK
            self._release_scheduler_lease(state, control)
            return num_new_tokens
        return min(num_new_tokens, boundary - num_computed_tokens)

    def reaches_calibration_boundary(
        self,
        request_id: str,
        *,
        num_computed_tokens: int,
        stopped: bool,
        produced_tokens: bool,
        request_running: bool,
        in_flight_tokens: int,
    ) -> bool:
        state = self._states.get(request_id)
        if not (
            state is not None
            and state.phase is SchedulerReusePhase.INITIAL
            and not stopped
            and not produced_tokens
            and request_running
            and num_computed_tokens == state.plan.calibration_start
            and in_flight_tokens == 0
        ):
            return False
        state.phase = SchedulerReusePhase.WAITING
        profile_event(
            "csk_reuse_boundary_ready",
            request_id,
            ticket=state.ticket,
            calibration_start=state.plan.calibration_start,
        )
        return True

    def is_waiting(self, request_id: str) -> bool:
        state = self._states.get(request_id)
        return state is not None and state.phase is SchedulerReusePhase.WAITING

    def poll(self, request_id: str, control: SchedulerControlPort) -> bool:
        state = self._states[request_id]
        readiness = control.query_csk_readiness(state.ticket, request_id)
        if readiness.get("status") == "loading":
            return False
        state.readiness = readiness
        return True

    def activate(
        self,
        request_id: str,
        *,
        num_computed_tokens: int,
        control: SchedulerControlPort,
    ) -> ActivationDirective:
        state = self._states[request_id]
        readiness = state.readiness
        state.readiness = None
        valid = (
            state.phase is SchedulerReusePhase.WAITING
            and num_computed_tokens == state.plan.calibration_start
            and readiness is not None
            and readiness.get("status") == "ready"
            and readiness.get("plan") == state.plan.to_dict()
        )
        activated = (
            control.activate_csk_reuse(state.ticket, request_id) if valid else None
        )
        if activated != state.plan:
            state.phase = SchedulerReusePhase.FALLBACK
            self._release_scheduler_lease(state, control)
            return ActivationDirective(activated=False)
        state.phase = SchedulerReusePhase.ACTIVATED
        external_tokens = state.plan.reuse_end - state.plan.calibration_start
        profile_event(
            "csk_reuse_scheduler_activate",
            request_id,
            ticket=state.ticket,
            external_tokens=external_tokens,
        )
        return ActivationDirective(
            activated=True,
            external_tokens=external_tokens,
        )

    def allocation_failed(
        self, request_id: str, control: SchedulerControlPort | None
    ) -> None:
        state = self._states.get(request_id)
        if state is None or state.phase is not SchedulerReusePhase.ACTIVATED:
            return
        self._release_scheduler_lease(state, control)
        state.phase = SchedulerReusePhase.FALLBACK

    def handoff_to_worker(self, request_id: str) -> None:
        state = self._states[request_id]
        if (
            state.phase is not SchedulerReusePhase.ACTIVATED
            or state.lease_owner is not LeaseOwner.SCHEDULER
        ):
            raise RuntimeError("CSKCache lease cannot be handed to the worker")
        state.lease_owner = LeaseOwner.WORKER

    def consume_invalid_blocks(
        self,
        invalid_block_ids: set[int],
        *,
        request_block_ids: Mapping[str, Sequence[int]],
        block_size: int,
    ) -> tuple[tuple[FailedReuseRange, ...], set[int]]:
        failures: list[FailedReuseRange] = []
        consumed: set[int] = set()
        for request_id, state in self._states.items():
            if state.phase is not SchedulerReusePhase.ACTIVATED:
                continue
            blocks = request_block_ids.get(request_id)
            if blocks is None:
                continue
            first = state.plan.calibration_start // block_size
            last = (state.plan.reuse_end + block_size - 1) // block_size
            reserved = frozenset(blocks[first:last])
            if not reserved.intersection(invalid_block_ids):
                continue
            consumed.update(reserved)
            state.phase = SchedulerReusePhase.FALLBACK
            state.lease_owner = LeaseOwner.RELEASED
            failures.append(
                FailedReuseRange(
                    request_id=request_id,
                    recompute_from=state.plan.calibration_start,
                    block_ids=reserved,
                )
            )
        return tuple(failures), consumed

    def finish(
        self, request_id: str, control: SchedulerControlPort | None
    ) -> None:
        state = self._states.pop(request_id, None)
        if state is not None:
            self._release_scheduler_lease(state, control)

    def state(self, request_id: str) -> SchedulerReuseState | None:
        return self._states.get(request_id)

    def request_ids(self) -> tuple[str, ...]:
        return tuple(self._states)

    @staticmethod
    def _matches_verified_request(
        plan: ReusePlan | None,
        verified: Mapping[object, object],
        *,
        request_id: str,
        prompt_tokens: int,
        block_alignment: int,
    ) -> bool:
        return plan is not None and (
            plan.request_id == request_id
            and plan.ticket == verified.get("ticket")
            and plan.cache_object_id == verified.get("cache_object_id")
            and plan.segment_start == verified.get("segment_start")
            and plan.segment_end == verified.get("segment_end")
            and plan.segment_end <= prompt_tokens
            and plan.block_alignment == block_alignment
        )

    @staticmethod
    def _release_scheduler_lease(
        state: SchedulerReuseState,
        control: SchedulerControlPort | None,
    ) -> bool:
        if state.lease_owner is not LeaseOwner.SCHEDULER or control is None:
            return False
        released = control.release_csk_reuse(state.ticket)
        if released:
            state.lease_owner = LeaseOwner.RELEASED
        return released
