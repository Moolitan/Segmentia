"""Typed CSKCache control transport and allocation ownership."""

from __future__ import annotations

from .base import (
    KVBlockAllocation,
    LookupControlPort,
    ReuseAllocation,
    ReusePlan,
)


class PlanTransportCoordinator:
    """Validate a transported plan once and own its activation lifecycle."""

    def __init__(self) -> None:
        self._prepared: dict[str, ReusePlan] = {}
        self._active: dict[str, ReusePlan] = {}

    def prepare(
        self,
        control: LookupControlPort,
        ticket: str,
        request_id: str,
        block_alignment: int,
    ) -> ReusePlan | None:
        raw_plan = control.prepare_csk_reuse(
            ticket, request_id, block_alignment
        )
        if raw_plan is None:
            return None
        try:
            plan = ReusePlan.from_dict(raw_plan)
        except (TypeError, ValueError):
            control.cancel_csk_prefetch(ticket, "invalid_reuse_plan")
            return None
        if (
            plan.ticket != ticket
            or plan.request_id != request_id
            or plan.block_alignment != block_alignment
        ):
            control.cancel_csk_prefetch(ticket, "plan_binding_mismatch")
            return None
        self._prepared[request_id] = plan
        return plan

    def activate(
        self,
        control: LookupControlPort,
        ticket: str,
        request_id: str,
    ) -> ReusePlan | None:
        raw_plan = control.activate_csk_reuse(ticket, request_id)
        prepared = self._prepared.pop(request_id, None)
        if raw_plan is None or prepared is None:
            return None
        if raw_plan != prepared.to_dict():
            control.release_csk_reuse(ticket)
            return None
        self._active[request_id] = prepared
        return prepared

    def bind_allocation(
        self,
        request_id: str,
        *,
        num_external_tokens: int,
        blocks: KVBlockAllocation | None,
    ) -> ReuseAllocation | None:
        plan = self._active.pop(request_id, None)
        if plan is None:
            return None
        expected_external = plan.reuse_end - plan.calibration_start
        if num_external_tokens != expected_external:
            raise ValueError(
                "CSKCache external-token count differs from ReusePlan"
            )
        if blocks is None:
            raise ValueError("CSKCache activation has no PagedKV allocation")
        block_groups = blocks.get_block_ids()
        if len(block_groups) != 1:
            raise ValueError("CSKCache currently requires one PagedKV group")
        return ReuseAllocation(
            plan=plan,
            computed_start=plan.calibration_start,
            computed_end=plan.reuse_end,
            block_ids=tuple(block_groups[0]),
        )

    def release(self, control: LookupControlPort, ticket: str) -> bool:
        self._forget(ticket)
        return control.release_csk_reuse(ticket)

    def cancel(
        self, control: LookupControlPort, ticket: str, reason: str
    ) -> None:
        self._forget(ticket)
        control.cancel_csk_prefetch(ticket, reason)

    def _forget(self, ticket: str) -> None:
        self._prepared = {
            request_id: plan
            for request_id, plan in self._prepared.items()
            if plan.ticket != ticket
        }
        self._active = {
            request_id: plan
            for request_id, plan in self._active.items()
            if plan.ticket != ticket
        }
