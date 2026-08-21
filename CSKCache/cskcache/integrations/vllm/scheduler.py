"""CSKCache implementation of vLLM's generic scheduler-extension contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ...runtime.coordinator import SchedulerReuseCoordinator
from .base import VERIFIED_REQUEST_FIELD


class CSKCacheSchedulerExtension:
    """Bind CSKCache's scheduler state machine to one connector transport."""

    def __init__(self, control: Any) -> None:
        self._control = control
        self._runtime = SchedulerReuseCoordinator()

    def register(
        self,
        request: Any,
        *,
        block_alignment: int,
        async_scheduling: bool,
    ) -> bool:
        verified = (request.kv_transfer_params or {}).get(
            VERIFIED_REQUEST_FIELD
        )
        return self._runtime.register(
            verified,
            request_id=request.request_id,
            prompt_tokens=request.num_prompt_tokens,
            block_alignment=block_alignment,
            async_scheduling=async_scheduling,
            control=self._control,
        )

    def limit_prefill(
        self,
        request_id: str,
        *,
        num_computed_tokens: int,
        num_new_tokens: int,
    ) -> int:
        return self._runtime.limit_prefill(
            request_id,
            num_computed_tokens=num_computed_tokens,
            num_new_tokens=num_new_tokens,
            control=self._control,
        )

    def is_waiting(self, request_id: str) -> bool:
        return self._runtime.is_waiting(request_id)

    def poll(self, request_id: str) -> bool:
        return self._runtime.poll(request_id, self._control)

    def activate(self, request_id: str, *, num_computed_tokens: int):
        return self._runtime.activate(
            request_id,
            num_computed_tokens=num_computed_tokens,
            control=self._control,
        )

    def allocation_failed(self, request_id: str) -> None:
        self._runtime.allocation_failed(request_id, self._control)

    def handoff_to_worker(self, request_id: str) -> None:
        self._runtime.handoff_to_worker(request_id)

    def reaches_waiting_boundary(
        self,
        request_id: str,
        *,
        num_computed_tokens: int,
        stopped: bool,
        produced_tokens: bool,
        request_running: bool,
        in_flight_tokens: int,
    ) -> bool:
        return self._runtime.reaches_calibration_boundary(
            request_id,
            num_computed_tokens=num_computed_tokens,
            stopped=stopped,
            produced_tokens=produced_tokens,
            request_running=request_running,
            in_flight_tokens=in_flight_tokens,
        )

    def request_ids(self) -> tuple[str, ...]:
        return self._runtime.request_ids()

    def consume_invalid_blocks(
        self,
        invalid_block_ids: set[int],
        *,
        request_block_ids: Mapping[str, Sequence[int]],
        block_size: int,
    ):
        return self._runtime.consume_invalid_blocks(
            invalid_block_ids,
            request_block_ids=request_block_ids,
            block_size=block_size,
        )

    def finish(self, request_id: str) -> None:
        self._runtime.finish(request_id, self._control)
