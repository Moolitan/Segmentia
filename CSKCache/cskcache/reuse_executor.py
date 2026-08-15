"""Layer-ordered online reuse execution owned by CSKCache."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import torch

from .context_aware_kv_corrector import ContextAwareKVCorrector
from .profile import PROFILE_ENABLED, profile_event
from .reuse_state import ReusePlan


class LayerwiseReuseStream(Protocol):
    """One request-local physical stream supplied by the serving backend."""

    def stage_layer(self, layer_id: int) -> None: ...

    def staged_key(self, layer_id: int) -> torch.Tensor: ...

    def recomputed_key(
        self, layer_id: int, start: int, end: int
    ) -> torch.Tensor: ...

    def commit_layer(self, layer_id: int) -> None: ...

    def finish(self) -> None: ...


class ReuseDataPlane(Protocol):
    """Narrow physical interface required by the CSKCache executor."""

    def get_active_layer_buffers(
        self, ticket: str, request_id: str
    ) -> Sequence[Any]: ...

    def open_layer_stream(
        self,
        plan: ReusePlan,
        buffers: Sequence[Any],
        *,
        kvcaches: Sequence[torch.Tensor],
        slot_mapping: torch.Tensor,
        profile_t0_event: torch.cuda.Event | None = None,
    ) -> LayerwiseReuseStream: ...

    def mark_layer_loaded(
        self, ticket: str, request_id: str, layer_id: int
    ) -> None: ...

    def mark_layer_corrected(
        self, ticket: str, request_id: str, layer_id: int
    ) -> None: ...


@dataclass(frozen=True)
class ReuseExecutionResult:
    """Evidence returned only after the complete layer group is committed."""

    ticket: str
    request_id: str
    processed_layers: int
    correction_alpha: float


class CSKCacheReuseExecutor:
    """Drive H2D, Key correction, and paged-KV commit in layer order."""

    def __init__(
        self,
        data_plane: ReuseDataPlane,
        *,
        expected_layers: int,
        corrector: ContextAwareKVCorrector | None = None,
    ) -> None:
        if expected_layers <= 0:
            raise ValueError("expected_layers must be positive")
        self._data_plane = data_plane
        self._expected_layers = expected_layers
        self._corrector = corrector or ContextAwareKVCorrector()

    def execute(
        self,
        plan: ReusePlan,
        *,
        kvcaches: Sequence[torch.Tensor],
        slot_mapping: torch.Tensor,
    ) -> ReuseExecutionResult:
        """Execute one authenticated plan without storage or key lookup."""

        plan.validate()
        if len(kvcaches) != self._expected_layers:
            raise ValueError("KV cache layer count differs from the CSKCache model")
        if not isinstance(slot_mapping, torch.Tensor) or slot_mapping.ndim != 1:
            raise ValueError("slot_mapping must be a one-dimensional tensor")
        if len(slot_mapping) < plan.reuse_end:
            raise ValueError("slot_mapping does not cover the CSKCache reuse range")

        buffers = tuple(
            self._data_plane.get_active_layer_buffers(
                plan.ticket, plan.request_id
            )
        )
        if len(buffers) != self._expected_layers:
            raise RuntimeError("CSKCache returned an incomplete layer group")
        profile_t0_event = None
        if PROFILE_ENABLED:
            # Establish one completed CUDA timestamp before either the default
            # compute stream or LMCache's load stream receives pipeline work.
            # All per-layer events can then be placed on the same GPU clock.
            profile_t0_event = torch.cuda.Event(enable_timing=True)
            profile_t0_event.record()
            profile_t0_event.synchronize()
        stream = self._data_plane.open_layer_stream(
            plan,
            buffers,
            kvcaches=kvcaches,
            slot_mapping=slot_mapping,
            profile_t0_event=profile_t0_event,
        )
        calibration_tokens = plan.calibration_end - plan.calibration_start
        suffix_offset = plan.reuse_start - plan.calibration_start
        correction_events: list[tuple[int, torch.cuda.Event, torch.cuda.Event]] = []
        commit_events: list[tuple[int, torch.cuda.Event, torch.cuda.Event]] = []

        for layer_id in range(self._expected_layers):
            stream.stage_layer(layer_id)
            self._data_plane.mark_layer_loaded(
                plan.ticket, plan.request_id, layer_id
            )
            staged_key = stream.staged_key(layer_id)
            recomputed_key = stream.recomputed_key(
                layer_id, plan.calibration_start, plan.calibration_end
            )
            correction_start = (
                torch.cuda.Event(enable_timing=True) if PROFILE_ENABLED else None
            )
            correction_end = (
                torch.cuda.Event(enable_timing=True) if PROFILE_ENABLED else None
            )
            if correction_start is not None:
                correction_start.record()
            self._corrector.correct_key_(
                staged_key,
                recomputed_key,
                calibration_tokens=calibration_tokens,
                suffix_offset=suffix_offset,
                alpha=plan.correction_alpha,
            )
            if correction_end is not None and correction_start is not None:
                correction_end.record()
                correction_events.append(
                    (layer_id, correction_start, correction_end)
                )
            commit_start = (
                torch.cuda.Event(enable_timing=True) if PROFILE_ENABLED else None
            )
            commit_end = (
                torch.cuda.Event(enable_timing=True) if PROFILE_ENABLED else None
            )
            if commit_start is not None:
                commit_start.record()
            stream.commit_layer(layer_id)
            if commit_end is not None and commit_start is not None:
                commit_end.record()
                commit_events.append((layer_id, commit_start, commit_end))
            self._data_plane.mark_layer_corrected(
                plan.ticket, plan.request_id, layer_id
            )

        stream.finish()
        if PROFILE_ENABLED:
            torch.cuda.synchronize()
            assert profile_t0_event is not None
            correction_by_layer = [
                {
                    "layer": layer_id,
                    "gpu_ms": start.elapsed_time(end),
                    "start_ms": profile_t0_event.elapsed_time(start),
                    "end_ms": profile_t0_event.elapsed_time(end),
                }
                for layer_id, start, end in correction_events
            ]
            commit_by_layer = [
                {
                    "layer": layer_id,
                    "gpu_ms": start.elapsed_time(end),
                    "start_ms": profile_t0_event.elapsed_time(start),
                    "end_ms": profile_t0_event.elapsed_time(end),
                }
                for layer_id, start, end in commit_events
            ]
            profile_event(
                "csk_reuse_gpu_breakdown",
                plan.request_id,
                ticket=plan.ticket,
                shared_cuda_timeline=True,
                timeline_origin="reuse_executor_start",
                correction_gpu_ms=sum(
                    item["gpu_ms"] for item in correction_by_layer
                ),
                commit_gpu_ms=sum(item["gpu_ms"] for item in commit_by_layer),
                correction_per_layer=correction_by_layer,
                commit_per_layer=commit_by_layer,
            )
        return ReuseExecutionResult(
            ticket=plan.ticket,
            request_id=plan.request_id,
            processed_layers=self._expected_layers,
            correction_alpha=plan.correction_alpha,
        )
