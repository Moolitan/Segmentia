"""Layer-ordered online reuse execution owned by CSKCache."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .base import (
    ExecutionOrder,
    LayerwiseCalibrationModel,
    LayerwiseReuseStream,
    ReuseDataPlane,
    ReuseExecutionResult,
)
from .compute_first import execute_compute_first
from .corrector import ContextAwareKVCorrector
from .h2d_first import execute_h2d_first
from ..profile import PROFILE_ENABLED, profile_event
from ..runtime.base import ReusePlan


class CSKCacheReuseExecutor:
    """Drive two layerwise generators: H2D and C/R/I computation."""

    def __init__(
        self,
        data_plane: ReuseDataPlane,
        *,
        expected_layers: int,
        execution_order: str = "h2d_first",
        corrector: ContextAwareKVCorrector | None = None,
    ) -> None:
        if expected_layers <= 0:
            raise ValueError("expected_layers must be positive")
        try:
            parsed_order = ExecutionOrder(execution_order)
        except ValueError as exc:
            raise ValueError(
                f"unsupported CSKCache execution order: {execution_order}"
            ) from exc
        self._data_plane = data_plane
        self._expected_layers = expected_layers
        self._execution_order = parsed_order.value
        self._corrector = corrector or ContextAwareKVCorrector()

    def execute(
        self,
        plan: ReusePlan,
        *,
        token_ids: Sequence[int],
        kvcaches: Sequence[torch.Tensor],
        slot_mapping: torch.Tensor,
    ) -> ReuseExecutionResult:
        """Run ``H2D(l+1) || [C_l(P) -> R_l(P) -> I_l(B-P)]``."""

        if len(token_ids) < plan.reuse_start:
            raise ValueError("token_ids do not cover the calibration range")
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
        calibration_model = self._data_plane.open_calibration_model(
            plan, token_ids
        )
        try:
            if self._execution_order == "h2d_first":
                compute_events = execute_h2d_first(
                    expected_layers=self._expected_layers,
                    plan=plan,
                    stream=stream,
                    calibration_model=calibration_model,
                    compute_layer=self._compute_correct_install_layer,
                    profile_t0_event=profile_t0_event,
                )
            else:
                compute_events = execute_compute_first(
                    expected_layers=self._expected_layers,
                    plan=plan,
                    stream=stream,
                    calibration_model=calibration_model,
                    compute_layer=self._compute_correct_install_layer,
                    profile_t0_event=profile_t0_event,
                )

            try:
                next(calibration_model)
            except StopIteration:
                pass
            else:
                raise RuntimeError("calibration model produced too many layers")
            stream.finish()

            if profile_t0_event is not None:
                profile_event(
                    "cskcache_layer_compute",
                    plan.request_id,
                    ticket=plan.ticket,
                    execution_order=self._execution_order,
                    synchronization="torch_cuda_device_wide_per_layer",
                    calibration_correct_install=[
                        {
                            "layer": event_layer_id,
                            "gpu_ms": start.elapsed_time(end),
                            "start_ms": profile_t0_event.elapsed_time(start),
                            "end_ms": profile_t0_event.elapsed_time(end),
                        }
                        for event_layer_id, start, end in compute_events
                    ],
                )
        except Exception:
            stream.abort()
            raise
        finally:
            calibration_model.close()

        return ReuseExecutionResult(
            ticket=plan.ticket,
            request_id=plan.request_id,
            processed_layers=self._expected_layers,
            correction_alpha=plan.correction_alpha,
        )

    def _compute_correct_install_layer(
        self,
        layer_id: int,
        plan: ReusePlan,
        stream: LayerwiseReuseStream,
        calibration_model: LayerwiseCalibrationModel,
        profile_t0_event: torch.cuda.Event | None,
    ) -> tuple[torch.cuda.Event, torch.cuda.Event] | None:

        calibration_tokens = plan.calibration_end - plan.calibration_start
        start = (
            torch.cuda.Event(enable_timing=True)
            if profile_t0_event is not None
            else None
        )
        end = (
            torch.cuda.Event(enable_timing=True)
            if profile_t0_event is not None
            else None
        )
        if start is not None:
            start.record()

        # C_l(P): actual auxiliary-model forward against the current request
        # prefix, returning position-corrected calibration KV.
        try:
            recomputed_key, recomputed_value = next(calibration_model)
        except StopIteration as exc:
            raise RuntimeError(
                f"calibration model ended before layer {layer_id}"
            ) from exc

        staged_key = stream.staged_key(layer_id)
        self._data_plane.mark_layer_loaded(plan.ticket, plan.request_id, layer_id)

        # Fully recomputed calibration KV becomes part of this request.
        stream.commit_calibration(layer_id, recomputed_key, recomputed_value)

        # Estimate the residual from P tokens, correct the remaining offline
        # suffix, and install that suffix into PagedKV.
        self._corrector.correct_key_(
            staged_key,
            recomputed_key,
            calibration_tokens=calibration_tokens,
            suffix_offset=calibration_tokens,
            alpha=plan.correction_alpha,
        )
        stream.commit_layer(layer_id)
        self._data_plane.mark_layer_corrected(
            plan.ticket, plan.request_id, layer_id
        )

        if end is None or start is None:
            return None
        end.record()
        return start, end
