"""Layer-ordered online reuse execution owned by CSKCache."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .base import (
    CalibrationResidualCorrectionMethod,
    DeviationTopKRecomputeMethod,
    DirectReuseMethod,
    ExecutionOrder,
    LayerComputeEvents,
    LayerwiseCalibrationModel,
    LayerwiseReuseStream,
    ReuseDataPlane,
    ReuseExecutionResult,
    execution_method_for,
)
from .compute_first import execute_compute_first
from .corrector import ContextAwareKVCorrector
from .h2d_first import execute_h2d_first
from ..profile import PROFILE_ENABLED, profile_event
from ..runtime.base import CorrectionStrategy, ReusePlan


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
        strategy = CorrectionStrategy(plan.correction_strategy)
        method = execution_method_for(strategy)
        if isinstance(method, DirectReuseMethod):
            try:
                self._execute_direct(plan, stream)
                stream.finish()
            except Exception:
                stream.abort()
                raise
            return ReuseExecutionResult(
                ticket=plan.ticket,
                request_id=plan.request_id,
                processed_layers=self._expected_layers,
                correction_alpha=plan.correction_alpha,
                correction_strategy=strategy,
                method=method,
            )

        if isinstance(method, DeviationTopKRecomputeMethod):
            self._execute_deviation_topk(
                plan,
                stream,
                token_ids=token_ids,
            )
            return ReuseExecutionResult(
                ticket=plan.ticket,
                request_id=plan.request_id,
                processed_layers=self._expected_layers,
                correction_alpha=plan.correction_alpha,
                correction_strategy=strategy,
                method=method,
            )

        if not isinstance(method, CalibrationResidualCorrectionMethod):
            raise RuntimeError(f"unsupported reuse execution method: {method.name}")

        calibration_model = self._data_plane.open_calibration_model(plan, token_ids)
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
                            "gpu_ms": events.start.elapsed_time(events.end),
                            "start_ms": profile_t0_event.elapsed_time(
                                events.start
                            ),
                            "end_ms": profile_t0_event.elapsed_time(events.end),
                            "calibration_forward_ms": events.start.elapsed_time(
                                events.calibration_forward_end
                            ),
                            "calibration_commit_ms": (
                                events.calibration_forward_end.elapsed_time(
                                    events.calibration_commit_end
                                )
                            ),
                            "residual_correction_ms": (
                                events.calibration_commit_end.elapsed_time(
                                    events.residual_correction_end
                                )
                            ),
                            "suffix_commit_ms": (
                                events.residual_correction_end.elapsed_time(
                                    events.end
                                )
                            ),
                        }
                        for event_layer_id, events in compute_events
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
            correction_strategy=strategy,
            method=method,
        )

    def _execute_deviation_topk(
        self,
        plan: ReusePlan,
        stream: LayerwiseReuseStream,
        *,
        token_ids: Sequence[int],
    ) -> None:
        """Run full-through-check-layer then selective top-k recomputation."""

        if plan.deviation_check_layer >= self._expected_layers:
            stream.abort()
            raise ValueError("deviation_topk check layer is outside the model")
        candidate_tokens = plan.reuse_end - plan.reuse_start
        selected_tokens = max(
            1,
            int(candidate_tokens * plan.deviation_recompute_ratio),
        )
        model = None
        try:
            model = self._data_plane.open_deviation_topk_model(plan, token_ids)
            stream.submit_layer(0)
            stream.wait_layer(0)
            for layer_id in range(self._expected_layers):
                next_layer = layer_id + 1
                if (
                    self._execution_order == ExecutionOrder.H2D_FIRST.value
                    and next_layer < self._expected_layers
                ):
                    stream.submit_layer(next_layer)
                try:
                    layer_result = next(model)
                except StopIteration as exc:
                    raise RuntimeError(
                        f"deviation model ended before layer {layer_id}"
                    ) from exc
                expected_recomputed = (
                    candidate_tokens
                    if layer_id < plan.deviation_check_layer
                    else selected_tokens
                )
                if (
                    layer_result.layer_id != layer_id
                    or layer_result.candidate_tokens != candidate_tokens
                    or layer_result.recomputed_tokens != expected_recomputed
                    or layer_result.selection_applied
                    != (layer_id == plan.deviation_check_layer)
                ):
                    raise RuntimeError(
                        "deviation_topk model returned inconsistent layer evidence"
                    )
                self._data_plane.mark_layer_loaded(
                    plan.ticket, plan.request_id, layer_id
                )
                stream.commit_layer(layer_id)
                self._data_plane.mark_layer_corrected(
                    plan.ticket, plan.request_id, layer_id
                )
                profile_event(
                    "cskcache_deviation_topk_layer",
                    plan.request_id,
                    ticket=plan.ticket,
                    layer=layer_id,
                    candidate_tokens=candidate_tokens,
                    recomputed_tokens=layer_result.recomputed_tokens,
                    selection_applied=layer_result.selection_applied,
                    recompute_ratio=plan.deviation_recompute_ratio,
                    check_layer=plan.deviation_check_layer,
                )
                if next_layer < self._expected_layers:
                    if self._execution_order == ExecutionOrder.COMPUTE_FIRST.value:
                        stream.submit_layer(next_layer)
                    stream.wait_layer(next_layer)
            try:
                next(model)
            except StopIteration:
                pass
            else:
                raise RuntimeError("deviation model produced too many layers")
            stream.finish()
        except Exception:
            stream.abort()
            raise
        finally:
            if model is not None:
                model.close()

    def _execute_direct(
        self,
        plan: ReusePlan,
        stream: LayerwiseReuseStream,
    ) -> None:
        """Install staged offline KV without running the auxiliary model."""

        stream.submit_layer(0)
        stream.wait_layer(0)
        for layer_id in range(self._expected_layers):
            self._data_plane.mark_layer_loaded(
                plan.ticket, plan.request_id, layer_id
            )
            stream.commit_layer(layer_id)
            self._data_plane.mark_layer_corrected(
                plan.ticket, plan.request_id, layer_id
            )
            next_layer = layer_id + 1
            if next_layer < self._expected_layers:
                stream.submit_layer(next_layer)
                stream.wait_layer(next_layer)

    def _compute_correct_install_layer(
        self,
        layer_id: int,
        plan: ReusePlan,
        stream: LayerwiseReuseStream,
        calibration_model: LayerwiseCalibrationModel,
        profile_t0_event: torch.cuda.Event | None,
    ) -> LayerComputeEvents | None:

        calibration_tokens = plan.calibration_end - plan.calibration_start
        events = (
            [torch.cuda.Event(enable_timing=True) for _ in range(5)]
            if profile_t0_event is not None
            else None
        )
        if events is not None:
            events[0].record()

        # C_l(P): actual auxiliary-model forward against the current request
        # prefix, returning position-corrected calibration KV.
        try:
            recomputed_key, recomputed_value = next(calibration_model)
        except StopIteration as exc:
            raise RuntimeError(
                f"calibration model ended before layer {layer_id}"
            ) from exc
        if events is not None:
            events[1].record()

        staged_key = stream.staged_key(layer_id)
        self._data_plane.mark_layer_loaded(plan.ticket, plan.request_id, layer_id)

        # Fully recomputed calibration KV becomes part of this request.
        stream.commit_calibration(layer_id, recomputed_key, recomputed_value)
        if events is not None:
            events[2].record()

        # Estimate the residual from P tokens, correct the remaining offline
        # suffix, and install that suffix into PagedKV.
        self._corrector.correct_key_(
            staged_key,
            recomputed_key,
            calibration_tokens=calibration_tokens,
            suffix_offset=calibration_tokens,
            alpha=plan.correction_alpha,
        )
        if events is not None:
            events[3].record()
        stream.commit_layer(layer_id)
        self._data_plane.mark_layer_corrected(
            plan.ticket, plan.request_id, layer_id
        )

        if events is None:
            return None
        events[4].record()
        return LayerComputeEvents(*events)
