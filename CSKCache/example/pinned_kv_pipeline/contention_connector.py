"""Example-only connector for the CE/SM bandwidth-contention diagnosis."""

from __future__ import annotations

import time
from collections.abc import Sequence
from enum import Enum
from typing import Any

import torch

from cskcache.execution.base import ReuseExecutionResult
from cskcache.execution.executor import CSKCacheReuseExecutor
from cskcache.host_memory.transfers import bind_layer_buffers
from cskcache.integrations.vllm.connector import CSKCacheConnectorV1
from cskcache.profile import profile_event
from cskcache.runtime.base import ReusePlan


class ContentionArm(str, Enum):
    H2D_ONLY = "h2d_only"
    CALIBRATION_ONLY = "calibration_only"
    CONCURRENT = "concurrent"
    FULL = "full"


class _ContentionDiagnosticExecutor:
    """Measure isolated H2D/forward interaction before normal restoration.

    The diagnostic copies the same pinned K/V range used by the production
    packed-layer path into two private GPU buffers.  It deliberately excludes
    RoPE adaptation, residual correction, and PagedKV installation.  After the
    measured arm completes, the normal executor restores the request so the
    serving request never consumes diagnostic buffers.
    """

    def __init__(
        self,
        data_plane: Any,
        *,
        expected_layers: int,
        arm: ContentionArm,
        execution_order: str,
    ) -> None:
        self._data_plane = data_plane
        self._expected_layers = expected_layers
        self._arm = arm
        self._normal = CSKCacheReuseExecutor(
            data_plane,
            expected_layers=expected_layers,
            execution_order=execution_order,
        )

    def execute(
        self,
        plan: ReusePlan,
        *,
        token_ids: Sequence[int],
        kvcaches: Sequence[torch.Tensor],
        slot_mapping: torch.Tensor,
    ) -> ReuseExecutionResult:
        if self._arm is ContentionArm.FULL:
            start_ns = time.perf_counter_ns()
            result = self._normal.execute(
                plan,
                token_ids=token_ids,
                kvcaches=kvcaches,
                slot_mapping=slot_mapping,
            )
            profile_event(
                "cskcache_contention_diagnostic",
                plan.request_id,
                arm=self._arm.value,
                wall_ms=(time.perf_counter_ns() - start_ns) / 1_000_000,
                h2d_layers=[],
                compute_layers=[],
            )
            return result

        buffers = tuple(
            self._data_plane.get_active_layer_buffers(
                plan.ticket, plan.request_id
            )
        )
        bound = bind_layer_buffers(
            buffers,
            token_count=plan.segment_end - plan.segment_start,
        )
        if len(bound.layer_objects) != self._expected_layers or any(
            len(objects) != 1 for objects in bound.layer_objects
        ):
            raise ValueError(
                "contention diagnosis requires one packed Host object per layer"
            )

        gpu_connector = self._data_plane._gpu_connector
        gpu_connector.initialize_kvcaches_ptr(kvcaches=kvcaches)
        gpu_connector._lazy_initialize_buffer(kvcaches)
        previous_full_mapping = gpu_connector._staged_full_slot_mapping
        gpu_connector._staged_full_slot_mapping = slot_mapping[: plan.reuse_end]

        source_start = plan.calibration_start - plan.segment_start
        source_end = plan.reuse_end - plan.segment_start
        sources = tuple(
            objects[0].tensor[:, source_start:source_end]
            for objects in bound.layer_objects
        )
        if any(source is None or not source.is_pinned() for source in sources):
            raise ValueError("diagnostic H2D sources must be pinned CPU tensors")

        load_stream = gpu_connector.load_stream
        destinations = tuple(
            torch.empty_like(sources[0], device=gpu_connector.device)
            for _ in range(2)
        )
        t0 = torch.cuda.Event(enable_timing=True)
        t0.record()
        t0.synchronize()
        h2d_events: list[tuple[int, torch.cuda.Event, torch.cuda.Event]] = []
        compute_events: list[
            tuple[int, torch.cuda.Event, torch.cuda.Event]
        ] = []
        calibration_model = None
        start_ns = time.perf_counter_ns()

        def submit_h2d(layer_id: int) -> torch.cuda.Event:
            source = sources[layer_id]
            destination = destinations[layer_id % len(destinations)]
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(load_stream):
                start.record(load_stream)
                destination[0].copy_(source[0], non_blocking=True)
                destination[1].copy_(source[1], non_blocking=True)
                end.record(load_stream)
            h2d_events.append((layer_id, start, end))
            return end

        def compute_layer(layer_id: int) -> None:
            assert calibration_model is not None
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                next(calibration_model)
            except StopIteration as exc:
                raise RuntimeError(
                    f"calibration model ended before layer {layer_id}"
                ) from exc
            end.record()
            compute_events.append((layer_id, start, end))

        try:
            if self._arm is ContentionArm.H2D_ONLY:
                for layer_id in range(self._expected_layers):
                    submit_h2d(layer_id).synchronize()
            else:
                calibration_model = self._data_plane.open_calibration_model(
                    plan, token_ids
                )
                if self._arm is ContentionArm.CALIBRATION_ONLY:
                    for layer_id in range(self._expected_layers):
                        compute_layer(layer_id)
                else:
                    pending = submit_h2d(0)
                    pending.synchronize()
                    for layer_id in range(self._expected_layers):
                        next_layer = layer_id + 1
                        if next_layer < self._expected_layers:
                            pending = submit_h2d(next_layer)
                        compute_layer(layer_id)
                        if next_layer < self._expected_layers:
                            pending.synchronize()
            torch.cuda.synchronize()
            wall_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
            all_events = [
                (start, end)
                for _layer, start, end in (*h2d_events, *compute_events)
            ]
            start_ms = min(
                (t0.elapsed_time(start) for start, _end in all_events),
                default=0.0,
            )
            end_ms = max(
                (t0.elapsed_time(end) for _start, end in all_events),
                default=0.0,
            )
            profile_event(
                "cskcache_contention_diagnostic",
                plan.request_id,
                arm=self._arm.value,
                wall_ms=wall_ms,
                gpu_span_ms=end_ms - start_ms,
                h2d_layers=[
                    {
                        "layer": layer_id,
                        "gpu_ms": start.elapsed_time(end),
                        "start_ms": t0.elapsed_time(start),
                        "end_ms": t0.elapsed_time(end),
                    }
                    for layer_id, start, end in h2d_events
                ],
                compute_layers=[
                    {
                        "layer": layer_id,
                        "gpu_ms": start.elapsed_time(end),
                        "start_ms": t0.elapsed_time(start),
                        "end_ms": t0.elapsed_time(end),
                    }
                    for layer_id, start, end in compute_events
                ],
            )
        finally:
            if calibration_model is not None:
                calibration_model.close()
            gpu_connector._staged_full_slot_mapping = previous_full_mapping

        return self._normal.execute(
            plan,
            token_ids=token_ids,
            kvcaches=kvcaches,
            slot_mapping=slot_mapping,
        )


class ContentionDiagnosticConnectorV1(CSKCacheConnectorV1):
    """CSKCache connector whose worker runs one example-only diagnostic arm."""

    def _initialize_csk_worker(self) -> None:
        super()._initialize_csk_worker()
        worker = self._csk_worker
        if worker is None:
            return
        arm = ContentionArm(
            str(
                self._lmcache_engine.config.get_extra_config_value(
                    "csk_contention_arm", ContentionArm.FULL.value
                )
            )
        )
        execution_order = str(
            self._lmcache_engine.config.get_extra_config_value(
                "csk_execution_order", "h2d_first"
            )
        )
        # This replacement is intentionally confined to the example connector;
        # the production worker and connector remain unchanged.
        worker._executor = _ContentionDiagnosticExecutor(
            worker._data_plane,
            expected_layers=worker._data_plane.num_layers,
            arm=arm,
            execution_order=execution_order,
        )
