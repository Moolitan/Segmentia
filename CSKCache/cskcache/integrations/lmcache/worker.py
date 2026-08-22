"""Worker-side LMCache/vLLM physical adapter and execution driver."""

from __future__ import annotations

import os
from collections.abc import Generator, Sequence
from typing import Any

import torch

from ...execution.executor import CSKCacheReuseExecutor
from ...profile import PROFILE_ENABLED, profile_event
from ...runtime.base import ReusePlan
from ...host_memory.transfers import PerObjectCopySession, bind_layer_buffers
from .forward_profile import CalibrationForwardProfiler
from lmcache import torch_device_type
from lmcache.v1.compute.models.utils import (
    VLLMModelTracker,
    infer_model_from_vllm,
)
from lmcache.v1.gpu_connector.utils import assert_layerwise_gpu_connector


class _CSKCalibrationBlender:
    """Supply current-prefix KV to LMCache's auxiliary model.

    Unlike CacheBlend's deviation selector, this adapter always forwards the
    fixed contiguous calibration suffix.  For each layer it combines the
    request's already-computed PagedKV prefix with the fresh calibration K/V,
    then retains the fresh K/V for CSKCache's residual/install stage.
    """

    def __init__(self, gpu_connector: Any) -> None:
        self._gpu_connector = gpu_connector
        self._model = None
        self._profiler: CalibrationForwardProfiler | None = None
        self._plan: ReusePlan | None = None
        self._results: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def bind_model(self, model: Any) -> None:
        self._model = model

    def bind_profiler(self, profiler: CalibrationForwardProfiler) -> None:
        self._profiler = profiler

    def begin(self, plan: ReusePlan) -> None:
        if self._plan is not None:
            raise RuntimeError("a CSK calibration model is already active")
        self._plan = plan
        self._results.clear()
        if self._profiler is not None:
            self._profiler.begin(plan)

    def finish(self) -> None:
        if self._profiler is not None:
            self._profiler.finish()
        self._plan = None
        self._results.clear()

    def take_result(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            return self._results.pop(layer_id)
        except KeyError as exc:
            raise RuntimeError(
                f"calibration model did not produce layer {layer_id} KV"
            ) from exc

    def process_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        residual: torch.Tensor,
        layer_id: int,
        attn_output: torch.Tensor | None,
        attn_metadata: Any,
    ):
        plan = self._plan
        model = self._model
        if plan is None or model is None:
            raise RuntimeError("CSK calibration model is not initialized")
        calibration_tokens = plan.calibration_end - plan.calibration_start
        if q.shape[0] != calibration_tokens:
            raise RuntimeError("calibration query length differs from ReusePlan")

        if attn_output is None:
            attn_output = torch.empty_like(q)
        if self._profiler is not None:
            self._profiler.start(layer_id, "position_build")
        positions = torch.arange(
            plan.calibration_start,
            plan.calibration_end,
            dtype=torch.int64,
            device=q.device,
        )
        if self._profiler is not None:
            self._profiler.end(layer_id, "position_build")
        layer = model.vllm_model.model.layers[layer_id]
        if self._profiler is not None:
            self._profiler.start(layer_id, "rope")
        q, k = layer.self_attn.rotary_emb(positions, q, k)
        if self._profiler is not None:
            self._profiler.end(layer_id, "rope")

        # Retain only the P fresh tokens.  The full attention bank below is a
        # temporary execution input and must not be mistaken for recomputed KV.
        fresh_key = k.reshape(calibration_tokens, -1)
        fresh_value = v.reshape(calibration_tokens, -1)
        self._results[layer_id] = (fresh_key, fresh_value)

        if self._profiler is not None:
            self._profiler.start(layer_id, "prefix_paged_kv")
        prefix_key, prefix_value = self._gpu_connector.get_paged_kv(
            layer_id, 0, plan.calibration_start
        )
        if self._profiler is not None:
            self._profiler.end(layer_id, "prefix_paged_kv")
            self._profiler.start(layer_id, "kv_concat")
        key_bank = torch.cat((prefix_key, fresh_key), dim=0)
        value_bank = torch.cat((prefix_value, fresh_value), dim=0)
        if self._profiler is not None:
            self._profiler.end(layer_id, "kv_concat")
            self._profiler.start(layer_id, "attention")
        return (
            q,
            key_bank,
            value_bank,
            residual,
            attn_output,
            attn_metadata,
        )


class LMCacheCSKDataPlane:
    """Map CSKCache's narrow layer API to existing LMCache primitives.

    This class intentionally contains no Skill policy, correction formula, or
    request state.  CSKCache decides the operation order; this adapter only
    exposes pinned buffers and GPU staging/commit operations.
    """

    def __init__(self, runtime: Any, gpu_connector: Any, engine_name: str) -> None:
        self._runtime = runtime
        self._gpu_connector = gpu_connector
        assert_layerwise_gpu_connector(gpu_connector)
        vllm_model = VLLMModelTracker.get_model(engine_name)
        self._calibration_blender = _CSKCalibrationBlender(gpu_connector)
        self.layerwise_model = infer_model_from_vllm(
            vllm_model,
            blender=self._calibration_blender,
            enable_sparse=False,
        )
        self._calibration_blender.bind_model(self.layerwise_model)
        self.num_layers = len(vllm_model.model.layers)
        profile_layers = os.getenv("CSKCACHE_FORWARD_PROFILE_LAYERS", "")
        if profile_layers:
            profiler = CalibrationForwardProfiler(
                self.layerwise_model,
                tuple(int(value) for value in profile_layers.split(",")),
            )
            self._calibration_blender.bind_profiler(profiler)

    def get_active_layer_buffers(
        self, ticket: str, request_id: str
    ) -> Sequence[Any]:
        return self._runtime.get_active_layer_buffers(ticket, request_id)

    def open_layer_stream(
        self,
        plan: ReusePlan,
        buffers: Sequence[Any],
        *,
        kvcaches: Sequence[torch.Tensor],
        slot_mapping: torch.Tensor,
        profile_t0_event: torch.cuda.Event | None = None,
    ) -> "_LMCacheCSKLayerStream":
        return _LMCacheCSKLayerStream(
            self._gpu_connector,
            plan,
            buffers,
            kvcaches=kvcaches,
            slot_mapping=slot_mapping,
            profile_t0_event=profile_t0_event,
        )

    def open_calibration_model(
        self,
        plan: ReusePlan,
        token_ids: Sequence[int],
    ) -> Generator[tuple[torch.Tensor, torch.Tensor], None, None]:
        """Create the CacheBlend-style auxiliary per-layer forward."""

        calibration_ids = token_ids[
            plan.calibration_start : plan.calibration_end
        ]
        if len(calibration_ids) != plan.calibration_end - plan.calibration_start:
            raise ValueError("token_ids do not cover the calibration range")
        tokens = torch.tensor(
            calibration_ids,
            dtype=torch.long,
            device=torch_device_type,
        )
        self._calibration_blender.begin(plan)
        model_executor = self.layerwise_model.compute_layer(
            tokens,
            position_start=plan.calibration_start,
            kv_seq_len=plan.calibration_end,
        )

        def run():
            try:
                for layer_id in range(self.num_layers):
                    next(model_executor)
                    yield self._calibration_blender.take_result(layer_id)
            finally:
                model_executor.close()
                self._calibration_blender.finish()

        return run()

    def mark_layer_loaded(
        self, ticket: str, request_id: str, layer_id: int
    ) -> None:
        self._runtime.mark_layer_loaded(ticket, request_id, layer_id)

    def mark_layer_corrected(
        self, ticket: str, request_id: str, layer_id: int
    ) -> None:
        self._runtime.mark_layer_corrected(ticket, request_id, layer_id)


class _LMCacheCSKLayerStream:
    """One request-local LMCache layerwise H2D and paged-KV stream."""

    def __init__(
        self,
        gpu_connector: Any,
        plan: ReusePlan,
        buffers: Sequence[Any],
        *,
        kvcaches: Sequence[torch.Tensor],
        slot_mapping: torch.Tensor,
        profile_t0_event: torch.cuda.Event | None = None,
    ) -> None:
        self._gpu_connector = gpu_connector
        self._plan = plan
        self._buffers = tuple(buffers)
        self._next_submit_layer = 0
        self._next_wait_layer = 0
        self._pending_layer: int | None = None
        self._finished = False

        token_count = plan.segment_end - plan.segment_start
        source_position_start = plan.source_reuse_start - (
            plan.reuse_start - plan.segment_start
        )
        if not 0 <= source_position_start < source_position_start + token_count:
            raise ValueError("CSKCache source position range is invalid")
        bound_transfer = bind_layer_buffers(
            self._buffers,
            token_count=token_count,
        )
        for step, objects in zip(
            bound_transfer.plan.steps,
            bound_transfer.layer_objects,
            strict=True,
        ):
            for source, memory_obj in zip(step.slices, objects, strict=True):
                tensor = memory_obj.tensor
                if (
                    tensor is None
                    or tensor.shape[1] != source.token_end - source.token_start
                ):
                    raise ValueError("host buffer token count is invalid")
                memory_obj.metadata.cached_positions = torch.arange(
                    source_position_start + source.token_start,
                    source_position_start + source.token_end,
                    dtype=torch.int64,
                    device=tensor.device,
                )

        first_step = bound_transfer.plan.steps[0]
        transfer_indices = tuple(
            index
            for index, source in enumerate(first_step.slices)
            if plan.segment_start + source.token_start < plan.reuse_end
        )
        self._transfer_objects = tuple(
            tuple(objects[index] for index in transfer_indices)
            for objects in bound_transfer.layer_objects
        )
        transfer_starts = [
            plan.segment_start + first_step.slices[index].token_start
            for index in transfer_indices
        ]
        transfer_ends = [
            min(
                plan.segment_start + first_step.slices[index].token_end,
                plan.reuse_end,
            )
            for index in transfer_indices
        ]

        def emit_transfer_profile(event: str, **fields: Any) -> None:
            profile_event(
                f"cskcache_{event}",
                plan.request_id,
                **fields,
            )

        self._copy_session = PerObjectCopySession(
            gpu_connector,
            transfer_starts,
            transfer_ends,
            kvcaches=kvcaches,
            slot_mapping=slot_mapping,
            staged_range_start=plan.calibration_start,
            staged_range_end=plan.reuse_end,
            deferred_write_start=plan.reuse_start,
            split_stage=True,
            transfer_profile_callback=(
                emit_transfer_profile if PROFILE_ENABLED else None
            ),
            profile_t0_event=profile_t0_event,
        )

    def submit_layer(self, layer_id: int) -> None:
        if self._finished:
            raise RuntimeError("CSKCache layer stream is already finished")
        if self._pending_layer is not None:
            raise RuntimeError("CSKCache already has a pending H2D layer")
        if (
            layer_id != self._next_submit_layer
            or layer_id >= len(self._buffers)
        ):
            raise ValueError(
                "CSKCache layers must be submitted exactly once in order"
            )
        self._copy_session.submit(self._transfer_objects[layer_id])
        self._pending_layer = layer_id
        self._next_submit_layer += 1

    def wait_layer(self, layer_id: int) -> None:
        if self._finished:
            raise RuntimeError("CSKCache layer stream is already finished")
        if layer_id != self._pending_layer or layer_id != self._next_wait_layer:
            raise ValueError("CSKCache must wait for the pending layer in order")
        self._copy_session.wait()
        self._pending_layer = None
        self._next_wait_layer += 1

    def staged_key(self, layer_id: int) -> torch.Tensor:
        if not 0 <= layer_id < self._next_wait_layer:
            raise ValueError("CSKCache layer has not been staged")
        key, _value = self._gpu_connector.get_kv(layer_id)
        return key

    def commit_calibration(
        self,
        layer_id: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        self._gpu_connector.commit_recomputed_kv(
            layer_id,
            key,
            value,
            self._plan.calibration_start,
            self._plan.calibration_end,
        )

    def commit_layer(self, layer_id: int) -> None:
        if not 0 <= layer_id < self._next_wait_layer:
            raise ValueError("CSKCache layer has not been staged")
        self._gpu_connector.commit_staged_kv(layer_id)

    def finish(self) -> None:
        if self._finished:
            raise RuntimeError("CSKCache layer stream was finished twice")
        if self._pending_layer is not None:
            raise RuntimeError("CSKCache cannot finish with a pending H2D layer")
        if (
            self._next_submit_layer != len(self._buffers)
            or self._next_wait_layer != len(self._buffers)
        ):
            raise RuntimeError("CSKCache cannot finish an incomplete layer stream")
        self._copy_session.finish()
        self._finished = True
        self._emit_complete_profile()

    def abort(self) -> None:
        """Drain the private load stream before the caller invalidates slots."""

        if self._finished:
            return
        torch.cuda.synchronize()
        self._copy_session.close()
        self._finished = True

    def _emit_complete_profile(self) -> None:
        profile_event(
            "csk_worker_load_complete",
            self._plan.request_id,
            ticket=self._plan.ticket,
            layers=len(self._buffers),
            reuse_start=self._plan.reuse_start,
            reuse_end=self._plan.reuse_end,
            loaded_tokens=self._plan.reuse_end - self._plan.reuse_start,
            source="t0_pinned_buffers",
        )


class LMCacheWorkerIntegration:
    """Own CSKCache's worker execution over generic LMCache mechanisms."""

    def __init__(
        self,
        runtime: Any,
        gpu_connector: Any,
        engine_name: str,
        *,
        execution_order: str,
    ) -> None:
        self._data_plane = LMCacheCSKDataPlane(
            runtime,
            gpu_connector,
            engine_name,
        )
        self._executor = CSKCacheReuseExecutor(
            self._data_plane,
            expected_layers=self._data_plane.num_layers,
            execution_order=execution_order,
        )

    @property
    def layerwise_model(self) -> Any:
        return self._data_plane.layerwise_model

    def execute(
        self,
        plan: ReusePlan,
        *,
        token_ids: Sequence[int],
        kvcaches: Sequence[torch.Tensor],
        slot_mapping: torch.Tensor,
    ) -> Any:
        result = self._executor.execute(
            plan,
            token_ids=token_ids,
            kvcaches=kvcaches,
            slot_mapping=slot_mapping[: plan.reuse_end],
        )
        profile_event(
            "csk_correction_complete",
            plan.request_id,
            ticket=result.ticket,
            layers=result.processed_layers,
            correction_alpha=result.correction_alpha,
            calibration_tokens=(
                plan.calibration_end - plan.calibration_start
            ),
            correction="auxiliary_forward_then_key_headwise",
        )
        return result
