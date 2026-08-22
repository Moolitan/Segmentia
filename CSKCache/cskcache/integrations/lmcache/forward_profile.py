"""Optional CUDA-stage profiler for CSKCache's auxiliary forward."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from ...profile import profile_event
from ...runtime.base import ReusePlan


class CalibrationForwardProfiler:
    """Time selected auxiliary-forward layers without instrumenting all layers."""

    _MODULE_STAGES = (
        ("input_norm", "input_layernorm"),
        ("qkv_projection", "self_attn.qkv_proj"),
        ("q_norm", "self_attn.q_norm"),
        ("k_norm", "self_attn.k_norm"),
        ("output_projection", "self_attn.o_proj"),
        ("post_attention_norm", "post_attention_layernorm"),
        ("mlp", "mlp"),
    )

    def __init__(self, model: Any, layer_ids: Sequence[int]) -> None:
        self._layer_ids = frozenset(int(layer_id) for layer_id in layer_ids)
        self._events: dict[int, dict[str, list[torch.cuda.Event]]] = {}
        self._active_request_id: str | None = None
        self._plan: ReusePlan | None = None
        self._handles = []
        for layer_id in sorted(self._layer_ids):
            layer = model.vllm_model.model.layers[layer_id]
            for stage, path in self._MODULE_STAGES:
                module = layer
                for component in path.split("."):
                    module = getattr(module, component)
                self._handles.append(
                    module.register_forward_pre_hook(
                        self._module_hook(layer_id, stage, 0)
                    )
                )
                self._handles.append(
                    module.register_forward_hook(
                        self._module_hook(layer_id, stage, 1)
                    )
                )

    def _module_hook(self, layer_id: int, stage: str, index: int):
        def hook(_module: Any, _inputs: Any, _output: Any = None) -> None:
            if self._active_request_id is None:
                return
            pair = self._events.setdefault(layer_id, {}).setdefault(stage, [])
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            if index == 0:
                pair[:] = [event]
            else:
                pair.append(event)

        return hook

    def begin(self, plan: ReusePlan) -> None:
        self._events.clear()
        self._active_request_id = plan.request_id
        self._plan = plan

    def start(self, layer_id: int, stage: str) -> None:
        if self._active_request_id is None or layer_id not in self._layer_ids:
            return
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self._events.setdefault(layer_id, {})[stage] = [event]

    def end(self, layer_id: int, stage: str) -> None:
        if self._active_request_id is None or layer_id not in self._layer_ids:
            return
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        self._events[layer_id][stage].append(event)

    def finish(self) -> None:
        request_id = self._active_request_id
        plan = self._plan
        if request_id is None or plan is None:
            return
        torch.cuda.synchronize()
        rows = []
        for layer_id in sorted(self._events):
            stages = self._events[layer_id]

            def elapsed(stage: str) -> float:
                start, end = stages[stage]
                return float(start.elapsed_time(end))

            row = {
                "layer": layer_id,
                "input_norm_ms": elapsed("input_norm"),
                "qkv_projection_ms": elapsed("qkv_projection"),
                "q_norm_ms": elapsed("q_norm"),
                "k_norm_ms": elapsed("k_norm"),
                "position_build_ms": elapsed("position_build"),
                "rope_ms": elapsed("rope"),
                "prefix_paged_kv_ms": elapsed("prefix_paged_kv"),
                "kv_concat_ms": elapsed("kv_concat"),
                "attention_ms": float(
                    stages["attention"][0].elapsed_time(
                        stages["output_projection"][0]
                    )
                ),
                "output_projection_ms": elapsed("output_projection"),
                "post_attention_norm_ms": elapsed("post_attention_norm"),
                "mlp_ms": elapsed("mlp"),
                "module_span_ms": float(
                    stages["input_norm"][0].elapsed_time(stages["mlp"][1])
                ),
            }
            rows.append(row)
        profile_event(
            "cskcache_calibration_forward_breakdown",
            request_id,
            calibration_tokens=plan.calibration_end - plan.calibration_start,
            prefix_tokens=plan.calibration_start,
            layers=rows,
        )
        self._active_request_id = None
        self._plan = None
        self._events.clear()
