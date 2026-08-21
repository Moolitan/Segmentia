"""H2D-first layer scheduling for Skill KV materialization."""

from __future__ import annotations

import torch

from ..runtime.base import ReusePlan
from .base import LayerCompute, LayerwiseCalibrationModel, LayerwiseReuseStream


def execute_h2d_first(
    *,
    expected_layers: int,
    plan: ReusePlan,
    stream: LayerwiseReuseStream,
    calibration_model: LayerwiseCalibrationModel,
    compute_layer: LayerCompute,
    profile_t0_event: torch.cuda.Event | None,
) -> list[tuple[int, torch.cuda.Event, torch.cuda.Event]]:
    """Submit H2D(l+1) before C/R/I(l), then join both streams."""

    compute_events: list[tuple[int, torch.cuda.Event, torch.cuda.Event]] = []
    stream.submit_layer(0)
    stream.wait_layer(0)
    for layer_id in range(expected_layers):
        next_layer = layer_id + 1
        if next_layer < expected_layers:
            stream.submit_layer(next_layer)
        compute_event = compute_layer(
            layer_id,
            plan,
            stream,
            calibration_model,
            profile_t0_event,
        )
        if compute_event is not None:
            compute_events.append((layer_id, *compute_event))
        if next_layer < expected_layers:
            stream.wait_layer(next_layer)
    return compute_events
