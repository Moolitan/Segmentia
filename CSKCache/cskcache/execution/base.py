"""Public contracts for layerwise Skill KV materialization.

This module defines the execution vocabulary only.  It does not perform model
forward, H2D transfer, correction, synchronization, or PagedKV writes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

import torch

from ..runtime.base import ReusePlan


class ExecutionOrder(str, Enum):
    """Ordering of next-layer H2D submission and current-layer computation."""

    H2D_FIRST = "h2d_first"
    COMPUTE_FIRST = "compute_first"


class LayerwiseCalibrationModel(Protocol):
    """Auxiliary Transformer forward that yields fresh calibration KV."""

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]: ...

    def close(self) -> None: ...


class LayerwiseReuseStream(Protocol):
    """Request-local physical stream supplied by the serving integration."""

    def submit_layer(self, layer_id: int) -> None: ...

    def wait_layer(self, layer_id: int) -> None: ...

    def staged_key(self, layer_id: int) -> torch.Tensor: ...

    def commit_calibration(
        self,
        layer_id: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None: ...

    def commit_layer(self, layer_id: int) -> None: ...

    def finish(self) -> None: ...

    def abort(self) -> None: ...


class ReuseDataPlane(Protocol):
    """Model and physical interfaces required by the execution layer."""

    def get_active_layer_buffers(
        self,
        ticket: str,
        request_id: str,
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

    def open_calibration_model(
        self,
        plan: ReusePlan,
        token_ids: Sequence[int],
    ) -> LayerwiseCalibrationModel: ...

    def mark_layer_loaded(
        self,
        ticket: str,
        request_id: str,
        layer_id: int,
    ) -> None: ...

    def mark_layer_corrected(
        self,
        ticket: str,
        request_id: str,
        layer_id: int,
    ) -> None: ...


@dataclass(frozen=True)
class ReuseExecutionResult:
    """Evidence returned only after the complete layer group is committed."""

    ticket: str
    request_id: str
    processed_layers: int
    correction_alpha: float


@dataclass(frozen=True)
class LayerComputeEvents:
    """CUDA event boundaries for one layer's compute/correct/install path."""

    start: torch.cuda.Event
    calibration_forward_end: torch.cuda.Event
    calibration_commit_end: torch.cuda.Event
    residual_correction_end: torch.cuda.Event
    end: torch.cuda.Event


LayerCompute = Callable[
    [
        int,
        ReusePlan,
        LayerwiseReuseStream,
        LayerwiseCalibrationModel,
        torch.cuda.Event | None,
    ],
    LayerComputeEvents | None,
]
