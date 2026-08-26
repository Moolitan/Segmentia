"""Public contracts for layerwise Skill KV materialization.

This module defines the execution vocabulary only.  It does not perform model
forward, H2D transfer, correction, synchronization, or PagedKV writes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Protocol

import torch

from ..runtime.base import CorrectionStrategy, ReusePlan


class ExecutionOrder(str, Enum):
    """Ordering of next-layer H2D submission and current-layer computation."""

    H2D_FIRST = "h2d_first"
    COMPUTE_FIRST = "compute_first"


@dataclass(frozen=True)
class ExecutionMethod:
    """One named TTFT path and its execution requirements.

    ``normal_prefill`` is represented here so benchmarks and result records use
    the same vocabulary as cache-reuse paths.  It never appears in a
    :class:`ReusePlan`, because normal prefill does not activate CSKCache.
    """

    name: str
    correction_strategy: CorrectionStrategy | None
    reuses_cache: bool
    uses_auxiliary_model: bool


class NormalPrefillMethod(ExecutionMethod):
    """Compute the complete prompt without activating cached Skill KV."""

    def __init__(self) -> None:
        super().__init__("normal_prefill", None, False, False)


class DirectReuseMethod(ExecutionMethod):
    """Install authenticated cached KV without online correction."""

    def __init__(self) -> None:
        super().__init__("direct_reuse", CorrectionStrategy.DIRECT, True, False)


class CalibrationResidualCorrectionMethod(ExecutionMethod):
    """Recompute a contiguous calibration prefix and compensate cached keys."""

    SUPPORTED_STRATEGIES: ClassVar[tuple[CorrectionStrategy, ...]] = (
        CorrectionStrategy.FIXED_PREFIX,
        CorrectionStrategy.RATIO_PREFIX,
    )

    def __init__(self, strategy: CorrectionStrategy) -> None:
        if strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError(
                f"calibration residual method does not support {strategy.value}"
            )
        super().__init__("calibration_residual", strategy, True, True)


class DeviationTopKRecomputeMethod(ExecutionMethod):
    """Select high-deviation tokens once, then recompute them layerwise."""

    def __init__(self) -> None:
        super().__init__(
            "deviation_topk",
            CorrectionStrategy.DEVIATION_TOPK,
            True,
            True,
        )


NORMAL_PREFILL_METHOD = NormalPrefillMethod()
DIRECT_REUSE_METHOD = DirectReuseMethod()
FIXED_PREFIX_RESIDUAL_METHOD = CalibrationResidualCorrectionMethod(
    CorrectionStrategy.FIXED_PREFIX
)
RATIO_PREFIX_RESIDUAL_METHOD = CalibrationResidualCorrectionMethod(
    CorrectionStrategy.RATIO_PREFIX
)
DEVIATION_TOPK_METHOD = DeviationTopKRecomputeMethod()


def execution_method_for(
    strategy: CorrectionStrategy | str,
) -> ExecutionMethod:
    """Resolve a runtime correction strategy to its concrete method class."""

    parsed = CorrectionStrategy(strategy)
    methods = {
        CorrectionStrategy.DIRECT: DIRECT_REUSE_METHOD,
        CorrectionStrategy.FIXED_PREFIX: FIXED_PREFIX_RESIDUAL_METHOD,
        CorrectionStrategy.RATIO_PREFIX: RATIO_PREFIX_RESIDUAL_METHOD,
        CorrectionStrategy.DEVIATION_TOPK: DEVIATION_TOPK_METHOD,
    }
    return methods[parsed]


class LayerwiseCalibrationModel(Protocol):
    """Auxiliary Transformer forward that yields fresh calibration KV."""

    def __next__(self) -> tuple[torch.Tensor, torch.Tensor]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class DeviationTopKLayerResult:
    """Per-layer evidence emitted by deviation-selective recomputation."""

    layer_id: int
    candidate_tokens: int
    recomputed_tokens: int
    selection_applied: bool


class LayerwiseDeviationTopKModel(Protocol):
    """Auxiliary forward that updates staged KV using deviation top-k."""

    def __next__(self) -> DeviationTopKLayerResult: ...

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

    def open_deviation_topk_model(
        self,
        plan: ReusePlan,
        token_ids: Sequence[int],
    ) -> LayerwiseDeviationTopKModel: ...

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
    correction_strategy: CorrectionStrategy = CorrectionStrategy.FIXED_PREFIX
    method: ExecutionMethod = FIXED_PREFIX_RESIDUAL_METHOD


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
