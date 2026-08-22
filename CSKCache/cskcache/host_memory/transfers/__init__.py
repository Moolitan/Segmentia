"""Pinned-to-GPU transfer contracts and the shared layerwise planner."""

from __future__ import annotations

from .base import (
    BoundH2DTransferPlan,
    H2DCopyStrategy,
    H2DCopySession,
    H2DRegionSlice,
    H2DTransferPlan,
    H2DTransferStep,
)
from .per_object_copy import PerObjectCopySession
from .planner import bind_layer_buffers, build_h2d_transfer_plan


__all__ = [
    "H2DRegionSlice",
    "BoundH2DTransferPlan",
    "H2DCopyStrategy",
    "H2DCopySession",
    "H2DTransferPlan",
    "H2DTransferStep",
    "PerObjectCopySession",
    "build_h2d_transfer_plan",
    "bind_layer_buffers",
]
