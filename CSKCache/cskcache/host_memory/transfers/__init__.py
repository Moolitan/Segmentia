"""Layout-specific Pinned-to-GPU transfer plans."""

from __future__ import annotations

from ...layouts import KVLayout, KVLayoutPlan
from .base import (
    BoundH2DTransferPlan,
    H2DRegionSlice,
    H2DTransferPlan,
    H2DTransferStep,
    bind_layer_buffers,
)


def build_h2d_transfer_plan(layout_plan: KVLayoutPlan) -> H2DTransferPlan:
    if layout_plan.layout is KVLayout.CHUNK_ALL_LAYERS:
        from .chunk_all_layers import build_transfer_plan
    elif layout_plan.layout is KVLayout.CHUNK_SINGLE_LAYER:
        from .chunk_single_layer import build_transfer_plan
    elif layout_plan.layout is KVLayout.PACKED_CHUNKS_SINGLE_LAYER:
        from .packed_chunks_single_layer import build_transfer_plan
    else:
        from .packed_chunks_all_layers import build_transfer_plan
    return build_transfer_plan(layout_plan)


__all__ = [
    "H2DRegionSlice",
    "BoundH2DTransferPlan",
    "H2DTransferPlan",
    "H2DTransferStep",
    "build_h2d_transfer_plan",
    "bind_layer_buffers",
]
