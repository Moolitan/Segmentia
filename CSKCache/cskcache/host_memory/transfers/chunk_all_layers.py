"""Layerwise H2D views over all-layer chunk buffers."""

from ...layouts import KVLayout, KVLayoutPlan
from .base import H2DTransferPlan, build_layerwise_transfer_plan


def build_transfer_plan(layout_plan: KVLayoutPlan) -> H2DTransferPlan:
    if layout_plan.layout is not KVLayout.CHUNK_ALL_LAYERS:
        raise ValueError("chunk_all_layers transfer received another layout")
    return build_layerwise_transfer_plan(layout_plan)
