"""Shared 2x2 chunk/layer layouts."""

from __future__ import annotations

from .base import KVLayout, KVLayoutPlan, KVRegion


def build_layout_plan(
    layout: KVLayout | str,
    chunk_plan,
    num_layers: int,
) -> KVLayoutPlan:
    parsed = KVLayout(layout)
    if parsed is KVLayout.CHUNK_ALL_LAYERS:
        from .chunk_all_layers import build_layout
    elif parsed is KVLayout.CHUNK_SINGLE_LAYER:
        from .chunk_single_layer import build_layout
    elif parsed is KVLayout.PACKED_CHUNKS_SINGLE_LAYER:
        from .packed_chunks_single_layer import build_layout
    else:
        from .packed_chunks_all_layers import build_layout
    return build_layout(chunk_plan, num_layers)


__all__ = ["KVLayout", "KVLayoutPlan", "KVRegion", "build_layout_plan"]
