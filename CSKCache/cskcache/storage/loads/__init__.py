"""Layout-specific SSD load grouping."""

from __future__ import annotations

from ...layouts import KVLayout, KVLayoutPlan
from .base import StorageLoadGroup, StorageLoadPlan


def build_storage_load_plan(layout_plan: KVLayoutPlan) -> StorageLoadPlan:
    if layout_plan.layout is KVLayout.CHUNK_ALL_LAYERS:
        from .chunk_all_layers import build_load_plan
    elif layout_plan.layout is KVLayout.CHUNK_SINGLE_LAYER:
        from .chunk_single_layer import build_load_plan
    elif layout_plan.layout is KVLayout.PACKED_CHUNKS_SINGLE_LAYER:
        from .packed_chunks_single_layer import build_load_plan
    else:
        from .packed_chunks_all_layers import build_load_plan
    return build_load_plan(layout_plan)


__all__ = [
    "StorageLoadGroup",
    "StorageLoadPlan",
    "build_storage_load_plan",
]
