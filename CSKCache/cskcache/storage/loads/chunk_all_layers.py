"""SSD load grouping for one all-layer region per chunk."""

from __future__ import annotations

from ...layouts import KVLayout, KVLayoutPlan
from .base import StorageLoadGroup, StorageLoadPlan


def build_load_plan(layout_plan: KVLayoutPlan) -> StorageLoadPlan:
    if layout_plan.layout is not KVLayout.CHUNK_ALL_LAYERS:
        raise ValueError("chunk_all_layers loader received another layout")
    groups = tuple(
        StorageLoadGroup(group_id=index, region_ids=(region.region_id,))
        for index, region in enumerate(layout_plan.regions)
    )
    return StorageLoadPlan(layout_plan.layout, layout_plan, groups)
