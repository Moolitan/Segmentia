"""SSD load grouping for separate chunk-layer regions."""

from __future__ import annotations

from ...layouts import KVLayout, KVLayoutPlan
from .base import StorageLoadGroup, StorageLoadPlan


def build_load_plan(layout_plan: KVLayoutPlan) -> StorageLoadPlan:
    if layout_plan.layout is not KVLayout.CHUNK_SINGLE_LAYER:
        raise ValueError("chunk_single_layer loader received another layout")
    groups: list[StorageLoadGroup] = []
    for chunk_id in range(layout_plan.chunk_plan.chunk_count):
        region_ids = tuple(
            region.region_id
            for region in layout_plan.regions
            if region.chunk_start == chunk_id
            and region.chunk_end == chunk_id + 1
        )
        groups.append(StorageLoadGroup(len(groups), region_ids))
    return StorageLoadPlan(layout_plan.layout, layout_plan, tuple(groups))
