"""SSD load grouping for one fully packed Skill region."""

from __future__ import annotations

from ...layouts import KVLayout, KVLayoutPlan
from .base import StorageLoadGroup, StorageLoadPlan


def build_load_plan(layout_plan: KVLayoutPlan) -> StorageLoadPlan:
    if layout_plan.layout is not KVLayout.PACKED_CHUNKS_ALL_LAYERS:
        raise ValueError("packed_chunks_all_layers loader received another layout")
    return StorageLoadPlan(
        layout_plan.layout,
        layout_plan,
        (StorageLoadGroup(0, (0,)),),
    )
