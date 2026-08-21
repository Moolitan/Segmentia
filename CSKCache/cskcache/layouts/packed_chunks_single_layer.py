"""One physical region per layer, packing every Skill chunk."""

from __future__ import annotations

from ..chunking import SkillChunkPlan
from .base import KVLayout, KVLayoutPlan, KVRegion


def build_layout(chunk_plan: SkillChunkPlan, num_layers: int) -> KVLayoutPlan:
    regions = tuple(
        KVRegion(
            region_id=layer_id,
            chunk_start=0,
            chunk_end=chunk_plan.chunk_count,
            token_start=0,
            token_end=chunk_plan.skill_token_count,
            layer_start=layer_id,
            layer_end=layer_id + 1,
        )
        for layer_id in range(num_layers)
    )
    return KVLayoutPlan(
        KVLayout.PACKED_CHUNKS_SINGLE_LAYER,
        chunk_plan,
        num_layers,
        regions,
    )
