"""One physical region per chunk, containing every model layer."""

from __future__ import annotations

from ..chunking import SkillChunkPlan
from .base import KVLayout, KVLayoutPlan, KVRegion


def build_layout(chunk_plan: SkillChunkPlan, num_layers: int) -> KVLayoutPlan:
    regions = tuple(
        KVRegion(
            region_id=chunk.chunk_id,
            chunk_start=chunk.chunk_id,
            chunk_end=chunk.chunk_id + 1,
            token_start=chunk.token_start,
            token_end=chunk.token_end,
            layer_start=0,
            layer_end=num_layers,
        )
        for chunk in chunk_plan.chunks
    )
    return KVLayoutPlan(KVLayout.CHUNK_ALL_LAYERS, chunk_plan, num_layers, regions)
