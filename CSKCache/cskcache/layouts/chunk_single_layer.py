"""One physical region per chunk and model layer."""

from __future__ import annotations

from ..chunking import SkillChunkPlan
from .base import KVLayout, KVLayoutPlan, KVRegion


def build_layout(chunk_plan: SkillChunkPlan, num_layers: int) -> KVLayoutPlan:
    regions: list[KVRegion] = []
    for chunk in chunk_plan.chunks:
        for layer_id in range(num_layers):
            regions.append(
                KVRegion(
                    region_id=len(regions),
                    chunk_start=chunk.chunk_id,
                    chunk_end=chunk.chunk_id + 1,
                    token_start=chunk.token_start,
                    token_end=chunk.token_end,
                    layer_start=layer_id,
                    layer_end=layer_id + 1,
                )
            )
    return KVLayoutPlan(
        KVLayout.CHUNK_SINGLE_LAYER,
        chunk_plan,
        num_layers,
        tuple(regions),
    )
