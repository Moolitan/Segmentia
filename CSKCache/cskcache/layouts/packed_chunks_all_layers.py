"""One physical region packing every Skill chunk and model layer."""

from __future__ import annotations

from ..chunking import SkillChunkPlan
from .base import KVLayout, KVLayoutPlan, KVRegion


def build_layout(chunk_plan: SkillChunkPlan, num_layers: int) -> KVLayoutPlan:
    return KVLayoutPlan(
        KVLayout.PACKED_CHUNKS_ALL_LAYERS,
        chunk_plan,
        num_layers,
        (
            KVRegion(
                region_id=0,
                chunk_start=0,
                chunk_end=chunk_plan.chunk_count,
                token_start=0,
                token_end=chunk_plan.skill_token_count,
                layer_start=0,
                layer_end=num_layers,
            ),
        ),
    )
