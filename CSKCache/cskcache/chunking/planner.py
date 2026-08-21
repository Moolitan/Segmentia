"""Single entry point for Skill chunk planning."""

from __future__ import annotations

from .base import ChunkingMode, ChunkingSpec, SkillChunkPlan
from .fixed_size import build_fixed_size_plan
from .whole_skill import build_whole_skill_plan


def build_chunk_plan(
    skill_token_count: int,
    spec: ChunkingSpec,
) -> SkillChunkPlan:
    """Build the canonical partition consumed by every data-path stage."""

    if spec.mode is ChunkingMode.WHOLE_SKILL:
        return build_whole_skill_plan(skill_token_count)
    assert spec.chunk_size_tokens is not None
    return build_fixed_size_plan(skill_token_count, spec.chunk_size_tokens)
