"""Whole-Skill chunking."""

from __future__ import annotations

from .base import ChunkingMode, ChunkingSpec, ChunkSpan, SkillChunkPlan


def build_whole_skill_plan(skill_token_count: int) -> SkillChunkPlan:
    """Represent one complete Skill as exactly one logical chunk."""

    if skill_token_count <= 0:
        raise ValueError("skill_token_count must be positive")
    return SkillChunkPlan(
        skill_token_count=skill_token_count,
        spec=ChunkingSpec(mode=ChunkingMode.WHOLE_SKILL),
        chunks=(ChunkSpan(0, 0, skill_token_count),),
    )
