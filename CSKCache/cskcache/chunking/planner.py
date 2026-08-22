"""Single entry point for Skill chunk planning."""

from __future__ import annotations

from .base import ChunkingSpec, SkillChunkPlan
from .fixed_size import build_fixed_size_plan


def build_chunk_plan(
    skill_token_count: int,
    spec: ChunkingSpec,
) -> SkillChunkPlan:
    """Build the canonical partition consumed by every data-path stage."""

    return build_fixed_size_plan(skill_token_count, spec.chunk_size_tokens)
