"""Fixed-token Skill chunking."""

from __future__ import annotations

from .base import ChunkingMode, ChunkingSpec, ChunkSpan, SkillChunkPlan


def build_fixed_size_plan(
    skill_token_count: int,
    chunk_size_tokens: int,
) -> SkillChunkPlan:
    """Partition a Skill into fixed chunks and retain the exact tail."""

    spec = ChunkingSpec(
        mode=ChunkingMode.FIXED_SIZE,
        chunk_size_tokens=chunk_size_tokens,
    )
    if skill_token_count <= 0:
        raise ValueError("skill_token_count must be positive")
    chunks = tuple(
        ChunkSpan(
            chunk_id=chunk_id,
            token_start=token_start,
            token_end=min(token_start + chunk_size_tokens, skill_token_count),
        )
        for chunk_id, token_start in enumerate(
            range(0, skill_token_count, chunk_size_tokens)
        )
    )
    return SkillChunkPlan(skill_token_count, spec, chunks)
