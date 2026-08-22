"""Token-partition contracts shared by storage, host memory, and H2D."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ChunkingSpec:
    """One positive token budget from which chunk count is derived."""

    chunk_size_tokens: int

    def __post_init__(self) -> None:
        if self.chunk_size_tokens <= 0:
            raise ValueError("chunk_size_tokens must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {"chunk_size_tokens": self.chunk_size_tokens}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChunkingSpec":
        if set(payload) != {"chunk_size_tokens"}:
            raise ValueError("ChunkingSpec fields differ from the canonical schema")
        return cls(chunk_size_tokens=int(payload["chunk_size_tokens"]))


@dataclass(frozen=True)
class ChunkSpan:
    """One Skill-relative, half-open token interval."""

    chunk_id: int
    token_start: int
    token_end: int

    def __post_init__(self) -> None:
        if self.chunk_id < 0:
            raise ValueError("chunk_id must be non-negative")
        if not 0 <= self.token_start < self.token_end:
            raise ValueError("chunk token interval must be non-empty")

    @property
    def token_count(self) -> int:
        return self.token_end - self.token_start


@dataclass(frozen=True)
class SkillChunkPlan:
    """The single authoritative chunk partition of one Skill."""

    skill_token_count: int
    spec: ChunkingSpec
    chunks: tuple[ChunkSpan, ...]

    def __post_init__(self) -> None:
        if self.skill_token_count <= 0:
            raise ValueError("skill_token_count must be positive")
        if not self.chunks:
            raise ValueError("a Skill chunk plan must contain at least one chunk")
        cursor = 0
        for expected_id, chunk in enumerate(self.chunks):
            if chunk.chunk_id != expected_id:
                raise ValueError("chunk IDs must be dense and ordered")
            if chunk.token_start != cursor:
                raise ValueError("chunks must cover the Skill without gaps")
            cursor = chunk.token_end
        if cursor != self.skill_token_count:
            raise ValueError("chunks must cover the complete Skill token span")

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def effective_chunk_size_tokens(self) -> int:
        return min(self.spec.chunk_size_tokens, self.skill_token_count)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_token_count": self.skill_token_count,
            "spec": self.spec.to_dict(),
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "token_start": chunk.token_start,
                    "token_end": chunk.token_end,
                }
                for chunk in self.chunks
            ],
        }


class ChunkPlanner(Protocol):
    """Build one exact Skill-relative token partition."""

    def build(self, skill_token_count: int) -> SkillChunkPlan: ...
