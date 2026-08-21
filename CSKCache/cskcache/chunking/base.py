"""Token-partition contracts shared by storage, host memory, and H2D."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol


class ChunkingMode(str, Enum):
    """How one logical Skill token span is partitioned."""

    FIXED_SIZE = "fixed_size"
    WHOLE_SKILL = "whole_skill"


@dataclass(frozen=True)
class ChunkingSpec:
    """Offline chunking choice; chunk count is always derived."""

    mode: ChunkingMode
    chunk_size_tokens: int | None = None

    def __post_init__(self) -> None:
        mode = ChunkingMode(self.mode)
        object.__setattr__(self, "mode", mode)
        if mode is ChunkingMode.FIXED_SIZE:
            if self.chunk_size_tokens is None or self.chunk_size_tokens <= 0:
                raise ValueError("fixed_size chunking requires a positive chunk size")
        elif self.chunk_size_tokens is not None:
            raise ValueError("whole_skill chunking derives its size from the Skill")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode.value,
            "chunk_size_tokens": self.chunk_size_tokens,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ChunkingSpec":
        if set(payload) != {"mode", "chunk_size_tokens"}:
            raise ValueError("ChunkingSpec fields differ from the canonical schema")
        size = payload["chunk_size_tokens"]
        return cls(
            mode=ChunkingMode(str(payload["mode"])),
            chunk_size_tokens=None if size is None else int(size),
        )


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
        if self.spec.mode is ChunkingMode.WHOLE_SKILL:
            return self.skill_token_count
        assert self.spec.chunk_size_tokens is not None
        return self.spec.chunk_size_tokens

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
