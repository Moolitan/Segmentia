from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


def _enabled(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ProfileConfig:
    """Configuration kept separate from the cache mechanism configuration."""

    enabled: bool = False
    jsonl_path: Path | None = None

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "ProfileConfig":
        env = os.environ if environ is None else environ
        raw_path = env.get("CSKCACHE_PROFILE_JSONL")
        return cls(
            enabled=_enabled(env.get("CSKCACHE_PROFILE_ENABLED")),
            jsonl_path=Path(raw_path) if raw_path else None,
        )
