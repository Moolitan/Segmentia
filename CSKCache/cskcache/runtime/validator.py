"""Central CSKCache validation performed before a request enters the data path."""

from __future__ import annotations

from collections.abc import Sequence

from ..chunking import ChunkingMode
from ..layouts import KVLayout
from ..metadata.base import CacheObjectMetadata


def validate_catalog_layout(
    objects: Sequence[CacheObjectMetadata],
    *,
    chunking_mode: str,
    chunk_size_tokens: int | None,
    storage_layout: str,
) -> None:
    """Verify the runtime configuration against every active offline object."""

    expected_mode = ChunkingMode(chunking_mode)
    expected_layout = KVLayout(storage_layout)
    for cache_object in objects:
        if cache_object.chunking.mode is not expected_mode:
            raise ValueError(
                f"Skill {cache_object.skill_name} uses another chunking mode"
            )
        if cache_object.chunking.chunk_size_tokens != chunk_size_tokens:
            raise ValueError(
                f"Skill {cache_object.skill_name} uses another chunk size"
            )
        if cache_object.storage_layout is not expected_layout:
            raise ValueError(
                f"Skill {cache_object.skill_name} uses another storage layout"
            )
