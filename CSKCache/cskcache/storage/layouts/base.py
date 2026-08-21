"""Public data contracts for pinned-host Skill KV layouts.

Layouts describe how one logical model layer is represented after SSD load and
before H2D.  They do not change the semantic Skill object or its persistent
per-layer metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class HostLayout(str, Enum):
    """Supported pinned-host representations of one logical model layer."""

    FULL_LAYER = "full_layer"
    CHUNK_MAJOR = "chunk_major"


@dataclass(frozen=True)
class LayerChunk:
    """One token-contiguous pinned object inside a logical model layer."""

    token_start: int
    token_end: int
    memory_obj: Any


@dataclass(frozen=True)
class ChunkedLayerBuffer:
    """Token chunks that jointly cover one complete logical model layer."""

    chunks: tuple[LayerChunk, ...]
