"""Chunk-major pinned-buffer allocation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch

from lmcache.v1.memory_management import MemoryFormat

from .base import ChunkedLayerBuffer, LayerChunk


def acquire_chunked_layers(
    backend: Any,
    *,
    layer_count: int,
    token_count: int,
    hidden_size: int,
    chunk_tokens: int,
    dtype: torch.dtype,
    memory_format: MemoryFormat,
    validate_memory_object: Callable[[Any, Sequence[int], torch.dtype], None],
) -> tuple[ChunkedLayerBuffer, ...]:
    """Allocate token chunks and regroup them into complete logical layers."""

    layer_chunks: list[list[LayerChunk]] = [[] for _ in range(layer_count)]
    allocated: list[Any] = []
    try:
        for token_start in range(0, token_count, chunk_tokens):
            token_end = min(token_start + chunk_tokens, token_count)
            chunk_shape = torch.Size((2, token_end - token_start, hidden_size))
            chunk_objects = backend.batched_allocate(
                chunk_shape,
                dtype,
                layer_count,
                fmt=memory_format,
                eviction=True,
                busy_loop=False,
            )
            if chunk_objects is None or len(chunk_objects) != layer_count:
                if chunk_objects:
                    allocated.extend(chunk_objects)
                raise MemoryError(
                    "LMCache pinned CPU pool cannot allocate a chunk layer group"
                )
            allocated.extend(chunk_objects)
            for layer_id, memory_obj in enumerate(chunk_objects):
                validate_memory_object(memory_obj, chunk_shape, dtype)
                layer_chunks[layer_id].append(
                    LayerChunk(
                        token_start=token_start,
                        token_end=token_end,
                        memory_obj=memory_obj,
                    )
                )
    except Exception:
        _release_memory_objects(allocated)
        raise

    return tuple(
        ChunkedLayerBuffer(tuple(chunks)) for chunks in layer_chunks
    )


def _release_memory_objects(memory_objects: Sequence[Any]) -> None:
    for memory_obj in memory_objects:
        memory_obj.ref_count_down()
