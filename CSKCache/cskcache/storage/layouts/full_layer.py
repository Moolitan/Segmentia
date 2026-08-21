"""Full-layer pinned-buffer allocation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from lmcache.v1.memory_management import MemoryFormat


def acquire_full_layers(
    backend: Any,
    *,
    shape: torch.Size,
    dtype: torch.dtype,
    memory_format: MemoryFormat,
    layer_count: int,
) -> tuple[Any, ...]:
    """Allocate one complete pinned tensor for every model layer."""

    objects = backend.batched_allocate(
        shape,
        dtype,
        layer_count,
        fmt=memory_format,
        eviction=True,
        busy_loop=False,
    )
    if objects is None or len(objects) != layer_count:
        _release_memory_objects(objects or ())
        raise MemoryError(
            "LMCache pinned CPU pool cannot allocate a full layer group"
        )
    return tuple(objects)


def _release_memory_objects(memory_objects: Sequence[Any]) -> None:
    for memory_obj in memory_objects:
        memory_obj.ref_count_down()
