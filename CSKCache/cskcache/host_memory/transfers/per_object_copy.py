"""Current H2D primitive: enqueue one K/V copy pair per source object."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .base import H2DCopyStrategy


class PerObjectCopySession:
    """Drive LMCache's layerwise generator with per-object source lists.

    ``batched_to_gpu`` batches objects at its Python API boundary, but its
    current implementation iterates over the objects and enqueues one Key copy
    and one Value copy for each object.  This session names that behavior
    explicitly without duplicating LMCache's CUDA implementation.
    """

    strategy = H2DCopyStrategy.PER_OBJECT_COPY

    def __init__(
        self,
        gpu_connector: Any,
        starts: Sequence[int],
        ends: Sequence[int],
        **kwargs: Any,
    ) -> None:
        self._consumer = gpu_connector.batched_to_gpu(
            list(starts),
            list(ends),
            **kwargs,
        )
        next(self._consumer)

    def submit(self, memory_objects: Sequence[Any]) -> None:
        self._consumer.send(list(memory_objects))

    def wait(self) -> None:
        next(self._consumer)

    def finish(self) -> None:
        next(self._consumer)

    def close(self) -> None:
        self._consumer.close()
