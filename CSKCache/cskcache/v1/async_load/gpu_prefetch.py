from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch


@dataclass
class GpuPrefetchHandle:
    """A callable's result, computed on a non-default CUDA stream.

    Mirrors ``disk_prefetch.PrefetchHandle``'s submit -> handle -> result()
    shape, but for GPU work: CUDA already has its own concurrency primitive
    (streams + events), so this wraps that instead of a thread pool.

    ``result()`` makes the *current* stream wait on the prefetch's
    completion event before returning the value. If the prefetch already
    finished by the time you call this, the wait is free; if not, the
    caller blocks no longer than a synchronous call would have -- the only
    difference is whatever other work the current stream did between
    ``submit_gpu_prefetch`` and ``result()`` got to run concurrently with
    the prefetch instead of waiting for it up front.
    """

    _value: Any
    _event: torch.cuda.Event

    def result(self) -> Any:
        torch.cuda.current_stream().wait_event(self._event)
        return self._value


def submit_gpu_prefetch(
    fn: Callable[[], Any], stream: torch.cuda.Stream
) -> GpuPrefetchHandle:
    """Run ``fn()`` on ``stream`` and return a handle for its result.

    ``fn`` must only touch tensors private to this call (its own inputs and
    freshly allocated outputs) -- never anything shared with what the
    caller's own current stream is doing, since nothing here makes that
    safe on its own. Recording the completion event happens on ``stream``
    right after ``fn()`` returns; ``GpuPrefetchHandle.result()`` is what
    inserts the actual cross-stream wait.
    """
    with torch.cuda.stream(stream):
        value = fn()
        event = torch.cuda.Event()
        event.record(stream)
    return GpuPrefetchHandle(_value=value, _event=event)
