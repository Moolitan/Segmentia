from __future__ import annotations

import time
from contextlib import AbstractContextManager
from dataclasses import dataclass
from types import TracebackType
from typing import Protocol

import torch


class TimingSink(Protocol):
    def add_host_time(self, stage: str, elapsed_ms: float) -> None: ...

    def add_cuda_measurement(self, measurement: "CudaMeasurement") -> None: ...


@dataclass
class CudaMeasurement:
    """Deferred CUDA-event measurement resolved once at trace completion."""

    stage: str
    start: torch.cuda.Event
    end: torch.cuda.Event

    def resolve_ms(self) -> float:
        self.end.synchronize()
        return float(self.start.elapsed_time(self.end))


class CpuStageTimer(AbstractContextManager[None]):
    def __init__(self, sink: TimingSink, stage: str) -> None:
        self._sink = sink
        self._stage = stage
        self._started = 0.0

    def __enter__(self) -> None:
        self._started = time.perf_counter()
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._sink.add_host_time(
            self._stage, (time.perf_counter() - self._started) * 1000.0
        )
        return None


class CudaStageTimer(AbstractContextManager[None]):
    """Record GPU execution without synchronizing at every stage boundary.

    Host launch time is retained separately. CUDA events are resolved together
    when the enclosing load trace finishes, avoiding forty per-layer syncs.
    CPU tensors use the ordinary wall-clock timer, which keeps unit tests and
    non-CUDA connectors usable.
    """

    def __init__(self, sink: TimingSink, stage: str, device: torch.device | str) -> None:
        self._sink = sink
        self._stage = stage
        self._device = torch.device(device)
        self._fallback: CpuStageTimer | None = None
        self._start: torch.cuda.Event | None = None
        self._end: torch.cuda.Event | None = None
        self._host_started = 0.0

    def __enter__(self) -> None:
        if self._device.type != "cuda" or not torch.cuda.is_available():
            self._fallback = CpuStageTimer(self._sink, self._stage)
            self._fallback.__enter__()
            return None

        self._host_started = time.perf_counter()
        with torch.cuda.device(self._device):
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
            self._start.record(torch.cuda.current_stream(self._device))
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._fallback is not None:
            self._fallback.__exit__(exc_type, exc_value, traceback)
            return None

        assert self._start is not None and self._end is not None
        with torch.cuda.device(self._device):
            self._end.record(torch.cuda.current_stream(self._device))
        self._sink.add_host_time(
            f"{self._stage}_host", (time.perf_counter() - self._host_started) * 1000.0
        )
        self._sink.add_cuda_measurement(
            CudaMeasurement(stage=self._stage, start=self._start, end=self._end)
        )
        return None
