from __future__ import annotations

import time
from collections import defaultdict
from contextlib import nullcontext
from threading import Lock
from typing import Any, Mapping

import torch

from cskcache.profiling.config import ProfileConfig
from cskcache.profiling.reporter import ProfileReporter
from cskcache.profiling.timer import CpuStageTimer, CudaMeasurement, CudaStageTimer


class NullLoadTrace:
    """No-op trace used on the default, profiling-disabled path."""

    enabled = False

    def cpu_stage(self, stage: str):
        return nullcontext()

    def cuda_stage(self, stage: str, device: torch.device | str):
        return nullcontext()

    def set(self, **values: Any) -> None:
        return None

    def add_host_time(self, stage: str, elapsed_ms: float) -> None:
        return None

    def add_cuda_measurement(self, measurement: CudaMeasurement) -> None:
        return None


class LoadTrace:
    """Structured timing record for one scheduler lookup or worker load."""

    enabled = True

    def __init__(
        self,
        *,
        kind: str,
        req_id: str,
        cache_id: str,
        reuse_index: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.kind = kind
        self.req_id = req_id
        self.cache_id = cache_id
        self.reuse_index = reuse_index
        self.trace_id = f"{kind}:{req_id}:{reuse_index}"
        self._created = time.perf_counter()
        self._metadata: dict[str, Any] = dict(metadata or {})
        self._host_ms: dict[str, list[float]] = defaultdict(list)
        self._cuda_measurements: list[CudaMeasurement] = []
        self._cuda_ms: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def cpu_stage(self, stage: str) -> CpuStageTimer:
        return CpuStageTimer(self, stage)

    def cuda_stage(
        self, stage: str, device: torch.device | str
    ) -> CudaStageTimer:
        return CudaStageTimer(self, stage, device)

    def set(self, **values: Any) -> None:
        with self._lock:
            self._metadata.update(values)

    def add_host_time(self, stage: str, elapsed_ms: float) -> None:
        with self._lock:
            self._host_ms[stage].append(float(elapsed_ms))

    def add_cuda_measurement(self, measurement: CudaMeasurement) -> None:
        with self._lock:
            self._cuda_measurements.append(measurement)

    @staticmethod
    def _totals(values: Mapping[str, list[float]]) -> dict[str, float]:
        return {name: sum(samples) for name, samples in values.items()}

    @staticmethod
    def _stats(values: Mapping[str, list[float]]) -> dict[str, dict[str, float | int]]:
        return {
            name: {
                "count": len(samples),
                "total_ms": sum(samples),
                "mean_ms": sum(samples) / len(samples),
                "max_ms": max(samples),
            }
            for name, samples in values.items()
            if samples
        }

    def finish(self, *, status: str, error: str | None = None) -> dict[str, Any]:
        # Copy before resolving so instrumentation remains safe if a future
        # connector records stages from more than one host thread.
        with self._lock:
            pending = tuple(self._cuda_measurements)
        for measurement in pending:
            self._cuda_ms[measurement.stage].append(measurement.resolve_ms())

        host_ms = self._totals(self._host_ms)
        cuda_ms = self._totals(self._cuda_ms)
        host_stats = self._stats(self._host_ms)
        cuda_stats = self._stats(self._cuda_ms)
        # Prefer device execution time for CUDA stages; CPU-only stages retain
        # wall-clock time. Host launch time remains available as a separate map.
        stage_ms = dict(host_ms)
        stage_ms.update(cuda_ms)
        total_ms = (time.perf_counter() - self._created) * 1000.0
        byte_count = int(self._metadata.get("bytes", 0) or 0)
        effective_gbps = (
            byte_count / (total_ms * 1_000_000.0)
            if self.kind == "worker_load" and total_ms > 0
            else 0.0
        )
        record: dict[str, Any] = {
            "kind": self.kind,
            "trace_id": self.trace_id,
            "req_id": self.req_id,
            "cache_id": self.cache_id,
            "reuse_index": self.reuse_index,
            **self._metadata,
            "host_stage_ms": host_ms,
            "cuda_stage_ms": cuda_ms,
            "host_stage_stats": host_stats,
            "cuda_stage_stats": cuda_stats,
            "stage_stats": {**host_stats, **cuda_stats},
            "stage_ms": stage_ms,
            "total_ms": total_ms,
            "effective_gbps": effective_gbps,
            "status": status,
        }
        if error is not None:
            record["error"] = error
        return record


class Profiler:
    """Trace factory and reporter owned by one CSKCache engine instance."""

    def __init__(
        self,
        config: ProfileConfig | None = None,
        reporter: ProfileReporter | None = None,
    ) -> None:
        self.config = config if config is not None else ProfileConfig.from_env()
        self.reporter = reporter if reporter is not None else ProfileReporter(self.config)
        self._next_trace_index = 0
        self._lock = Lock()

    def start(
        self,
        *,
        kind: str,
        req_id: str,
        cache_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> LoadTrace | NullLoadTrace:
        if not self.config.enabled:
            return NullLoadTrace()
        with self._lock:
            self._next_trace_index += 1
            reuse_index = self._next_trace_index
        return LoadTrace(
            kind=kind,
            req_id=req_id,
            cache_id=cache_id,
            reuse_index=reuse_index,
            metadata=metadata,
        )

    def start_worker_load(
        self,
        *,
        req_id: str,
        cache_id: str,
        target_start: int,
        target_end: int,
        tokens: int,
        source_offset: int,
    ) -> LoadTrace | NullLoadTrace:
        return self.start(
            kind="worker_load",
            req_id=req_id,
            cache_id=cache_id,
            metadata={
                "target_start": target_start,
                "target_end": target_end,
                "tokens": tokens,
                "source_offset": source_offset,
            },
        )

    def start_scheduler_lookup(
        self,
        *,
        req_id: str,
        cache_id: str,
        target_start: int | None,
        target_end: int | None,
    ) -> LoadTrace | NullLoadTrace:
        tokens = (
            target_end - target_start
            if target_start is not None and target_end is not None
            else None
        )
        return self.start(
            kind="scheduler_lookup",
            req_id=req_id,
            cache_id=cache_id,
            metadata={
                "target_start": target_start,
                "target_end": target_end,
                "tokens": tokens,
            },
        )

    def finish(
        self,
        trace: LoadTrace | NullLoadTrace,
        *,
        status: str = "ok",
        error: str | None = None,
    ) -> dict[str, Any] | None:
        if not trace.enabled:
            return None
        assert isinstance(trace, LoadTrace)
        record = trace.finish(status=status, error=error)
        self.reporter.report(record)
        return record
