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

    def record_tier(self, tier: str) -> None:
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
        self._started_at_ns = time.time_ns()
        self._metadata: dict[str, Any] = dict(metadata or {})
        self._host_ms: dict[str, list[float]] = defaultdict(list)
        self._cuda_measurements: list[CudaMeasurement] = []
        self._cuda_ms: dict[str, list[float]] = defaultdict(list)
        self._tier_counts: dict[str, int] = defaultdict(int)
        self._first_tier: str | None = None
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

    def record_tier(self, tier: str) -> None:
        with self._lock:
            if self._first_tier is None:
                self._first_tier = tier
            self._tier_counts[tier] += 1

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
            "started_at_ns": self._started_at_ns,
            "finished_at_ns": time.time_ns(),
            "source_tier": self._first_tier or "unknown",
            "tier_access_counts": dict(self._tier_counts),
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


class TimelineTrace:
    """Ordered scheduler milestones for one reusable skill occurrence."""

    # bulk_preload_dispatched fires the instant the frontier reaches the
    # span start (CSKReuseStage.LOADING -> PROBING), so it is the accurate
    # end of "gap_prefill" -- the delayed gap_completed log (recorded a step
    # later, inside cap_prefill_before_reuse) is not, since by then the
    # worker has already finished scattering the whole span. Using
    # gap_completed as bulk_preload's end instead is correct precisely
    # because engine steps are sequential: that later step cannot start
    # until the bulk-preload step's worker-side to_gpu() call has returned.
    _DURATION_PAIRS = {
        "gap_prefill": ("gap_scheduled", "bulk_preload_dispatched"),
        "bulk_preload": ("bulk_preload_dispatched", "gap_completed"),
        "probe_roundtrip": ("probe_dispatched", "probe_decision_received"),
        "recompute_prefill": ("recompute_scheduled", "recompute_completed"),
        "load_dispatch": ("load_planned", "load_dispatched"),
    }

    def __init__(
        self,
        *,
        req_id: str,
        cache_id: str,
        reuse_index: int,
        target_start: int,
        target_end: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.req_id = req_id
        self.cache_id = cache_id
        self.reuse_index = reuse_index
        self.trace_id = f"request_timeline:{req_id}:{reuse_index}"
        self.target_start = target_start
        self.target_end = target_end
        self._created = time.perf_counter()
        self._started_at_ns = time.time_ns()
        self._metadata = dict(metadata or {})
        self._events: list[dict[str, Any]] = []

    def mark(self, event: str, metadata: Mapping[str, Any] | None = None) -> None:
        item: dict[str, Any] = {
            "event": event,
            "offset_ms": (time.perf_counter() - self._created) * 1000.0,
            "wall_time_ns": time.time_ns(),
        }
        if metadata:
            item["metadata"] = dict(metadata)
        self._events.append(item)

    def finish(self) -> dict[str, Any]:
        self.mark("request_finished")
        first_offsets = {
            str(item["event"]): float(item["offset_ms"])
            for item in self._events
        }
        stage_ms = {
            name: first_offsets[end] - first_offsets[start]
            for name, (start, end) in self._DURATION_PAIRS.items()
            if start in first_offsets and end in first_offsets
        }
        return {
            "kind": "request_timeline",
            "trace_id": self.trace_id,
            "req_id": self.req_id,
            "cache_id": self.cache_id,
            "reuse_index": self.reuse_index,
            "target_start": self.target_start,
            "target_end": self.target_end,
            "tokens": self.target_end - self.target_start,
            "started_at_ns": self._started_at_ns,
            "finished_at_ns": time.time_ns(),
            **self._metadata,
            "events": list(self._events),
            "stage_ms": stage_ms,
            "total_ms": first_offsets.get("request_finished", 0.0),
            "effective_gbps": 0.0,
            "status": "ok",
        }


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
        self._active_probe_traces: dict[tuple[str, str], LoadTrace] = {}
        self._active_timelines: dict[tuple[str, str, int], TimelineTrace] = {}
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

    def get_or_start_probe_capture(
        self,
        *,
        req_id: str,
        cache_id: str,
        target_start: int,
        target_end: int,
    ) -> LoadTrace | NullLoadTrace:
        if not self.config.enabled:
            return NullLoadTrace()
        key = (req_id, cache_id)
        with self._lock:
            trace = self._active_probe_traces.get(key)
            if trace is not None:
                return trace
            self._next_trace_index += 1
            trace = LoadTrace(
                kind="worker_probe_capture",
                req_id=req_id,
                cache_id=cache_id,
                reuse_index=self._next_trace_index,
                metadata={
                    "target_start": target_start,
                    "target_end": target_end,
                    "tokens": target_end - target_start,
                },
            )
            self._active_probe_traces[key] = trace
            return trace

    def finish_probe_capture(
        self,
        *,
        req_id: str,
        cache_id: str,
        expected_layers: int,
        captured_layers: int,
        status: str = "ok",
        error: str | None = None,
    ) -> dict[str, Any] | None:
        with self._lock:
            trace = self._active_probe_traces.pop((req_id, cache_id), None)
        if trace is None:
            return None
        trace.set(
            expected_layers=expected_layers,
            captured_layers=captured_layers,
        )
        return self.finish(trace, status=status, error=error)

    def discard_probe_capture(self, *, req_id: str, cache_id: str) -> None:
        with self._lock:
            self._active_probe_traces.pop((req_id, cache_id), None)

    def register_timeline(
        self,
        *,
        req_id: str,
        cache_id: str,
        target_start: int,
        target_end: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.config.enabled:
            return
        key = (req_id, cache_id, target_start)
        with self._lock:
            if key in self._active_timelines:
                return
            self._next_trace_index += 1
            timeline = TimelineTrace(
                req_id=req_id,
                cache_id=cache_id,
                reuse_index=self._next_trace_index,
                target_start=target_start,
                target_end=target_end,
                metadata=metadata,
            )
            timeline.mark("reuse_accepted")
            self._active_timelines[key] = timeline

    def mark_timeline(
        self,
        *,
        req_id: str,
        cache_id: str,
        target_start: int,
        event: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not self.config.enabled:
            return
        with self._lock:
            timeline = self._active_timelines.get((req_id, cache_id, target_start))
            if timeline is not None:
                timeline.mark(event, metadata)

    def finish_request_timelines(self, req_id: str) -> list[dict[str, Any]]:
        if not self.config.enabled:
            return []
        with self._lock:
            keys = [key for key in self._active_timelines if key[0] == req_id]
            timelines = [self._active_timelines.pop(key) for key in keys]
        records = [timeline.finish() for timeline in timelines]
        for record in records:
            self.reporter.report(record)
        return records

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
