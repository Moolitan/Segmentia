"""Measure sustained Skill-invocation throughput at fixed concurrency."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import os
from pathlib import Path
import threading
import time
from typing import Any
from urllib.request import urlopen

import config as local
from paper_evaluation import config as suite
from paper_evaluation.config import BASE_PORT, OUTPUT_ROOT, PLATFORMS
from section_6_3_latency_scaling.workload import (
    Workload,
    load_fixed_workloads,
    write_catalog_view,
)
from common.driver import (
    RequestResult,
    SystemVariant,
    execute_prepared_request,
    make_server_config,
    prepare_request_pair,
)
from common.run_state import RunContext, utc_now
from common.server import VLLMServer


SECTION = "section_6_6_concurrency"


@dataclass(frozen=True)
class CompletedRequest:
    case_id: str
    slot: int
    sequence: int
    started_ns: int
    completed_ns: int
    result: RequestResult


@dataclass(frozen=True)
class WindowResult:
    started_ns: int
    deadline_ns: int
    drained_ns: int
    requests: tuple[CompletedRequest, ...]
    errors: tuple[str, ...]

    @property
    def counted(self) -> tuple[CompletedRequest, ...]:
        return tuple(
            item for item in self.requests if item.completed_ns <= self.deadline_ns
        )


def _process_group_rss_bytes(process_group: int) -> int:
    page_size = os.sysconf("SC_PAGE_SIZE")
    rss_pages = 0
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="utf-8")
            fields = stat[stat.rfind(")") + 2 :].split()
            if int(fields[2]) == process_group:
                rss_pages += int(fields[21])
        except (FileNotFoundError, IndexError, OSError, ValueError):
            continue
    return rss_pages * page_size


def _system_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("/proc/meminfo has no MemAvailable entry")


def _lmcache_metrics(base_url: str) -> tuple[int | None, int | None]:
    with urlopen(f"{base_url}/metrics", timeout=1.0) as response:
        payload = response.read().decode("utf-8")
    usage_values: list[float] = []
    object_values: list[float] = []
    for line in payload.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2:
            continue
        metric = fields[0].split("{", 1)[0].replace(":", "_")
        try:
            value = float(fields[1])
        except ValueError:
            continue
        if metric.endswith("lmcache_local_cache_usage"):
            usage_values.append(value)
        elif metric.endswith("lmcache_active_memory_objs_count"):
            object_values.append(value)
    return (
        None if not usage_values else round(sum(usage_values)),
        None if not object_values else round(sum(object_values)),
    )


class MemoryMonitor:
    """Sample LMCache page use and process RSS without touching request flow."""

    def __init__(
        self,
        server: VLLMServer,
        path: Path,
        *,
        interval_seconds: float,
        host_pool_gib: float,
    ) -> None:
        if server.process is None:
            raise RuntimeError("memory monitor requires a running server")
        if interval_seconds <= 0:
            raise ValueError("memory sampling interval must be positive")
        self._server = server
        self._path = path
        self._interval_seconds = interval_seconds
        self._host_pool_gib = host_pool_gib
        self._process_group = server.process.pid
        self._phase = "startup"
        self._phase_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_ns = 0

    def start(self, phase: str) -> None:
        if self._thread is not None:
            raise RuntimeError("memory monitor already started")
        self.set_phase(phase)
        self._path.write_text("", encoding="utf-8")
        self._started_ns = time.monotonic_ns()
        self._thread = threading.Thread(
            target=self._run,
            name="concurrency-memory-monitor",
            daemon=True,
        )
        self._thread.start()

    def set_phase(self, phase: str) -> None:
        with self._phase_lock:
            self._phase = phase

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                raise RuntimeError("memory monitor did not stop")
        summary = self._summarize()
        self._path.with_name("memory_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return summary

    def _run(self) -> None:
        while not self._stop.is_set():
            sampled_ns = time.monotonic_ns()
            with self._phase_lock:
                phase = self._phase
            error = ""
            try:
                host_bytes, active_objects = _lmcache_metrics(
                    self._server.base_url
                )
            except Exception as exc:
                host_bytes, active_objects = None, None
                error = f"{type(exc).__name__}: {exc}"
            try:
                process_rss = _process_group_rss_bytes(self._process_group)
                available = _system_available_bytes()
            except Exception as exc:
                process_rss, available = None, None
                suffix = f"{type(exc).__name__}: {exc}"
                error = f"{error}; {suffix}" if error else suffix
            record = {
                "elapsed_seconds": (sampled_ns - self._started_ns) / 1e9,
                "phase": phase,
                "host_pool_capacity_bytes": round(
                    self._host_pool_gib * 1024**3
                ),
                "lmcache_host_allocated_bytes": host_bytes,
                "lmcache_active_memory_objects": active_objects,
                "server_process_group_rss_bytes": process_rss,
                "system_available_bytes": available,
                "error": error,
            }
            with self._path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(record, sort_keys=True) + "\n")
            self._stop.wait(self._interval_seconds)

    def _summarize(self) -> dict[str, Any]:
        samples = [
            json.loads(line)
            for line in self._path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        measured = [row for row in samples if row["phase"] == "measure"]

        def maximum(field: str) -> int | None:
            values = [row[field] for row in measured if row[field] is not None]
            return None if not values else max(values)

        def minimum(field: str) -> int | None:
            values = [row[field] for row in measured if row[field] is not None]
            return None if not values else min(values)

        return {
            "host_pool_capacity_bytes": round(self._host_pool_gib * 1024**3),
            "sample_interval_seconds": self._interval_seconds,
            "measurement_sample_count": len(measured),
            "measurement_error_count": sum(bool(row["error"]) for row in measured),
            "peak_lmcache_host_allocated_bytes": maximum(
                "lmcache_host_allocated_bytes"
            ),
            "peak_lmcache_active_memory_objects": maximum(
                "lmcache_active_memory_objects"
            ),
            "peak_server_process_group_rss_bytes": maximum(
                "server_process_group_rss_bytes"
            ),
            "minimum_system_available_bytes": minimum("system_available_bytes"),
        }


def _next_attempt(point_root: Path) -> Path:
    point_root.mkdir(parents=True, exist_ok=True)
    indexes = []
    for path in point_root.glob("attempt-*"):
        try:
            indexes.append(int(path.name.removeprefix("attempt-")))
        except ValueError:
            continue
    attempt = point_root / f"attempt-{max(indexes, default=0) + 1:03d}"
    attempt.mkdir()
    return attempt


def _request_id(
    point_id: str, phase: str, slot: int, sequence: int
) -> str:
    return f"{point_id}__{phase}__slot{slot}__request{sequence:05d}"


def _worker_loop(
    *,
    server: VLLMServer,
    variant: SystemVariant,
    workload: Workload,
    skill_text: str,
    base_task: str,
    point_id: str,
    phase: str,
    slot: int,
    barrier: threading.Barrier,
    clock_ns: list[int],
    stop: threading.Event,
) -> tuple[list[CompletedRequest], list[str]]:
    requests: list[CompletedRequest] = []
    errors: list[str] = []
    sequence = 0
    barrier.wait()
    deadline_ns = clock_ns[1]
    while time.monotonic_ns() < deadline_ns and not stop.is_set():
        case_id = _request_id(point_id, phase, slot, sequence)
        request_started_ns = time.monotonic_ns()
        unique_task = (
            f"Benchmark request nonce: {case_id}.\n\n{base_task}"
        )
        try:
            # Request A is part of the Skill invocation. It produces a unique
            # tool-call ticket and triggers SkillCache prefetch before request B.
            prepared = prepare_request_pair(
                server,
                variant=variant,
                skill_name=workload.skill_name,
                skill_text=skill_text,
                task_prompt=unique_task,
                case_id=case_id,
                selection_prompt=(
                    f"Load the {workload.skill_name} Skill for benchmark "
                    f"request {case_id}."
                ),
            )
            result = execute_prepared_request(
                server,
                variant=variant,
                prepared=prepared,
                skill_name=workload.skill_name,
                case_id=case_id,
                max_tokens=local.MAX_TOKENS,
                stream=True,
                reset_prefix_cache=False,
            )
        except Exception as exc:  # Preserve the first concrete request failure.
            errors.append(f"{case_id}: {type(exc).__name__}: {exc}")
            stop.set()
            break
        requests.append(
            CompletedRequest(
                case_id=case_id,
                slot=slot,
                sequence=sequence,
                started_ns=request_started_ns,
                completed_ns=time.monotonic_ns(),
                result=result,
            )
        )
        sequence += 1
    return requests, errors


def _run_window(
    *,
    server: VLLMServer,
    variant: SystemVariant,
    workload: Workload,
    skill_text: str,
    base_task: str,
    point_id: str,
    phase: str,
    concurrency: int,
    duration_seconds: float,
) -> WindowResult:
    if concurrency <= 0 or duration_seconds <= 0:
        raise ValueError("concurrency and duration_seconds must be positive")
    barrier = threading.Barrier(concurrency + 1)
    stop = threading.Event()
    clock_ns = [0, 0]
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                _worker_loop,
                server=server,
                variant=variant,
                workload=workload,
                skill_text=skill_text,
                base_task=base_task,
                point_id=point_id,
                phase=phase,
                slot=slot,
                barrier=barrier,
                clock_ns=clock_ns,
                stop=stop,
            )
            for slot in range(concurrency)
        ]
        started_ns = time.monotonic_ns()
        deadline_ns = started_ns + int(duration_seconds * 1e9)
        clock_ns[:] = [started_ns, deadline_ns]
        barrier.wait()
        worker_results = [future.result() for future in futures]
    drained_ns = time.monotonic_ns()
    requests = tuple(
        item for completed, _errors in worker_results for item in completed
    )
    errors = tuple(error for _completed, values in worker_results for error in values)
    return WindowResult(started_ns, deadline_ns, drained_ns, requests, errors)


def _point_id(
    platform_id: str,
    variant: SystemVariant,
    workload: Workload,
    concurrency: int,
    replica: int,
) -> str:
    return (
        f"{platform_id}__{variant.name}__{workload.length_bucket}__"
        f"{workload.skill_name}__c{concurrency}__replica{replica}"
    ).replace(">", "gt").replace("%", "pct").replace(" ", "-")


def _write_point_result(
    path: Path,
    *,
    point_id: str,
    warmup: WindowResult,
    measured: WindowResult,
    measurement_seconds: float,
    memory: dict[str, Any],
    point_metadata: dict[str, Any],
) -> None:
    counted = measured.counted
    payload: dict[str, Any] = {
        "point_id": point_id,
        "warmup_completed_within_window": len(warmup.counted),
        "warmup_completed_total_after_drain": len(warmup.requests),
        "measurement_seconds": measurement_seconds,
        "completed_within_window": len(counted),
        "completed_total_after_drain": len(measured.requests),
        "drain_seconds": max(
            0.0, (measured.drained_ns - measured.deadline_ns) / 1e9
        ),
        "throughput_requests_per_s": len(counted) / measurement_seconds,
        "memory": memory,
        "point_metadata": point_metadata,
        "errors": list(warmup.errors + measured.errors),
        "requests": [
            {
                "case_id": item.case_id,
                "slot": item.slot,
                "sequence": item.sequence,
                "completed_within_window": item.completed_ns <= measured.deadline_ns,
                "ttft_ms": item.result.server_ttft_ms,
                "latency_ms": item.result.completion.client_latency_ms,
                "prompt_tokens": item.result.prompt_tokens,
                "reused_tokens": item.result.cached_tokens,
                "fallback": item.result.fallback,
                "fallback_reason": item.result.fallback_reason,
            }
            for item in measured.requests
        ],
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    if tuple(local.PLATFORM_IDS) != ("a6000_qwen3_14b",):
        raise RuntimeError("the verified fixed-length pool is for Qwen3-14B")
    workloads, master_catalog, selection = load_fixed_workloads(
        pool_root=local.FIXED_LENGTH_POOL_ROOT,
        expected_model_id=PLATFORMS[local.PLATFORM_IDS[0]].model_id,
        buckets=(
            ("<1K", 0, 1_000),
            ("1K-3K", 1_000, 3_000),
            ("3K-5K", 3_000, 5_000),
            ("5K-8K", 5_000, 8_000),
            ("8K-10K", 8_000, 10_000),
            (">10K", 10_000, None),
        ),
        max_ratio=local.MAX_CALIBRATION_RATIO,
        minimum_reuse_tokens=local.MINIMUM_REUSE_TOKENS,
    )
    selected = [
        workload for workload in workloads
        if workload.length_bucket in local.SELECTED_LENGTH_BUCKETS
    ]
    if [item.length_bucket for item in selected] != list(
        local.SELECTED_LENGTH_BUCKETS
    ):
        raise RuntimeError("selected throughput length buckets are incomplete")

    source_paths = [
        local.FIXED_LENGTH_POOL_ROOT / "fixed_length_manifest.json",
        local.FIXED_LENGTH_POOL_ROOT / "raw/catalog.json",
        local.WORKLOAD_MODULE,
        *(item.skill_path for item in selected),
        *(item.task_path for item in selected),
    ]
    values = {
        "platform_ids": list(local.PLATFORM_IDS),
        "systems": [variant.__dict__ for variant in local.SYSTEMS],
        "concurrencies": list(local.CONCURRENCIES),
        "replicas": local.REPLICAS,
        "warmup_seconds": local.WARMUP_SECONDS,
        "measurement_seconds": local.MEASUREMENT_SECONDS,
        "max_tokens": local.MAX_TOKENS,
        "chunk_tokens": local.CHUNK_TOKENS,
        "host_page_tokens": local.HOST_PAGE_TOKENS,
        "host_pool_gib": local.HOST_POOL_GIB,
        "memory_sample_interval_seconds": local.MEMORY_SAMPLE_INTERVAL_SECONDS,
        "catalog_sha256": selection["catalog_sha256"],
        "workloads": [
            {
                "task_id": item.task_id,
                "skill_name": item.skill_name,
                "skill_tokens": item.skill_tokens,
                "length_bucket": item.length_bucket,
                "object_id": item.object_id,
            }
            for item in selected
        ],
        "load_model": "closed_loop_skill_invocation",
        "throughput_unit": "completed_request_b_per_second",
        "request_a_policy": "included_in_closed_loop_wall_time",
        "prefix_policy": "unique_nonce_before_skill_no_cross_point_reuse",
    }
    run = RunContext.open(
        output_root=OUTPUT_ROOT,
        section=SECTION,
        config_paths=(Path(__file__), Path(local.__file__), Path(suite.__file__), *source_paths),
        config_values=values,
    )
    catalog_view = run.run_dir / "selected_catalog.json"
    write_catalog_view(master_catalog, selected, catalog_view)

    for platform_index, platform_id in enumerate(local.PLATFORM_IDS):
        platform = PLATFORMS[platform_id]
        if not platform.model_path.is_dir():
            raise FileNotFoundError(f"model does not exist: {platform.model_path}")
        for system_index, variant in enumerate(local.SYSTEMS):
            for workload_index, workload in enumerate(selected):
                skill_text = workload.skill_path.read_text(encoding="utf-8")
                base_task = workload.task_path.read_text(encoding="utf-8").strip()
                for concurrency_index, concurrency in enumerate(local.CONCURRENCIES):
                    for replica in range(local.REPLICAS):
                        point_id = _point_id(
                            platform_id, variant, workload, concurrency, replica
                        )
                        if run.completed(point_id):
                            continue
                        attempt = _next_attempt(run.run_dir / "points" / point_id)
                        run.mark(point_id, "running", attempt_dir=str(attempt))
                        port = (
                            BASE_PORT
                            + platform_index * 100
                            + system_index * 30
                            + workload_index * 12
                            + concurrency_index * 3
                            + replica
                        )
                        server_cfg = make_server_config(
                            platform=platform,
                            variant=variant,
                            port=port,
                            case_root=attempt,
                            chunk_tokens=local.CHUNK_TOKENS,
                            correction_alpha=local.CORRECTION_ALPHA,
                            minimum_reuse_tokens=local.MINIMUM_REUSE_TOKENS,
                            catalog_override=catalog_view,
                            host_page_tokens=local.HOST_PAGE_TOKENS,
                            max_local_cpu_gib=local.HOST_POOL_GIB,
                        )
                        print(
                            f"[point] {point_id} attempt={attempt.name}",
                            flush=True,
                        )
                        try:
                            with VLLMServer(server_cfg) as server:
                                monitor = MemoryMonitor(
                                    server,
                                    attempt / "memory_samples.jsonl",
                                    interval_seconds=(
                                        local.MEMORY_SAMPLE_INTERVAL_SECONDS
                                    ),
                                    host_pool_gib=(
                                        local.HOST_POOL_GIB
                                        if variant.family == "cskcache"
                                        else 0.0
                                    ),
                                )
                                monitor.start("warmup")
                                try:
                                    warmup = _run_window(
                                        server=server,
                                        variant=variant,
                                        workload=workload,
                                        skill_text=skill_text,
                                        base_task=base_task,
                                        point_id=point_id,
                                        phase="warmup",
                                        concurrency=concurrency,
                                        duration_seconds=local.WARMUP_SECONDS,
                                    )
                                    if warmup.errors:
                                        raise RuntimeError(warmup.errors[0])
                                    monitor.set_phase("reset")
                                    server.reset_prefix_cache()
                                    monitor.set_phase("measure")
                                    measured = _run_window(
                                        server=server,
                                        variant=variant,
                                        workload=workload,
                                        skill_text=skill_text,
                                        base_task=base_task,
                                        point_id=point_id,
                                        phase="measure",
                                        concurrency=concurrency,
                                        duration_seconds=local.MEASUREMENT_SECONDS,
                                    )
                                finally:
                                    memory = monitor.stop()
                            _write_point_result(
                                attempt / "result.json",
                                point_id=point_id,
                                warmup=warmup,
                                measured=measured,
                                measurement_seconds=local.MEASUREMENT_SECONDS,
                                memory=memory,
                                point_metadata={
                                    "platform_id": platform_id,
                                    "system": variant.name,
                                    "skill_name": workload.skill_name,
                                    "skill_tokens": workload.skill_tokens,
                                    "concurrency": concurrency,
                                    "replica": replica,
                                },
                            )
                            if measured.errors:
                                raise RuntimeError(measured.errors[0])
                            counted = measured.counted
                            if not counted:
                                raise RuntimeError(
                                    "no request B completed inside measurement window"
                                )
                            fallbacks = [
                                item for item in counted if item.result.fallback
                            ]
                            if fallbacks:
                                raise RuntimeError(
                                    f"{len(fallbacks)} measured requests fell back"
                                )
                            if variant.family == "cskcache" and any(
                                item.result.cached_tokens <= 0 for item in counted
                            ):
                                raise RuntimeError(
                                    "SkillCache measured request reported zero reused tokens"
                                )
                            throughput = len(counted) / local.MEASUREMENT_SECONDS
                            for repetition, item in enumerate(counted):
                                result = item.result
                                run.record(
                                    {
                                        "case_id": item.case_id,
                                        "status": "completed",
                                        "platform_id": platform_id,
                                        "gpu_name": platform.gpu_name,
                                        "model_id": platform.model_id,
                                        "model_path": str(platform.model_path),
                                        "tensor_parallel_size": platform.tensor_parallel_size,
                                        "system": variant.name,
                                        "source_dataset": workload.source_type,
                                        "skill_name": workload.skill_name,
                                        "skill_tokens": workload.skill_tokens,
                                        "task_id": workload.task_id,
                                        "chunk_tokens": local.CHUNK_TOKENS,
                                        "correction_strategy": variant.correction_strategy,
                                        "correction_budget_tokens": "",
                                        "correction_ratio": (
                                            variant.calibration_ratio
                                            if variant.calibration_ratio is not None
                                            else ""
                                        ),
                                        "concurrency": concurrency,
                                        "replica": replica,
                                        "repetition": repetition,
                                        "warmup": False,
                                        "prompt_tokens": result.prompt_tokens,
                                        "reused_tokens": result.cached_tokens,
                                        "reuse_ratio": (
                                            result.cached_tokens / result.prompt_tokens
                                            if result.prompt_tokens else 0.0
                                        ),
                                        "ttft_ms": result.server_ttft_ms,
                                        "latency_ms": result.completion.client_latency_ms,
                                        "batch_elapsed_ms": (
                                            local.MEASUREMENT_SECONDS * 1000.0
                                        ),
                                        "throughput_requests_per_s": throughput,
                                        "output_tokens": result.completion.output_tokens,
                                        "fallback": result.fallback,
                                        "fallback_reason": result.fallback_reason,
                                        "started_utc": utc_now(),
                                        "completed_utc": utc_now(),
                                    }
                                )
                            run.mark(
                                point_id,
                                "completed",
                                attempt_dir=str(attempt),
                                completed_requests=len(counted),
                                throughput_requests_per_s=throughput,
                            )
                        except Exception as exc:
                            (attempt / "failure.txt").write_text(
                                f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
                            )
                            run.mark(
                                point_id,
                                "failed",
                                attempt_dir=str(attempt),
                                error=f"{type(exc).__name__}: {exc}",
                            )
                            raise

    from analyze import analyze

    analyze(run.run_dir)
    run.finish()
    print(f"results={run.run_dir}")


if __name__ == "__main__":
    main()
