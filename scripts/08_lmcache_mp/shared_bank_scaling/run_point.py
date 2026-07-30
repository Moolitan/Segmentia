#!/usr/bin/env python3
"""Send one fixed follower batch and sample vLLM KV-pool usage."""
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from capture_common import atomic_write_json
from shared_bank_concurrency_closure.send_concurrent import load_specs
from shared_bank_gpu_closure.send_request import post_json


def kv_cache_usage(base_url: str, api_key: str) -> float | None:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/metrics",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            for line in response.read().decode("utf-8").splitlines():
                if line.startswith("vllm:kv_cache_usage_perc"):
                    return float(line.rsplit(" ", 1)[1])
    except Exception:
        return None
    return None


class KVUsageSampler(threading.Thread):
    def __init__(
        self, base_url: str, api_key: str, interval_s: float = 0.05
    ) -> None:
        super().__init__(daemon=True)
        self.base_url = base_url
        self.api_key = api_key
        self.interval_s = interval_s
        self.samples: list[dict[str, float]] = []
        self._stop_event = threading.Event()

    def run(self) -> None:
        started = time.perf_counter()
        while not self._stop_event.is_set():
            value = kv_cache_usage(self.base_url, self.api_key)
            if value is not None:
                self.samples.append(
                    {
                        "elapsed_s": round(time.perf_counter() - started, 6),
                        "usage": value,
                    }
                )
            self._stop_event.wait(self.interval_s)

    def stop(self) -> list[dict[str, float]]:
        self._stop_event.set()
        self.join(timeout=2)
        return self.samples


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * quantile))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--followers", type=int, required=True)
    parser.add_argument(
        "--mode", choices=("full", "materialized", "shared"), required=True
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8100")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--sample-interval-s", type=float, default=0.05)
    args = parser.parse_args()
    if args.followers < 1:
        raise ValueError("followers must be positive")
    if args.sample_interval_s <= 0:
        raise ValueError("sample interval must be positive")
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")

    records = load_specs(args.spec_dir, args.followers)
    args.output_dir.mkdir(parents=True)
    barrier = threading.Barrier(args.followers + 1)

    def send_one(index: int) -> dict[str, Any]:
        record = dict(records[index])
        barrier.wait()
        started = time.perf_counter()
        try:
            response = post_json(
                f"{args.base_url.rstrip('/')}/v1/completions",
                args.api_key,
                record["request"],
            )
            if "error" in response:
                raise RuntimeError(
                    f"vLLM returned an error object: {response['error']}"
                )
            record.update(
                status="completed",
                elapsed_s=round(time.perf_counter() - started, 6),
                response=response,
                response_id=response.get("id"),
            )
        except Exception as exc:
            record.update(
                status="failed",
                elapsed_s=round(time.perf_counter() - started, 6),
                error=f"{type(exc).__name__}: {exc}",
            )
        atomic_write_json(
            args.output_dir / f"follower-{index:03d}.json", record
        )
        return record

    baseline_usage = kv_cache_usage(args.base_url, args.api_key)
    sampler = KVUsageSampler(
        args.base_url, args.api_key, interval_s=args.sample_interval_s
    )
    sampler.start()
    wall_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.followers) as pool:
        futures = [pool.submit(send_one, index) for index in range(args.followers)]
        barrier.wait()
        completed = [future.result() for future in futures]
    wall_s = round(time.perf_counter() - wall_started, 6)
    samples = sampler.stop()

    failures = [record for record in completed if record["status"] != "completed"]
    latencies_ms = [record["elapsed_s"] * 1000 for record in completed]
    usage_values = [sample["usage"] for sample in samples]
    peak_usage = max(usage_values) if usage_values else None
    summary = {
        "schema_version": 1,
        "mode": args.mode,
        "followers": args.followers,
        "completed": args.followers - len(failures),
        "failed": len(failures),
        "wall_s": wall_s,
        "throughput_req_s": round((args.followers - len(failures)) / wall_s, 6),
        "latency_p50_ms": round(statistics.median(latencies_ms), 3),
        "latency_p95_ms": round(percentile(latencies_ms, 0.95), 3),
        "kv_usage_before": baseline_usage,
        "kv_usage_peak": peak_usage,
        "kv_usage_peak_delta": (
            round(peak_usage - baseline_usage, 6)
            if peak_usage is not None and baseline_usage is not None
            else None
        ),
        "kv_usage_samples": samples,
        "request_ids": [record["request_id"] for record in completed],
        "response_ids": [record.get("response_id") for record in completed],
    }
    atomic_write_json(args.output_dir / "manifest.json", summary)
    if failures:
        failed_ids = ", ".join(record["request_id"] for record in failures)
        raise RuntimeError(f"requests failed without retry: {failed_ids}")
    print(
        f"[point] mode={args.mode} followers={args.followers} "
        f"wall_s={wall_s} kv_peak_delta={summary['kv_usage_peak_delta']}"
    )


if __name__ == "__main__":
    main()
