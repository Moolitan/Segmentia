#!/usr/bin/env python3
"""Measure real layer-file reads from SSD or page cache into CPU staging."""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from common import (
    LayerFile,
    discover_layers,
    gib_per_second,
    load_config,
    metadata_for_layers,
    percentile,
    write_test_result,
)


def evict_file(path: Path) -> None:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        raise RuntimeError("POSIX_FADV_DONTNEED is unavailable on this platform")
    fd = os.open(path, os.O_RDONLY)
    try:
        # 尝试将整个 layer 文件从 Linux Page Cache 中移除
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def read_layer(layer: LayerFile) -> dict[str, Any]:
    buffer = bytearray(layer.size_bytes)
    view = memoryview(buffer)
    offset = 0
    started = time.perf_counter_ns()
    with layer.path.open("rb", buffering=0) as handle:
        while offset < layer.size_bytes:
            count = handle.readinto(view[offset:])
            if not count:
                break
            offset += count
    duration_ms = (time.perf_counter_ns() - started) / 1_000_000
    if offset != layer.size_bytes:
        raise IOError(f"short read for layer {layer.layer_id}: {offset}/{layer.size_bytes}")
    checksum = int(buffer[0]) + int(buffer[-1])
    return {
        "layer_id": layer.layer_id,
        "bytes": offset,
        "duration_ms": duration_ms,
        "checksum_guard": checksum,
    }


def read_group(layers: list[LayerFile], threads: int) -> tuple[float, list[dict[str, Any]]]:
    started = time.perf_counter_ns()
    if threads == 1:
        layer_rows = [read_layer(layer) for layer in layers]
    else:
        with ThreadPoolExecutor(max_workers=threads, thread_name_prefix="skill-kv-read") as pool:
            layer_rows = list(pool.map(read_layer, layers))
    wall_ms = (time.perf_counter_ns() - started) / 1_000_000
    return wall_ms, layer_rows


def warm_page_cache(layers: list[LayerFile]) -> None:
    read_group(layers, 1)


def main() -> None:
    config = load_config()
    manifest, layers = discover_layers(config)
    settings = config["ssd"]
    if settings["cold_method"] != "posix_fadvise":
        raise ValueError("ssd.cold_method must be posix_fadvise")
    repetitions = int(settings["repetitions"])
    thread_counts = [int(value) for value in settings["thread_counts"]]
    if repetitions <= 0 or any(value <= 0 for value in thread_counts):
        raise ValueError("SSD repetitions and thread counts must be positive")
    total_bytes = sum(layer.size_bytes for layer in layers)
    rows: list[dict[str, Any]] = []
    for state in ("cold", "warm"):
        if state == "warm":
            warm_page_cache(layers)
        for threads in thread_counts:
            for repetition in range(repetitions):
                if state == "cold":
                    for layer in layers:
                        evict_file(layer.path)
                wall_ms, layer_rows = read_group(layers, threads)
                layer_times = [float(row["duration_ms"]) for row in layer_rows]
                rows.append(
                    {
                        "state": state,
                        "threads": threads,
                        "repetition": repetition,
                        "bytes": total_bytes,
                        "wall_ms": wall_ms,
                        "bandwidth_gib_s": gib_per_second(total_bytes, wall_ms),
                        "layer_p50_ms": percentile(layer_times, 0.50),
                        "layer_p95_ms": percentile(layer_times, 0.95),
                        "checksum_guard": sum(int(row["checksum_guard"]) for row in layer_rows),
                    }
                )
    path = write_test_result(
        "02_ssd_to_cpu",
        config,
        {
            "skill_cache": metadata_for_layers(manifest, layers),
            "cold_definition": "POSIX_FADV_DONTNEED on each target layer before the timed group",
            "measurements": rows,
        },
    )
    print(f"[completed] {path}")


if __name__ == "__main__":
    main()
