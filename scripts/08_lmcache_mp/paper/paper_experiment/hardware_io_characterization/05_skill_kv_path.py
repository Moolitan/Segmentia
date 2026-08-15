#!/usr/bin/env python3
"""Measure the serial real-file SSD/page-cache → pinned CPU → GPU path."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import torch

from common import (
    LayerFile,
    discover_layers,
    gib_per_second,
    load_config,
    metadata_for_layers,
    write_test_result,
)


def evict_file(path: Path) -> None:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        raise RuntimeError("POSIX_FADV_DONTNEED is unavailable on this platform")
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def read_into(layer: LayerFile, destination: memoryview) -> int:
    offset = 0
    with layer.path.open("rb", buffering=0) as handle:
        while offset < layer.size_bytes:
            count = handle.readinto(destination[offset:layer.size_bytes])
            if not count:
                break
            offset += count
    if offset != layer.size_bytes:
        raise IOError(f"short read for layer {layer.layer_id}: {offset}/{layer.size_bytes}")
    return offset


def warm_page_cache(layers: list[LayerFile], buffer: memoryview) -> None:
    for layer in layers:
        read_into(layer, buffer)


def run_once(
    state: str,
    layers: list[LayerFile],
    pinned: torch.Tensor,
    destination: torch.Tensor,
) -> dict[str, Any]:
    if state == "cold":
        for layer in layers:
            evict_file(layer.path)
    pinned_view = memoryview(pinned.numpy())
    disk_ms = 0.0
    h2d_wall_ms = 0.0
    h2d_cuda_ms = 0.0
    checksum_guard = 0
    total_started = time.perf_counter_ns()
    for layer in layers:
        read_started = time.perf_counter_ns()
        read_into(layer, pinned_view)
        disk_ms += (time.perf_counter_ns() - read_started) / 1_000_000
        checksum_guard += int(pinned[0]) + int(pinned[layer.size_bytes - 1])
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        copy_started = time.perf_counter_ns()
        start_event.record()
        destination[: layer.size_bytes].copy_(
            pinned[: layer.size_bytes], non_blocking=True
        )
        end_event.record()
        end_event.synchronize()
        h2d_wall_ms += (time.perf_counter_ns() - copy_started) / 1_000_000
        h2d_cuda_ms += float(start_event.elapsed_time(end_event))
    total_ms = (time.perf_counter_ns() - total_started) / 1_000_000
    total_bytes = sum(layer.size_bytes for layer in layers)
    return {
        "state": state,
        "bytes": total_bytes,
        "total_ms": total_ms,
        "disk_read_ms": disk_ms,
        "h2d_wall_ms": h2d_wall_ms,
        "h2d_cuda_ms": h2d_cuda_ms,
        "other_ms": max(0.0, total_ms - disk_ms - h2d_wall_ms),
        "path_bandwidth_gib_s": gib_per_second(total_bytes, total_ms),
        "checksum_guard": checksum_guard,
    }


def main() -> None:
    config = load_config()
    manifest, layers = discover_layers(config)
    settings = config["end_to_end"]
    device_index = int(config["pcie"]["device"])
    repetitions = int(settings["repetitions"])
    warmups = int(settings["warmups"])
    states = [str(value) for value in settings["disk_states"]]
    if repetitions <= 0 or warmups < 0 or not states:
        raise ValueError("invalid end-to-end benchmark configuration")
    if any(state not in {"cold", "warm"} for state in states):
        raise ValueError("end_to_end.disk_states only supports cold and warm")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; the complete Skill KV path cannot run")
    torch.cuda.set_device(device_index)
    maximum_layer_bytes = max(layer.size_bytes for layer in layers)
    pinned = torch.empty(maximum_layer_bytes, dtype=torch.uint8, pin_memory=True)
    destination = torch.empty(
        maximum_layer_bytes, dtype=torch.uint8, device=f"cuda:{device_index}"
    )
    pinned_view = memoryview(pinned.numpy())
    destination.copy_(pinned, non_blocking=True)
    torch.cuda.synchronize()
    rows: list[dict[str, Any]] = []
    for state in states:
        if state == "warm":
            warm_page_cache(layers, pinned_view)
        for _ in range(warmups):
            run_once(state, layers, pinned, destination)
        for repetition in range(repetitions):
            row = run_once(state, layers, pinned, destination)
            row["repetition"] = repetition
            rows.append(row)
    path = write_test_result(
        "05_skill_kv_path",
        config,
        {
            "skill_cache": metadata_for_layers(manifest, layers),
            "path_definition": "serial layer-by-layer file read into one pinned buffer followed by synchronized async H2D into one GPU layer buffer",
            "cold_definition": "POSIX_FADV_DONTNEED on each target layer before the timed path",
            "measurements": rows,
        },
    )
    print(f"[completed] {path}")


if __name__ == "__main__":
    main()
