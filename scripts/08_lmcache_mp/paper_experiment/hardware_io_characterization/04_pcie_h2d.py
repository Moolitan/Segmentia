#!/usr/bin/env python3
"""Measure pageable, pinned, synchronous, and asynchronous PCIe H2D copies."""
from __future__ import annotations

import time
from typing import Any

import torch

from common import (
    discover_layers,
    gib_per_second,
    load_config,
    metadata_for_layers,
    percentile,
    write_test_result,
)


def copy_group(
    source: torch.Tensor,
    destination: torch.Tensor,
    copies: int,
    non_blocking: bool,
) -> tuple[float, float]:
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_started = time.perf_counter_ns()
    start_event.record()
    for _ in range(copies):
        destination.copy_(source, non_blocking=non_blocking)
    end_event.record()
    end_event.synchronize()
    wall_ms = (time.perf_counter_ns() - wall_started) / 1_000_000
    return wall_ms, float(start_event.elapsed_time(end_event))


def measure_mode(
    name: str,
    source: torch.Tensor,
    destination: torch.Tensor,
    copies: int,
    non_blocking: bool,
    warmups: int,
    repetitions: int,
) -> dict[str, Any]:
    for _ in range(warmups):
        copy_group(source, destination, copies, non_blocking)
    wall_samples: list[float] = []
    cuda_samples: list[float] = []
    for _ in range(repetitions):
        wall_ms, cuda_ms = copy_group(source, destination, copies, non_blocking)
        wall_samples.append(wall_ms)
        cuda_samples.append(cuda_ms)
    bytes_moved = source.numel() * source.element_size() * copies
    wall_p50 = percentile(wall_samples, 0.50)
    cuda_p50 = percentile(cuda_samples, 0.50)
    return {
        "mode": name,
        "copies": copies,
        "bytes": bytes_moved,
        "wall_p50_ms": wall_p50,
        "wall_p95_ms": percentile(wall_samples, 0.95),
        "cuda_p50_ms": cuda_p50,
        "cuda_p95_ms": percentile(cuda_samples, 0.95),
        "wall_bandwidth_gib_s": gib_per_second(bytes_moved, wall_p50),
        "cuda_bandwidth_gib_s": gib_per_second(bytes_moved, cuda_p50),
        "wall_samples_ms": wall_samples,
        "cuda_samples_ms": cuda_samples,
    }


def main() -> None:
    config = load_config()
    manifest, layers = discover_layers(config)
    settings = config["pcie"]
    device_index = int(settings["device"])
    warmups = int(settings["warmups"])
    repetitions = int(settings["repetitions"])
    layer_groups = [int(value) for value in settings["layer_groups"]]
    if repetitions <= 0 or warmups < 0 or any(value <= 0 for value in layer_groups):
        raise ValueError("invalid PCIe benchmark configuration")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; PCIe H2D cannot be measured")
    torch.cuda.set_device(device_index)
    layer_bytes = max(layer.size_bytes for layer in layers)
    pageable = torch.empty(layer_bytes, dtype=torch.uint8, device="cpu")
    pageable.fill_(17)
    pinned = torch.empty(layer_bytes, dtype=torch.uint8, device="cpu", pin_memory=True)
    pinned.fill_(29)
    destination = torch.empty(layer_bytes, dtype=torch.uint8, device=f"cuda:{device_index}")
    torch.cuda.synchronize()
    rows: list[dict[str, Any]] = []
    modes = (
        ("pageable_sync", pageable, False),
        ("pinned_sync", pinned, False),
        ("pinned_async", pinned, True),
    )
    for copies in layer_groups:
        for name, source, non_blocking in modes:
            rows.append(
                measure_mode(
                    name,
                    source,
                    destination,
                    copies,
                    non_blocking,
                    warmups,
                    repetitions,
                )
            )
    device = torch.cuda.get_device_properties(device_index)
    path = write_test_result(
        "04_pcie_h2d",
        config,
        {
            "skill_cache": metadata_for_layers(manifest, layers),
            "device": {
                "index": device_index,
                "name": device.name,
                "total_memory_bytes": device.total_memory,
                "compute_capability": f"{device.major}.{device.minor}",
            },
            "measurement_note": "40-layer volume reuses one real-layer source and destination buffer for 40 sequential copies",
            "measurements": rows,
        },
    )
    print(f"[completed] {path}")


if __name__ == "__main__":
    main()
