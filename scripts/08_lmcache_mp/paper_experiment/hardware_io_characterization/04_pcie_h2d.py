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
    # CUDA kernel/copy launch is asynchronous from the CPU's point of view.  If we
    # only put perf_counter() around destination.copy_(), an asynchronous copy can
    # return after the command is queued but before the bytes reach GPU memory.
    # CUDA events live on the GPU stream, so elapsed_time() measures the interval
    # seen by the GPU.  wall_ms additionally includes the small Python/API cost.
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_started = time.perf_counter_ns()
    start_event.record()
    for _ in range(copies):
        # non_blocking=False makes each copy wait before Python proceeds.  With a
        # pinned source and non_blocking=True, all copies can first be placed on
        # the CUDA stream and are waited for together below.  Pageable memory is
        # not tested with non_blocking=True because CUDA must first stage it in
        # page-locked memory; that would not represent a true asynchronous H2D
        # path controlled by this program.
        destination.copy_(source, non_blocking=non_blocking)
    end_event.record()

    # This synchronization is essential: it makes both returned times cover the
    # completed transfer.  For pinned_async it synchronizes once per group rather
    # than once per layer, which is exactly the batching benefit being measured.
    end_event.synchronize()
    wall_ms = (time.perf_counter_ns() - wall_started) / 1_000_000
    return wall_ms, float(start_event.elapsed_time(end_event))


def summarize_mode(
    name: str,
    source: torch.Tensor,
    copies: int,
    wall_samples: list[float],
    cuda_samples: list[float],
) -> dict[str, Any]:
    # Effective bandwidth counts the payload once.  PCIe physically transports
    # exactly these H2D bytes, so unlike a CPU memcpy benchmark there is no
    # separate DRAM read+write factor to add to this number.
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


def rotated_modes(
    modes: tuple[tuple[str, torch.Tensor, bool], ...],
    round_index: int,
) -> tuple[tuple[str, torch.Tensor, bool], ...]:
    # The GPU and PCIe link may move from an idle power state to a faster active
    # state while the benchmark runs.  Running all pageable samples first and all
    # pinned samples last would then incorrectly attribute part of this warm-up to
    # the memory type.  A deterministic cyclic rotation gives every mode turns in
    # the first, middle, and last position while keeping the run reproducible:
    #
    #   round 0: pageable_sync, pinned_sync, pinned_async
    #   round 1: pinned_sync, pinned_async, pageable_sync
    #   round 2: pinned_async, pageable_sync, pinned_sync
    #
    # This is ordering control, not concurrent execution: copy_group() completes
    # and synchronizes one mode before the next mode begins.
    offset = round_index % len(modes)
    return modes[offset:] + modes[:offset]


def measure_interleaved(
    modes: tuple[tuple[str, torch.Tensor, bool], ...],
    destination: torch.Tensor,
    copies: int,
    warmups: int,
    repetitions: int,
) -> list[dict[str, Any]]:
    # Warm up every mode in the same rotated schedule used for measurement.  One
    # warm-up round therefore means one completed sample for each mode, not one
    # total copy shared among the three modes.
    for round_index in range(warmups):
        for _, source, non_blocking in rotated_modes(modes, round_index):
            copy_group(source, destination, copies, non_blocking)

    # Samples are keyed by mode name so interleaved execution can still produce
    # one conventional row per mode in the JSON result.  Duplicate names would
    # merge unrelated samples, so reject them rather than silently corrupting the
    # benchmark.
    samples = {
        name: {"wall": [], "cuda": []}
        for name, _, _ in modes
    }
    if len(samples) != len(modes):
        raise ValueError("PCIe benchmark mode names must be unique")

    for round_index in range(repetitions):
        for name, source, non_blocking in rotated_modes(modes, round_index):
            wall_ms, cuda_ms = copy_group(
                source,
                destination,
                copies,
                non_blocking,
            )
            samples[name]["wall"].append(wall_ms)
            samples[name]["cuda"].append(cuda_ms)

    # Preserve the canonical pageable → pinned-sync → pinned-async row order in
    # the output even though the execution order was rotated during measurement.
    return [
        summarize_mode(
            name,
            source,
            copies,
            samples[name]["wall"],
            samples[name]["cuda"],
        )
        for name, source, _ in modes
    ]


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

    # pageable is ordinary CPU memory.  The OS may page it out, and CUDA cannot
    # use it as a stable DMA source, so a pageable H2D normally needs an internal
    # staging step.  pinned is page-locked CPU memory: its physical pages cannot
    # be swapped away while the GPU DMA engine is reading them, enabling a true
    # asynchronous H2D copy and usually higher throughput.
    pageable = torch.empty(layer_bytes, dtype=torch.uint8, device="cpu")
    pageable.fill_(17)
    pinned = torch.empty(layer_bytes, dtype=torch.uint8, device="cpu", pin_memory=True)
    pinned.fill_(29)

    # A single real layer-sized source and destination are reused.  Repeating a
    # 32,854,016-byte copy 40 times transfers the same byte volume as all 40 Skill
    # KV layers without requiring another full 1.3-GiB GPU allocation.  This is a
    # link-throughput microbenchmark; 05_skill_kv_path.py measures real per-layer
    # file reads and the complete serial path.
    destination = torch.empty(layer_bytes, dtype=torch.uint8, device=f"cuda:{device_index}")
    torch.cuda.synchronize()
    rows: list[dict[str, Any]] = []
    modes = (
        ("pageable_sync", pageable, False),
        ("pinned_sync", pinned, False),
        ("pinned_async", pinned, True),
    )
    for copies in layer_groups:
        rows.extend(
            measure_interleaved(
                modes,
                destination,
                copies,
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
            "execution_policy": {
                "mode_order": "cyclic_rotation",
                "concurrent": False,
                "warmup_rounds_per_mode": warmups,
                "measured_rounds_per_mode": repetitions,
            },
            "measurements": rows,
        },
    )
    print(f"[completed] {path}")


if __name__ == "__main__":
    main()
