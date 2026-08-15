#!/usr/bin/env python3
"""Measure complete-Skill KV loading from the 990 PRO through pinned CPU.

This is a standalone data-movement microbenchmark. It does not start an Agent,
vLLM, or LMCache. The pinned arena is allocated once before any timed region;
all 40 layers of one Skill are then read into distinct arena slices so the
measurement mirrors a LocalCPUBackend hot-cache admission rather than a
single-layer staging buffer that is immediately reused.
"""
from __future__ import annotations

import csv
import io
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/segmentia-mpl")
import matplotlib.pyplot as plt

from common import (
    LayerFile,
    atomic_write_text,
    discover_manifest_layers,
    gib_per_second,
    load_config,
    metadata_for_layers,
    require_mounted_device,
    resolve_skill_manifest,
    result_dir,
    write_test_result,
)


OUTPUT_STEM = "skill_kv_loading_990pro"
PATHS = ("pinned_h2d", "warm_load", "cold_load")
CSV_FIELDS = (
    "task",
    "skill",
    "skill_tokens",
    "cache_bytes",
    "pool_bytes_reserved",
    "pool_bytes_used",
    "cold_ssd_to_pinned_ms",
    "cold_pinned_to_gpu_ms",
    "cold_total_ms",
    "warm_page_cache_to_pinned_ms",
    "warm_pinned_to_gpu_ms",
    "warm_total_ms",
    "standalone_pinned_to_gpu_ms",
)


@dataclass(frozen=True)
class ResidentLayer:
    layer: LayerFile
    offset: int


@dataclass(frozen=True)
class SkillCase:
    task: str
    skill: str
    manifest_path: Path
    manifest: dict[str, Any]
    resident_layers: list[ResidentLayer]
    token_count: int
    cache_bytes: int
    pool_bytes_used: int


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def layout_layers(
    layers: list[LayerFile], alignment: int
) -> tuple[list[ResidentLayer], int]:
    cursor = 0
    resident: list[ResidentLayer] = []
    for layer in layers:
        cursor = align_up(cursor, alignment)
        resident.append(ResidentLayer(layer=layer, offset=cursor))
        cursor += layer.size_bytes
    return resident, cursor


def evict_layers(layers: list[ResidentLayer]) -> None:
    if not hasattr(os, "posix_fadvise") or not hasattr(
        os, "POSIX_FADV_DONTNEED"
    ):
        raise RuntimeError("POSIX_FADV_DONTNEED is unavailable")
    for resident in layers:
        descriptor = os.open(resident.layer.path, os.O_RDONLY)
        try:
            os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(descriptor)


def read_layer_into_pool(
    resident: ResidentLayer, pinned_view: memoryview
) -> None:
    layer = resident.layer
    destination = pinned_view[
        resident.offset : resident.offset + layer.size_bytes
    ]
    offset = 0
    with layer.path.open("rb", buffering=0) as handle:
        while offset < layer.size_bytes:
            count = handle.readinto(destination[offset:])
            if not count:
                break
            offset += count
    if offset != layer.size_bytes:
        raise IOError(
            f"short read for layer {layer.layer_id}: "
            f"{offset}/{layer.size_bytes}"
        )


def read_skill_into_pool(
    layers: list[ResidentLayer], pinned_view: memoryview
) -> tuple[float, int]:
    started = time.perf_counter_ns()
    for resident in layers:
        read_layer_into_pool(resident, pinned_view)
    duration_ms = (time.perf_counter_ns() - started) / 1_000_000
    checksum = sum(
        int(pinned_view[resident.offset])
        + int(
            pinned_view[
                resident.offset + resident.layer.size_bytes - 1
            ]
        )
        for resident in layers
    )
    return duration_ms, checksum


def copy_skill_to_gpu(
    layers: list[ResidentLayer],
    pinned: torch.Tensor,
    destination: torch.Tensor,
) -> tuple[float, float]:
    """Copy all layer payloads on one CUDA stream and wait for completion."""
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    started = time.perf_counter_ns()
    start_event.record()
    for resident in layers:
        size = resident.layer.size_bytes
        destination[:size].copy_(
            pinned[resident.offset : resident.offset + size],
            non_blocking=True,
        )
    end_event.record()
    end_event.synchronize()
    wall_ms = (time.perf_counter_ns() - started) / 1_000_000
    return wall_ms, float(start_event.elapsed_time(end_event))


def run_h2d_once(
    case: SkillCase,
    pinned: torch.Tensor,
    destination: torch.Tensor,
) -> dict[str, Any]:
    wall_ms, cuda_ms = copy_skill_to_gpu(
        case.resident_layers, pinned, destination
    )
    return {
        "path": "pinned_h2d",
        "read_ms": 0.0,
        "h2d_wall_ms": wall_ms,
        "h2d_cuda_ms": cuda_ms,
        "total_ms": wall_ms,
        "checksum": None,
    }


def run_load_once(
    state: str,
    case: SkillCase,
    pinned: torch.Tensor,
    pinned_view: memoryview,
    destination: torch.Tensor,
) -> dict[str, Any]:
    if state == "cold":
        # Cache eviction is preparation, not part of the measured read path.
        evict_layers(case.resident_layers)
    total_started = time.perf_counter_ns()
    read_ms, checksum = read_skill_into_pool(
        case.resident_layers, pinned_view
    )
    h2d_wall_ms, h2d_cuda_ms = copy_skill_to_gpu(
        case.resident_layers, pinned, destination
    )
    total_ms = (time.perf_counter_ns() - total_started) / 1_000_000
    return {
        "path": f"{state}_load",
        "read_ms": read_ms,
        "h2d_wall_ms": h2d_wall_ms,
        "h2d_cuda_ms": h2d_cuda_ms,
        "total_ms": total_ms,
        "checksum": checksum,
    }


def configured_pairs(config: dict[str, Any]) -> list[tuple[str, str]]:
    cases = config["agent_schedule"].get("cases")
    settings = config["agent_kv_loading_actual"]
    excluded = {str(value) for value in settings["excluded_skills"]}
    if excluded != {"docx", "writing-systems-papers"}:
        raise ValueError(
            "excluded_skills must contain exactly docx and "
            "writing-systems-papers"
        )
    if not isinstance(cases, list):
        raise ValueError("agent_schedule.cases must be a list")
    pairs = [
        (str(case["task"]), str(case["skill"]))
        for case in cases
        if str(case["skill"]) not in excluded
    ]
    if len(pairs) != 11 or len(set(pairs)) != 11:
        raise ValueError(f"expected 11 unique task/Skill pairs, found {pairs}")
    return pairs


def load_skill_cases(
    config: dict[str, Any], pool_bytes: int, alignment: int
) -> list[SkillCase]:
    pool_dir = Path(config["fast_ssd_skill_cache"]["pool_dir"]).resolve()
    expected_layers = int(config["skill_cache"]["expected_layers"])
    cases: list[SkillCase] = []
    for task, skill in configured_pairs(config):
        manifest_path = resolve_skill_manifest(pool_dir, skill)
        manifest, layers = discover_manifest_layers(
            manifest_path, expected_layers
        )
        token_count = manifest.get("token_count")
        if (
            isinstance(token_count, bool)
            or not isinstance(token_count, int)
            or token_count <= 0
        ):
            raise ValueError(f"invalid token_count in {manifest_path}")
        resident_layers, pool_bytes_used = layout_layers(layers, alignment)
        if pool_bytes_used > pool_bytes:
            raise MemoryError(
                f"{skill} requires {pool_bytes_used} pinned bytes, but the "
                f"configured pool contains only {pool_bytes} bytes"
            )
        cases.append(
            SkillCase(
                task=task,
                skill=skill,
                manifest_path=manifest_path,
                manifest=manifest,
                resident_layers=resident_layers,
                token_count=token_count,
                cache_bytes=sum(layer.size_bytes for layer in layers),
                pool_bytes_used=pool_bytes_used,
            )
        )
    return cases


def measure_case(
    case: SkillCase,
    pinned: torch.Tensor,
    destination: torch.Tensor,
    warmups: int,
    repetitions: int,
) -> list[dict[str, Any]]:
    pinned_view = memoryview(pinned.numpy())

    # Establish valid resident bytes before the H2D-only measurements.
    read_skill_into_pool(case.resident_layers, pinned_view)
    for _ in range(warmups):
        run_h2d_once(case, pinned, destination)
    rows = [
        {**run_h2d_once(case, pinned, destination), "repetition": repetition}
        for repetition in range(repetitions)
    ]

    # Warm reads measure Linux page cache -> the already allocated pinned pool.
    read_skill_into_pool(case.resident_layers, pinned_view)
    for _ in range(warmups):
        run_load_once("warm", case, pinned, pinned_view, destination)
    rows.extend(
        {
            **run_load_once("warm", case, pinned, pinned_view, destination),
            "repetition": repetition,
        }
        for repetition in range(repetitions)
    )

    # Cold reads use a per-file eviction hint before every complete-Skill load.
    for _ in range(warmups):
        run_load_once("cold", case, pinned, pinned_view, destination)
    rows.extend(
        {
            **run_load_once("cold", case, pinned, pinned_view, destination),
            "repetition": repetition,
        }
        for repetition in range(repetitions)
    )

    for path in PATHS:
        selected = [row for row in rows if row["path"] == path]
        if len(selected) != repetitions:
            raise ValueError(f"{case.skill}/{path} has {len(selected)} samples")
        if any(float(row["total_ms"]) <= 0 for row in selected):
            raise ValueError(f"{case.skill}/{path} has a non-positive sample")
        checksums = {
            row["checksum"]
            for row in selected
            if row["checksum"] is not None
        }
        if path != "pinned_h2d" and len(checksums) != 1:
            raise ValueError(f"{case.skill}/{path} checksum is inconsistent")
    return rows


def median_field(rows: list[dict[str, Any]], path: str, field: str) -> float:
    return float(
        statistics.median(
            float(row[field]) for row in rows if row["path"] == path
        )
    )


def summarize_case(
    case: SkillCase, measurements: list[dict[str, Any]], pool_bytes: int
) -> dict[str, Any]:
    cold_read = median_field(measurements, "cold_load", "read_ms")
    cold_h2d = median_field(measurements, "cold_load", "h2d_wall_ms")
    warm_read = median_field(measurements, "warm_load", "read_ms")
    warm_h2d = median_field(measurements, "warm_load", "h2d_wall_ms")
    h2d = median_field(measurements, "pinned_h2d", "total_ms")
    return {
        "task": case.task,
        "skill": case.skill,
        "skill_tokens": case.token_count,
        "cache_bytes": case.cache_bytes,
        "pool_bytes_reserved": pool_bytes,
        "pool_bytes_used": case.pool_bytes_used,
        "cold_ssd_to_pinned_ms": cold_read,
        "cold_pinned_to_gpu_ms": cold_h2d,
        "cold_total_ms": median_field(
            measurements, "cold_load", "total_ms"
        ),
        "warm_page_cache_to_pinned_ms": warm_read,
        "warm_pinned_to_gpu_ms": warm_h2d,
        "warm_total_ms": median_field(
            measurements, "warm_load", "total_ms"
        ),
        "standalone_pinned_to_gpu_ms": h2d,
        "cold_ssd_to_pinned_gib_s": gib_per_second(
            case.cache_bytes, cold_read
        ),
        "warm_page_cache_to_pinned_gib_s": gib_per_second(
            case.cache_bytes, warm_read
        ),
        "standalone_pinned_to_gpu_gib_s": gib_per_second(
            case.cache_bytes, h2d
        ),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row[field] for field in CSV_FIELDS})
    atomic_write_text(path, buffer.getvalue())


def plot_loading(rows: list[dict[str, Any]], figure_dir: Path) -> None:
    ordered = sorted(rows, key=lambda row: int(row["skill_tokens"]))
    panels = (
        ("cold_ssd_to_pinned_ms", "990 PRO to pinned CPU", "#D98E04"),
        (
            "warm_page_cache_to_pinned_ms",
            "Page cache to pinned CPU",
            "#4F772D",
        ),
        ("standalone_pinned_to_gpu_ms", "Pinned CPU to GPU", "#2A6F97"),
    )
    positions = list(range(len(ordered)))
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(17.2, 6.6),
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )
    for axis, (field, label, color) in zip(axes, panels):
        values = [float(row[field]) for row in ordered]
        bars = axis.barh(
            positions,
            values,
            color=color,
            edgecolor="black",
            linewidth=0.8,
        )
        padding = max(values) * 0.015
        for bar, value in zip(bars, values):
            axis.text(
                value + padding,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}",
                va="center",
                fontsize=8.5,
            )
        axis.set_xlabel("Latency (ms)")
        axis.set_title(label)
        axis.set_xlim(0, max(values) * 1.18)
        axis.grid(axis="x", alpha=0.25)
        axis.set_axisbelow(True)
        for spine in axis.spines.values():
            spine.set_visible(True)
    axes[0].set_yticks(
        positions, [str(row["skill"]) for row in ordered]
    )
    figure.subplots_adjust(left=0.18, right=0.99, top=0.91, bottom=0.12)
    for suffix, dpi in (("pdf", None), ("png", 300)):
        figure.savefig(
            figure_dir / f"{OUTPUT_STEM}.{suffix}",
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(figure)


def update_source_manifest(path: Path, raw_result: Path) -> None:
    rows: list[dict[str, str]] = []
    if path.is_file():
        rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    artifacts = {
        f"figures/{OUTPUT_STEM}.pdf",
        f"figures/{OUTPUT_STEM}.png",
        f"tables/{OUTPUT_STEM}.csv",
    }
    rows = [row for row in rows if row.get("artifact") not in artifacts]
    rows.extend(
        {"artifact": artifact, "source": str(raw_result)}
        for artifact in sorted(artifacts)
    )
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=("artifact", "source"))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())


def main() -> None:
    config = load_config()
    settings = config["agent_kv_loading_actual"]
    fast_ssd = config["fast_ssd_skill_cache"]
    mount = require_mounted_device(
        Path(fast_ssd["mount_point"]),
        Path(fast_ssd["expected_source"]),
        writable=False,
    )
    warmups = int(settings["warmups"])
    repetitions = int(settings["repetitions"])
    pool_bytes = int(float(settings["pinned_pool_gib"]) * 1024**3)
    alignment = int(settings["alignment_bytes"])
    if warmups < 0 or repetitions <= 0:
        raise ValueError("invalid loading benchmark repetition counts")
    if pool_bytes <= 0 or alignment <= 0:
        raise ValueError("pinned pool size and alignment must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; KV loading cannot run")

    cases = load_skill_cases(config, pool_bytes, alignment)
    device_index = int(config["pcie"]["device"])
    torch.cuda.set_device(device_index)
    maximum_layer_bytes = max(
        resident.layer.size_bytes
        for case in cases
        for resident in case.resident_layers
    )

    allocation_started = time.perf_counter_ns()
    pinned = torch.empty(pool_bytes, dtype=torch.uint8, pin_memory=True)
    pinned_pool_allocation_ms = (
        time.perf_counter_ns() - allocation_started
    ) / 1_000_000
    destination = torch.empty(
        maximum_layer_bytes,
        dtype=torch.uint8,
        device=f"cuda:{device_index}",
    )

    all_measurements: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for case in cases:
        print(
            f"[measure] {case.skill} tokens={case.token_count} "
            f"bytes={case.cache_bytes}"
        )
        measurements = measure_case(
            case, pinned, destination, warmups, repetitions
        )
        all_measurements.extend(
            {"task": case.task, "skill": case.skill, **row}
            for row in measurements
        )
        summary = summarize_case(case, measurements, pool_bytes)
        summaries.append(summary)
        print(
            f"[completed] {case.skill} "
            f"cold-read={summary['cold_ssd_to_pinned_ms']:.3f}ms "
            f"warm-read={summary['warm_page_cache_to_pinned_ms']:.3f}ms "
            f"h2d={summary['standalone_pinned_to_gpu_ms']:.3f}ms"
        )

    device = torch.cuda.get_device_properties(device_index)
    raw_result = write_test_result(
        "08_skill_kv_loading_990pro",
        config,
        {
            "mount": mount,
            "skill_pool_dir": str(Path(fast_ssd["pool_dir"]).resolve()),
            "device": {
                "index": device_index,
                "name": device.name,
                "total_memory_bytes": device.total_memory,
            },
            "execution_policy": {
                "warmups_per_skill_and_path": warmups,
                "measurements_per_skill_and_path": repetitions,
                "concurrent": False,
                "pinned_pool_bytes": pool_bytes,
                "pinned_pool_alignment_bytes": alignment,
                "pinned_pool_allocation_ms_excluded_from_measurement": (
                    pinned_pool_allocation_ms
                ),
                "residency": (
                    "all 40 layers of one Skill occupy distinct slices of one "
                    "process-lifetime pinned arena until that Skill finishes"
                ),
                "gpu_destination": (
                    "one maximum-layer GPU buffer reused in CUDA stream order"
                ),
            },
            "cold_definition": (
                "POSIX_FADV_DONTNEED on all 40 files before every measured "
                "complete-Skill read; best-effort buffered-I/O cold state"
            ),
            "cases": [
                {
                    **summary,
                    "manifest_path": str(case.manifest_path),
                    "skill_cache": metadata_for_layers(
                        case.manifest,
                        [resident.layer for resident in case.resident_layers],
                    ),
                }
                for case, summary in zip(cases, summaries)
            ],
            "measurements": all_measurements,
        },
    )

    output = result_dir(config)
    figure_dir = output / "figures"
    table_dir = output / "tables"
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    write_csv(table_dir / f"{OUTPUT_STEM}.csv", summaries)
    plot_loading(summaries, figure_dir)
    update_source_manifest(output / "source_manifest.csv", raw_result)
    print(f"[completed] {raw_result}")
    print(f"[plotted] {figure_dir / f'{OUTPUT_STEM}.pdf'}")


if __name__ == "__main__":
    main()
