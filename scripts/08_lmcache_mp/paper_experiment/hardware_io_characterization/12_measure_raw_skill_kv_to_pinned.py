#!/usr/bin/env python3
"""Measure one io_uring batch from raw Skill KV storage to pinned CPU."""

from __future__ import annotations

import csv
import io
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/segmentia-mpl")
import matplotlib.pyplot as plt

from common import (
    atomic_write_text,
    gib_per_second,
    load_config,
    percentile,
    require_mounted_device,
    result_dir,
    write_test_result,
)
from raw_skill_kv_common import (
    build_layout,
    discover_raw_sources,
    key_spec,
    memory_object,
    open_core,
    sha256_bytes,
)


OUTPUT_STEM = "raw_skill_kv_to_pinned"


def load_manifest(path: Path, layout: dict[str, Any]) -> dict[str, Any]:
    """Load and validate the completed offline conversion manifest."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise RuntimeError(f"raw Skill KV conversion is incomplete: {path}")
    if payload.get("layout") != layout:
        raise RuntimeError("raw Skill KV manifest layout disagrees with config")
    return payload


def make_pinned_objects(source, pinned: torch.Tensor) -> list[Any]:
    """Map one Skill's layers onto distinct persistent pinned pages."""
    return [
        memory_object(pinned[index, : layer.size_bytes], layer)
        for index, layer in enumerate(source.layers)
    ]


def verify_skill(source, objects: list[Any], manifest: dict[str, Any]) -> None:
    """Compare every loaded pinned layer with its offline source SHA-256."""
    expected = {
        int(layer["layer_id"]): str(layer["sha256"])
        for layer in manifest["skills"][source.skill]["layers"]
    }
    for layer, memory_obj in zip(source.layers, objects, strict=True):
        actual = sha256_bytes(memory_obj.byte_array)
        if actual != expected[layer.layer_id]:
            raise RuntimeError(
                f"SHA-256 mismatch for {source.skill}/layer-{layer.layer_id}"
            )


def read_once(core, source, objects: list[Any]) -> float:
    """Issue and wait for one 40-layer LMCache batched read."""
    started = time.perf_counter_ns()
    loaded = core.load_many_into(
        [key_spec(layer).encoded for layer in source.layers],
        objects,
        raise_on_error=True,
    )
    duration_ms = (time.perf_counter_ns() - started) / 1_000_000
    if loaded != [True] * len(source.layers):
        raise RuntimeError(f"incomplete raw-block load for {source.skill}")
    return duration_ms


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "task",
        "skill",
        "skill_tokens",
        "cache_bytes",
        "p50_ms",
        "p95_ms",
        "p50_gib_s",
        "bare_nvme_time_ms",
    )
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows({field: row[field] for field in fields} for row in rows)
    atomic_write_text(path, buffer.getvalue())


def update_source_manifest(path: Path, raw_result: Path) -> None:
    """Register every lightweight paper artifact and its raw-data source."""
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


def plot(rows: list[dict[str, Any]], figure_dir: Path) -> None:
    ordered = sorted(rows, key=lambda row: int(row["skill_tokens"]))
    positions = list(range(len(ordered)))
    values = [float(row["p50_ms"]) for row in ordered]
    ideals = [float(row["bare_nvme_time_ms"]) for row in ordered]
    figure, axis = plt.subplots(figsize=(10.8, 5.8))
    height = 0.36
    axis.barh(
        [value - height / 2 for value in positions],
        values,
        height=height,
        label="Measured raw-block to pinned CPU",
        color="#4F772D",
        edgecolor="black",
        linewidth=0.8,
    )
    axis.barh(
        [value + height / 2 for value in positions],
        ideals,
        height=height,
        label="Ideal time at 6.926 GiB/s",
        color="#D9ED92",
        edgecolor="black",
        linewidth=0.8,
    )
    axis.set_yticks(positions, [str(row["skill"]) for row in ordered])
    axis.set_xlabel("SSD to pinned CPU latency (ms)")
    axis.grid(axis="x", alpha=0.25)
    axis.set_axisbelow(True)
    axis.legend(frameon=False)
    figure.tight_layout()
    for suffix, dpi in (("pdf", None), ("png", 300)):
        figure.savefig(
            figure_dir / f"{OUTPUT_STEM}.{suffix}",
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(figure)


def main() -> None:
    config = load_config()
    settings = config["raw_skill_kv"]
    fast_ssd = config["fast_ssd_skill_cache"]
    mount = require_mounted_device(
        Path(fast_ssd["mount_point"]),
        Path(fast_ssd["expected_source"]),
        writable=True,
    )
    sources = discover_raw_sources(config)
    layout = build_layout(config, sources)
    manifest_path = Path(settings["manifest"]).resolve()
    manifest = load_manifest(manifest_path, layout)
    maximum_layer_bytes = int(layout["maximum_layer_bytes"])
    layer_count = int(layout["expected_layers"])
    warmups = int(settings["warmups"])
    repetitions = int(settings["repetitions"])
    if warmups < 0 or repetitions <= 0:
        raise ValueError("invalid raw-block measurement counts")

    allocation_started = time.perf_counter_ns()
    pinned = torch.empty(
        (layer_count, maximum_layer_bytes),
        dtype=torch.uint8,
        pin_memory=True,
    )
    alignment = int(layout["block_alignment_bytes"])
    misaligned = [
        index
        for index in range(layer_count)
        if pinned[index].data_ptr() % alignment != 0
    ]
    if misaligned:
        raise RuntimeError(
            f"pinned raw-block buffers are not {alignment}-byte aligned: {misaligned}"
        )
    allocation_ms = (time.perf_counter_ns() - allocation_started) / 1_000_000
    core = open_core(layout)
    registration_started = time.perf_counter_ns()
    core.raw_device().register_fixed_buffers(
        [pinned[index].data_ptr() for index in range(layer_count)],
        [maximum_layer_bytes] * layer_count,
    )
    registration_ms = (time.perf_counter_ns() - registration_started) / 1_000_000

    measurements: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    try:
        for source in sources:
            objects = make_pinned_objects(source, pinned)
            for _ in range(warmups):
                read_once(core, source, objects)
            durations = []
            for repetition in range(repetitions):
                duration_ms = read_once(core, source, objects)
                durations.append(duration_ms)
                measurements.append(
                    {
                        "task": source.task,
                        "skill": source.skill,
                        "repetition": repetition,
                        "duration_ms": duration_ms,
                    }
                )
            verify_skill(source, objects, manifest)
            p50_ms = statistics.median(durations)
            p95_ms = percentile(durations, 0.95)
            bare_nvme_time_ms = (
                source.cache_bytes
                / (float(settings["bare_nvme_gib_s"]) * 1024**3)
                * 1000
            )
            summary = {
                "task": source.task,
                "skill": source.skill,
                "skill_tokens": source.token_count,
                "cache_bytes": source.cache_bytes,
                "p50_ms": p50_ms,
                "p95_ms": p95_ms,
                "p50_gib_s": gib_per_second(source.cache_bytes, p50_ms),
                "bare_nvme_time_ms": bare_nvme_time_ms,
            }
            summaries.append(summary)
            print(
                f"[completed] {source.skill} p50={p50_ms:.3f}ms "
                f"bandwidth={summary['p50_gib_s']:.3f}GiB/s"
            )
    finally:
        core.close()

    raw_result = write_test_result(
        "12_raw_skill_kv_to_pinned",
        config,
        {
            "mount": mount,
            "layout": layout,
            "raw_manifest": str(manifest_path),
            "execution_policy": {
                "io_requests_per_skill": layer_count,
                "io_batches_per_skill": 1,
                "odirect": True,
                "fixed_buffers_registered": layer_count,
                "pinned_pool_allocation_ms_excluded": allocation_ms,
                "fixed_buffer_registration_ms_excluded": registration_ms,
                "warmups": warmups,
                "repetitions": repetitions,
                "agent_vllm_h2d_correction_excluded": True,
            },
            "cases": summaries,
            "measurements": measurements,
        },
    )
    output = result_dir(config)
    table_dir = output / "tables"
    figure_dir = output / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    write_csv(table_dir / f"{OUTPUT_STEM}.csv", summaries)
    plot(summaries, figure_dir)
    update_source_manifest(output / "source_manifest.csv", raw_result)
    print(f"[done] {raw_result}")
    print(f"[plotted] {figure_dir / f'{OUTPUT_STEM}.pdf'}")


if __name__ == "__main__":
    main()
