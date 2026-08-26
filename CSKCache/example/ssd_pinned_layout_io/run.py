#!/usr/bin/env python3
"""Measure the 4-layout x 2-engine x 2-open-mode SSD-to-pinned matrix."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Sequence

import torch

import config as cfg
from common import (
    LAYOUTS,
    BenchmarkCase,
    artifact_for_layout,
    atomic_write_json,
    build_case_matrix,
    read_manifest,
    require_writable_data_mount,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
LMCACHE_ROOT = REPOSITORY_ROOT / "LMCache"
if str(LMCACHE_ROOT) not in sys.path:
    sys.path.insert(0, str(LMCACHE_ROOT))


def make_memory_obj(raw_data: torch.Tensor) -> Any:
    from lmcache.v1.memory_management import (
        MemoryFormat,
        MemoryObjMetadata,
        TensorMemoryObj,
    )

    metadata = MemoryObjMetadata(
        shape=torch.Size([raw_data.numel()]),
        dtype=torch.uint8,
        address=raw_data.data_ptr(),
        phy_size=raw_data.numel(),
        ref_count=1,
        fmt=MemoryFormat.BINARY,
        shapes=[torch.Size([raw_data.numel()])],
        dtypes=[torch.uint8],
    )
    return TensorMemoryObj(raw_data, metadata, parent_allocator=None)


def allocate_aligned_pinned_arena(
    total_bytes: int, alignment: int
) -> tuple[torch.Tensor, torch.Tensor]:
    backing = torch.empty(
        total_bytes + alignment - 1, dtype=torch.uint8, pin_memory=True
    )
    shift = (-backing.data_ptr()) % alignment
    arena = backing[shift : shift + total_bytes]
    if arena.data_ptr() % alignment:
        raise AssertionError("failed to align pinned destination arena")
    return backing, arena


def make_region_objects(
    arena: torch.Tensor, lengths: Sequence[int], alignment: int
) -> list[Any]:
    objects = []
    cursor = 0
    for length in lengths:
        raw = arena[cursor : cursor + int(length)]
        if raw.data_ptr() % alignment:
            raise RuntimeError("a pinned layout region is not block aligned")
        objects.append(make_memory_obj(raw))
        cursor += int(length)
    if cursor != arena.numel():
        raise ValueError("region lengths do not consume the pinned arena")
    return objects


def open_core(case: BenchmarkCase, artifact: Any) -> Any:
    from lmcache.v1.storage_backend.raw_block import RawBlockCore, RawBlockCoreConfig

    return RawBlockCore(
        RawBlockCoreConfig(
            device_path=artifact.raw_file,
            capacity_bytes=artifact.capacity_bytes,
            block_align=artifact.alignment_bytes,
            header_bytes=artifact.header_bytes,
            slot_bytes=artifact.slot_bytes,
            use_odirect=case.use_odirect,
            enable_zero_copy=True,
            meta_total_bytes=artifact.metadata_bytes,
            meta_magic=b"LMCIDX01",
            meta_version=1,
            meta_checkpoint_interval_sec=3600,
            meta_idle_quiet_ms=0,
            meta_enable_periodic=False,
            meta_verify_on_load=False,
            load_checkpoint_on_init=False,
            io_engine=case.io_engine,
            iouring_queue_depth=cfg.IO_URING_QUEUE_DEPTH,
            use_uring_cmd=False,
        ),
        key_namespace="legacy",
    )


def evict_file_pages(path: Path) -> None:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        raise RuntimeError("buffered SSD cases require POSIX_FADV_DONTNEED")
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def execute_read(core: Any, artifact: Any, objects: Sequence[Any]) -> float:
    offsets = [extent.offset_bytes for extent in artifact.extents]
    lengths = [extent.length_bytes for extent in artifact.extents]
    started = time.perf_counter_ns()
    result = core.read_extents_into(offsets, lengths, objects)
    ended = time.perf_counter_ns()
    if result != [True] * len(objects):
        raise RuntimeError("raw-block extent batch was incomplete")
    return (ended - started) / 1_000_000


def verify_objects(objects: Sequence[Any], expected: Sequence[str]) -> None:
    actual = [
        hashlib.sha256(memoryview(obj.byte_array).cast("B")).hexdigest()
        for obj in objects
    ]
    if actual != list(expected):
        raise RuntimeError("SSD-to-pinned payload SHA-256 mismatch")


def measure_case(
    case: BenchmarkCase,
    *,
    arena: torch.Tensor,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    artifact = artifact_for_layout(cfg, case.layout)
    lengths = [extent.length_bytes for extent in artifact.extents]
    objects = make_region_objects(arena, lengths, artifact.alignment_bytes)
    expected_hashes = [
        str(record["payload_sha256"])
        for record in sorted(manifest["regions"], key=lambda row: row["region_id"])
    ]
    core = open_core(case, artifact)
    try:
        for _ in range(cfg.WARMUPS):
            if not case.use_odirect:
                evict_file_pages(Path(artifact.raw_file))
            execute_read(core, artifact, objects)
        samples = []
        for repetition in range(cfg.REPETITIONS):
            if not case.use_odirect:
                evict_file_pages(Path(artifact.raw_file))
            duration_ms = execute_read(core, artifact, objects)
            gib_s = artifact.payload_bytes / 1024**3 / (duration_ms / 1000)
            samples.append(
                {
                    "case_id": case.case_id,
                    "layout": case.layout,
                    "io_engine": case.io_engine,
                    "use_odirect": case.use_odirect,
                    "repetition": repetition,
                    "duration_ms": duration_ms,
                    "gib_per_second": gib_s,
                    "payload_bytes": artifact.payload_bytes,
                    "region_count": len(artifact.extents),
                    "queue_depth": (
                        cfg.IO_URING_QUEUE_DEPTH
                        if case.io_engine == "io_uring"
                        else None
                    ),
                    "io_uring_buffer_mode": (
                        "non_fixed" if case.io_engine == "io_uring" else None
                    ),
                    "buffered_cache_control": (
                        None if case.use_odirect else "POSIX_FADV_DONTNEED"
                    ),
                }
            )
        verify_objects(objects, expected_hashes)
        return samples
    finally:
        core.close()


def main() -> None:
    require_writable_data_mount(cfg)
    manifests = {layout: read_manifest(cfg, layout) for layout in LAYOUTS}
    run_name = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    output_dir = Path(cfg.OUTPUT_ROOT) / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    payload_bytes = artifact_for_layout(cfg, build_case_matrix()[0].layout).payload_bytes
    _backing, arena = allocate_aligned_pinned_arena(payload_bytes, cfg.ALIGNMENT_BYTES)
    cases = list(build_case_matrix())
    random.Random(cfg.CASE_ORDER_SEED).shuffle(cases)
    samples: list[dict[str, Any]] = []
    raw_path = output_dir / "samples.json"
    for position, case in enumerate(cases):
        print(f"[case {position + 1}/16] {case.case_id}")
        samples.extend(
            measure_case(case, arena=arena, manifest=manifests[case.layout])
        )
        atomic_write_json(
            raw_path,
            {
                "status": "running",
                "benchmark": {
                    "warmups": cfg.WARMUPS,
                    "repetitions": cfg.REPETITIONS,
                    "io_uring_queue_depth": cfg.IO_URING_QUEUE_DEPTH,
                    "case_order_seed": cfg.CASE_ORDER_SEED,
                    "buffered_cache_control": "POSIX_FADV_DONTNEED",
                    "expected_device": cfg.EXPECTED_DEVICE,
                    "token_count": cfg.TOKEN_COUNT,
                    "io_uring_buffer_mode": "non_fixed",
                },
                "case_order": [item.case_id for item in cases],
                "samples": samples,
            },
        )
    atomic_write_json(
        raw_path,
        {
            "status": "completed",
            "benchmark": {
                "warmups": cfg.WARMUPS,
                "repetitions": cfg.REPETITIONS,
                "io_uring_queue_depth": cfg.IO_URING_QUEUE_DEPTH,
                "case_order_seed": cfg.CASE_ORDER_SEED,
                "buffered_cache_control": "POSIX_FADV_DONTNEED",
                "expected_device": cfg.EXPECTED_DEVICE,
                "token_count": cfg.TOKEN_COUNT,
                "io_uring_buffer_mode": "non_fixed",
            },
            "case_order": [item.case_id for item in cases],
            "samples": samples,
        },
    )
    from analyze import analyze

    analyze(raw_path)
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
