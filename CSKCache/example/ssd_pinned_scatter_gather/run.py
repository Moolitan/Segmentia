#!/usr/bin/env python3
"""Compare independent reads with true vectored reads into four host layouts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Sequence

import torch

try:
    from . import config as cfg
    from .common import (
        LAYOUTS,
        BenchmarkCase,
        ScatterPlan,
        build_case_matrix,
        build_scatter_plan,
        load_layout_manifest,
    )
except ImportError:
    import config as cfg
    from common import (
        LAYOUTS,
        BenchmarkCase,
        ScatterPlan,
        build_case_matrix,
        build_scatter_plan,
        load_layout_manifest,
    )


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
LMCACHE_ROOT = REPOSITORY_ROOT / "LMCache"
if str(LMCACHE_ROOT) not in sys.path:
    sys.path.insert(0, str(LMCACHE_ROOT))


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def resolve_mount(path: Path) -> tuple[str, Path, set[str]]:
    resolved = path.resolve()
    matches = []
    for line in Path("/proc/self/mounts").read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        source, mount_raw, _fstype, options = fields[:4]
        mount_point = Path(mount_raw.replace("\\040", " ")).resolve()
        try:
            resolved.relative_to(mount_point)
        except ValueError:
            continue
        matches.append((source, mount_point, set(options.split(","))))
    if not matches:
        raise RuntimeError(f"no mount contains {resolved}")
    return max(matches, key=lambda item: len(item[1].parts))


def require_writable_data_mount() -> None:
    source, mount_point, options = resolve_mount(Path(cfg.DATA_ROOT))
    if mount_point != Path(cfg.DATA_MOUNT).resolve():
        raise RuntimeError(f"DATA_ROOT is not on the configured NVMe mount: {mount_point}")
    if Path(source).resolve() != Path(cfg.EXPECTED_DEVICE).resolve():
        raise RuntimeError(f"DATA_ROOT is backed by {source}, expected {cfg.EXPECTED_DEVICE}")
    if "rw" not in options:
        raise RuntimeError(f"{mount_point} must be mounted read-write")


def allocate_aligned_pinned_arena(total_bytes: int) -> tuple[torch.Tensor, torch.Tensor]:
    backing = torch.empty(
        total_bytes + cfg.ALIGNMENT_BYTES - 1,
        dtype=torch.uint8,
        pin_memory=True,
    )
    shift = (-backing.data_ptr()) % cfg.ALIGNMENT_BYTES
    arena = backing[shift : shift + total_bytes]
    if arena.data_ptr() % cfg.ALIGNMENT_BYTES:
        raise AssertionError("failed to align pinned arena")
    return backing, arena


def segment_views(byte_view: memoryview, plan: ScatterPlan) -> list[memoryview]:
    views = [
        byte_view[
            segment.destination_offset : segment.destination_offset
            + segment.length_bytes
        ]
        for segment in plan.segments
    ]
    if any(len(view) != segment.length_bytes for view, segment in zip(views, plan.segments)):
        raise AssertionError("a scatter view has the wrong length")
    return views


def open_device(case: BenchmarkCase, source_artifact: dict[str, Any]) -> Any:
    from lmcache_rust_raw_block_io import RawBlockDevice

    return RawBlockDevice(
        str(source_artifact["raw_file"]),
        writable=True,
        use_odirect=case.use_odirect,
        alignment=cfg.ALIGNMENT_BYTES,
        io_engine=case.io_engine,
        iouring_queue_depth=cfg.IO_URING_QUEUE_DEPTH,
    )


def require_vectored_extension() -> None:
    from lmcache_rust_raw_block_io import RawBlockDevice

    missing = [
        name for name in ("preadv_into", "readv_uring") if not hasattr(RawBlockDevice, name)
    ]
    if missing:
        raise RuntimeError(
            "lmcache_rust_raw_block_io must be rebuilt; missing methods: "
            + ", ".join(missing)
        )


def evict_file_pages(path: Path) -> None:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        raise RuntimeError("buffered cases require POSIX_FADV_DONTNEED")
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def execute_case(
    case: BenchmarkCase,
    *,
    device: Any,
    source_payload_offset: int,
    plan: ScatterPlan,
    views: Sequence[memoryview],
) -> float:
    started = time.perf_counter_ns()
    if case.submission_mode == "multi_read":
        offsets = [
            source_payload_offset + segment.source_relative_offset
            for segment in plan.segments
        ]
        lengths = [segment.length_bytes for segment in plan.segments]
        if case.io_engine == "posix":
            for offset, view, length in zip(offsets, views, lengths, strict=True):
                device.pread_into(offset, view, length, length)
        else:
            batch_id = device.batched_read(offsets, list(views), lengths)
            device.wait_iouring(batch_id)
    elif case.submission_mode == "readv":
        view_index = 0
        for group in plan.vector_groups:
            count = len(group.segments)
            group_views = list(views[view_index : view_index + count])
            lengths = [segment.length_bytes for segment in group.segments]
            offset = source_payload_offset + group.source_relative_offset
            if case.io_engine == "posix":
                device.preadv_into(offset, group_views, lengths)
            else:
                device.readv_uring(offset, group_views, lengths)
            view_index += count
        if view_index != len(views):
            raise AssertionError("readv groups did not consume every destination view")
    else:
        raise ValueError(f"unsupported submission mode: {case.submission_mode}")
    return (time.perf_counter_ns() - started) / 1_000_000


def verify_regions(
    byte_view: memoryview,
    plan: ScatterPlan,
    manifest: dict[str, Any],
) -> None:
    expected = [
        str(record["payload_sha256"])
        for record in sorted(manifest["regions"], key=lambda row: int(row["region_id"]))
    ]
    actual = []
    for offset, length in zip(plan.region_offsets, plan.region_lengths, strict=True):
        actual.append(hashlib.sha256(byte_view[offset : offset + length]).hexdigest())
    if actual != expected:
        raise RuntimeError(f"scatter result SHA-256 mismatch for {plan.layout}")


def measure_case(
    case: BenchmarkCase,
    *,
    byte_view: memoryview,
    source_payload_offset: int,
    source_artifact: dict[str, Any],
    plan: ScatterPlan,
    target_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    views = segment_views(byte_view, plan)
    source_path = Path(str(source_artifact["raw_file"]))
    device = open_device(case, source_artifact)
    try:
        for _ in range(cfg.WARMUPS):
            if not case.use_odirect:
                evict_file_pages(source_path)
            execute_case(
                case,
                device=device,
                source_payload_offset=source_payload_offset,
                plan=plan,
                views=views,
            )
        samples = []
        for repetition in range(cfg.REPETITIONS):
            if not case.use_odirect:
                evict_file_pages(source_path)
            duration_ms = execute_case(
                case,
                device=device,
                source_payload_offset=source_payload_offset,
                plan=plan,
                views=views,
            )
            request_count = (
                len(plan.segments)
                if case.submission_mode == "multi_read"
                else len(plan.vector_groups)
            )
            samples.append(
                {
                    "case_id": case.case_id,
                    "layout": case.layout,
                    "io_engine": case.io_engine,
                    "use_odirect": case.use_odirect,
                    "submission_mode": case.submission_mode,
                    "repetition": repetition,
                    "duration_ms": duration_ms,
                    "gib_per_second": (
                        plan.payload_bytes / 1024**3 / (duration_ms / 1000)
                    ),
                    "payload_bytes": plan.payload_bytes,
                    "region_count": len(plan.region_lengths),
                    "segment_count": len(plan.segments),
                    "request_count": request_count,
                    "iovec_group_count": len(plan.vector_groups),
                    "maximum_iovecs_per_call": cfg.MAX_IOVECS_PER_CALL,
                    "buffered_cache_control": (
                        None if case.use_odirect else "POSIX_FADV_DONTNEED"
                    ),
                }
            )
        verify_regions(byte_view, plan, target_manifest)
        return samples
    finally:
        device.close()


def main() -> None:
    require_writable_data_mount()
    require_vectored_extension()
    manifests = {layout: load_layout_manifest(cfg, layout) for layout in LAYOUTS}
    source_manifest = manifests[cfg.SOURCE_LAYOUT]
    source_artifact = source_manifest["layout_artifact"]
    source_regions = source_manifest["regions"]
    if len(source_regions) != 1:
        raise RuntimeError("packed-all source must contain exactly one physical region")
    source_payload_offset = int(source_regions[0]["offset_bytes"])
    plans = {layout: build_scatter_plan(cfg, layout) for layout in LAYOUTS}
    payload_bytes = next(iter(plans.values())).payload_bytes
    if {plan.payload_bytes for plan in plans.values()} != {payload_bytes}:
        raise AssertionError("scatter plans disagree on payload size")

    _backing, arena = allocate_aligned_pinned_arena(payload_bytes)
    byte_view = memoryview(arena.numpy()).cast("B")
    cases = list(build_case_matrix())
    random.Random(cfg.CASE_ORDER_SEED).shuffle(cases)
    run_name = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    output_dir = Path(cfg.OUTPUT_ROOT) / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    samples: list[dict[str, Any]] = []
    samples_path = output_dir / "samples.json"
    benchmark = {
        "token_count": cfg.TOKEN_COUNT,
        "payload_bytes": payload_bytes,
        "source_layout": cfg.SOURCE_LAYOUT,
        "warmups": cfg.WARMUPS,
        "repetitions": cfg.REPETITIONS,
        "io_uring_queue_depth": cfg.IO_URING_QUEUE_DEPTH,
        "maximum_iovecs_per_call": cfg.MAX_IOVECS_PER_CALL,
        "io_uring_buffer_mode": "non_fixed",
        "case_order_seed": cfg.CASE_ORDER_SEED,
    }
    for position, case in enumerate(cases):
        print(f"[case {position + 1}/32] {case.case_id}")
        samples.extend(
            measure_case(
                case,
                byte_view=byte_view,
                source_payload_offset=source_payload_offset,
                source_artifact=source_artifact,
                plan=plans[case.layout],
                target_manifest=manifests[case.layout],
            )
        )
        atomic_write_json(
            samples_path,
            {
                "status": "running",
                "benchmark": benchmark,
                "case_order": [item.case_id for item in cases],
                "samples": samples,
            },
        )
    atomic_write_json(
        samples_path,
        {
            "status": "completed",
            "benchmark": benchmark,
            "case_order": [item.case_id for item in cases],
            "samples": samples,
        },
    )
    try:
        from .analyze import analyze
    except ImportError:
        from analyze import analyze

    analyze(samples_path)
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
