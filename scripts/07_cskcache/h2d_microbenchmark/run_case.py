#!/usr/bin/env python3
"""Run one isolated CSKCache H2D microbenchmark condition.

This script measures, in an isolated process, the production CSKCache code
path `VLLMPagedGPUConnector.to_gpu()` — copying one cached KV entry from CPU
to GPU layer by layer, re-rotating RoPE positions when needed, and scattering
the result into a paged KV cache — without starting an agent or the vLLM
scheduler. Three independently togglable experiment factors:
  - --profiling {off,on}: whether LoadTrace records per-stage timings
    (measures the overhead profiling itself adds).
  - --memory {pageable,pinned}: whether the source KV tensors are pinned
    ahead of time (pinning happens once before warmup and is excluded from
    per-iteration timing).
  - --position-shift: offset of the target token position relative to the
    entry's original position when reusing it (0 = reuse in place, skips
    RoPE recomputation; nonzero triggers a relative key re-rotation).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from cskcache.profiling import LoadTrace, NullLoadTrace
from cskcache.v1.kv_transfer.gpu_connector import VLLMPagedGPUConnector
from cskcache.v1.metadata import CSKCacheEntry, CSKCacheMode, CSKLoadPlan
from cskcache.v1.storage import entry_nbytes
from cskcache.v1.storage.local_disk_backend import LocalDiskBackend
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding import get_rope


class _RopeHolder(torch.nn.Module):
    """A fake "model" container that exists only so that
    connector.set_model() can locate the RoPE module via
    find_rotary_embedding() walking .modules(); this script never runs a
    real forward pass, it only needs the RoPE cos/sin cache."""

    def __init__(self, rotary_emb: torch.nn.Module) -> None:
        super().__init__()
        self.rotary_emb = rotary_emb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--kv-dir", type=Path, required=True)
    parser.add_argument("--cache-id", default="doc-coauthoring")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory", choices=("pageable", "pinned"), required=True)
    parser.add_argument("--profiling", choices=("off", "on"), required=True)
    parser.add_argument("--position-shift", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--expected-layers", type=int, default=40)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--max-position", type=int, default=40960)
    parser.add_argument("--rope-theta", type=float, default=1_000_000.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--sample-gpu-clock",
        action="store_true",
        help="Diagnostic only: after each measured iteration (outside the "
        "timed region), shell out to nvidia-smi and attach the GPU's SM "
        "clock/P-state/temperature to that iteration's record. Used to "
        "check whether a within-run speed step coincides with the GPU "
        "leaving its idle power state, not to explain the base benchmark.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.repetitions <= 0:
        raise ValueError("--repetitions must be positive")
    if args.position_shift < 0:
        raise ValueError("--position-shift must be non-negative")
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive")


def sample_gpu_clock(gpu_index: int) -> dict[str, Any]:
    """Shell out to nvidia-smi for a single point-in-time reading of SM
    clock / power state. Meant to be called *outside* the timed region of
    an iteration (it costs several ms itself and would otherwise pollute
    the very numbers being diagnosed)."""
    fields = "clocks.sm,clocks.mem,pstate,temperature.gpu,power.draw"
    output = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    ).stdout.strip()
    sm_clock_mhz, mem_clock_mhz, pstate, temperature_c, power_w = (
        part.strip() for part in output.split(",")
    )
    return {
        "gpu_sm_clock_mhz": int(sm_clock_mhz),
        "gpu_mem_clock_mhz": int(mem_clock_mhz),
        "gpu_pstate": pstate,
        "gpu_temperature_c": int(temperature_c),
        "gpu_power_w": float(power_w),
    }


def pin_entry(entry: CSKCacheEntry) -> float:
    """Replace each layer's key/value tensors in-place with pinned-memory
    copies. This is a one-time setup cost (done before warmup) that is
    timed separately and excluded from per-iteration H2D timing, so the
    "pinned vs pageable" comparison isolates the effect on H2D bandwidth
    itself."""
    started = time.perf_counter()
    for layer_name, (key, value) in tuple(entry.kv_by_layer.items()):
        entry.kv_by_layer[layer_name] = (key.pin_memory(), value.pin_memory())
    if not all(
        key.is_pinned() and value.is_pinned()
        for key, value in entry.kv_by_layer.values()
    ):
        raise RuntimeError("Pinned condition contains a non-pinned source tensor")
    return (time.perf_counter() - started) * 1000.0


def build_block_mapping(
    start: int, end: int, block_size: int
) -> tuple[list[int], int]:
    """Map the target token span [start, end) to a paged-KV-cache
    (logical block id -> physical block id) array, used for scatter/gather
    addressing.

    This script only allocates exactly enough physical blocks to cover the
    span, so the first logical block that's actually used maps to physical
    block 0, incrementing contiguously from there (no fragmentation).
    Logical block ids before `start` are left as placeholder 0 but are
    never accessed. Returns (block_ids, number of physical blocks used)."""
    first_logical = start // block_size
    last_logical = (end - 1) // block_size
    block_ids = [0] * (last_logical + 1)
    for physical, logical in enumerate(range(first_logical, last_logical + 1)):
        block_ids[logical] = physical
    return block_ids, last_logical - first_logical + 1


def allocate_kv_caches(
    entry: CSKCacheEntry,
    *,
    num_blocks: int,
    block_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Allocate one GPU tensor per layer, shaped like a production vLLM
    paged KV cache: (2, num_blocks, block_size, num_kv_heads, head_dim),
    where dim 0's size-2 axis is key/value. This cache is private to this
    benchmark run and not shared with a real vLLM worker — it exists only
    so the to_gpu()/scatter_span() write path matches production code
    exactly."""
    caches: dict[str, torch.Tensor] = {}
    for layer_name, (key, _) in entry.kv_by_layer.items():
        caches[layer_name] = torch.empty(
            (2, num_blocks, block_size, *key.shape[1:]),
            dtype=key.dtype,
            device=device,
        )
    return caches


def build_connector(
    entry: CSKCacheEntry,
    *,
    kv_caches: dict[str, torch.Tensor],
    block_size: int,
    head_size: int,
    max_position: int,
    rope_theta: float,
    device: torch.device,
) -> VLLMPagedGPUConnector:
    """Build a vLLM RoPE module using the same geometry parameters
    (head_size/rope_theta/max_position) as the real model, wrap it in a
    _RopeHolder and hand it to connector.set_model(), so that when
    position_shift != 0, the reuse_slice() call inside connector.to_gpu()
    picks up the exact same cos/sin cache production uses to re-rotate
    keys."""
    dtype = next(iter(entry.kv_by_layer.values()))[0].dtype
    # RotaryEmbedding is a vLLM CustomOp; its __init__ immediately
    # dispatches a forward implementation, which requires a global
    # VllmConfig to exist. This script has no real engine/model, so we
    # push a default config just long enough for that dispatch to read it.
    with set_current_vllm_config(VllmConfig()):
        rope = get_rope(
            head_size,
            max_position=max_position,
            rope_parameters={"rope_theta": rope_theta},
            dtype=dtype,
        )
    holder = _RopeHolder(rope).to(device)
    connector = VLLMPagedGPUConnector(block_size)
    connector.bind_kv_caches(kv_caches)
    connector.set_model(holder)
    return connector


def validate_scatter(
    connector: VLLMPagedGPUConnector,
    entry: CSKCacheEntry,
    plan: CSKLoadPlan,
    block_ids: list[int],
) -> None:
    """After warmup, before real timing starts, run one spot-check for
    correctness: for the first and last layer, take one token each from
    the start and end of the span, independently recompute the "expected"
    value via reuse_slice() (including RoPE if applicable), then read back
    what to_gpu() actually wrote into the GPU paged cache during warmup via
    gather(), and compare byte-for-byte (atol=rtol=0). Only two layers and
    two endpoints are sampled — not a full scan — to keep this check's own
    cost bounded."""
    layer_names = list(entry.kv_by_layer)
    selected_layers = [layer_names[0], layer_names[-1]]
    for layer_name in selected_layers:
        target_cache = connector.kv_caches[layer_name]
        for source_offset in (0, plan.length - 1):
            target_start = plan.start + source_offset
            expected_key, expected_value = connector.reuse_slice(
                entry,
                layer_name=layer_name,
                source_offset=source_offset,
                length=1,
                target_start=target_start,
                device=target_cache.device,
            )
            actual_key, actual_value = connector.gather(
                target_cache,
                block_ids,
                target_start,
                target_start + 1,
            )
            torch.testing.assert_close(actual_key, expected_key, rtol=0, atol=0)
            torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)


def run_iteration(
    *,
    iteration: int,
    measured: bool,
    profiling: bool,
    connector: VLLMPagedGPUConnector,
    entry: CSKCacheEntry,
    plan: CSKLoadPlan,
    block_ids: list[int],
    byte_count: int,
    device: torch.device,
) -> dict[str, Any] | None:
    """Run one connector.to_gpu() call and time it. When measured=False
    (warmup iterations), the exact same path still runs in full to warm up
    the CUDA allocator/driver, but the result is discarded and no record is
    returned, so it doesn't pollute the statistics.

    The three timing figures mean different things:
      - outer_cuda_ms: a single pair of CUDA events wrapped only around the
        to_gpu() call itself — actual GPU execution time (H2D copies,
        optional RoPE, scatter kernels). Used to compute path_gbps
        bandwidth. Both profiling modes use only this one outer event
        pair, otherwise profiling=off would have no way to know when the
        GPU actually finished.
      - operation_wall_ms: host wall-clock time around the to_gpu() call,
        including Python call overhead and the wait on
        end_event.synchronize().
      - end_to_end_wall_ms: operation_wall_ms plus the overhead of
        trace.finish() aggregating per-stage timings; comparing
        profiling on vs. off must use end_to_end_wall_ms (not
        outer_cuda_ms) to actually capture the cost of profiling itself.
    """
    trace: LoadTrace | NullLoadTrace
    if profiling:
        trace = LoadTrace(
            kind="h2d_microbenchmark",
            req_id=f"iteration-{iteration}",
            cache_id=entry.cache_id,
            reuse_index=iteration,
            metadata={
                "bytes": byte_count,
                "tokens": plan.length,
                "target_start": plan.start,
                "target_end": plan.end,
            },
        )
    else:
        trace = NullLoadTrace()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    # Sync once before the iteration so any GPU work left over from the
    # previous iteration has drained and can't bleed into this timing.
    torch.cuda.synchronize(device)
    wall_started = time.perf_counter()
    start_event.record(torch.cuda.current_stream(device))
    expected, scattered, skipped = connector.to_gpu(
        entry, plan, block_ids, trace=trace
    )
    end_event.record(torch.cuda.current_stream(device))
    end_event.synchronize()
    operation_wall_ms = (time.perf_counter() - wall_started) * 1000.0
    outer_cuda_ms = float(start_event.elapsed_time(end_event))
    # to_gpu() either scatters every KV layer or raises immediately
    # (fail-fast on a missing layer) — skipped is always 0. This check is
    # a second line of defense in case a future connector change silently
    # drops a layer without anyone noticing.
    if scattered != expected or skipped != 0:
        raise RuntimeError(
            "Layer accounting mismatch: "
            f"expected={expected}, scattered={scattered}, skipped={skipped}"
        )

    trace_record = trace.finish(status="ok") if profiling else None
    end_to_end_wall_ms = (time.perf_counter() - wall_started) * 1000.0
    if not measured:
        return None
    return {
        "record_type": "iteration",
        "iteration": iteration,
        "operation_wall_ms": operation_wall_ms,
        "end_to_end_wall_ms": end_to_end_wall_ms,
        "outer_cuda_ms": outer_cuda_ms,
        "path_gbps": byte_count / (outer_cuda_ms * 1_000_000.0),
        "expected_layers": expected,
        "scattered_layers": scattered,
        "skipped_layers": skipped,
        "host_stage_ms": trace_record["host_stage_ms"] if trace_record else {},
        "cuda_stage_ms": trace_record["cuda_stage_ms"] if trace_record else {},
        "host_stage_stats": trace_record["host_stage_stats"] if trace_record else {},
        "cuda_stage_stats": trace_record["cuda_stage_stats"] if trace_record else {},
    }


def main() -> None:
    # This process runs exactly one fixed combination of the three switches
    # (one "case"); run.sh sweeps all 8 combinations by launching one fresh
    # process per combination. Where each switch takes effect below, and
    # what question it answers:
    #   --memory          -> the pin_entry() call further down: whether to
    #                         pre-pin the CPU source tensors, measuring the
    #                         effect of pinning on H2D copy bandwidth.
    #   --position-shift  -> the target_start computation further down:
    #                         reuse the KV entry at its original position
    #                         plus this offset; a nonzero offset makes
    #                         to_gpu() trigger a RoPE key re-rotation,
    #                         measuring that recomputation's extra cost.
    #   --profiling       -> the profiling flag passed to run_iteration():
    #                         whether each iteration records per-stage
    #                         timings via LoadTrace, measuring the cost of
    #                         that recording itself (trace.finish()).
    args = parse_args()
    validate_args(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Idempotent: the sweep script may call this process repeatedly; an
    # existing output is skipped by default.
    if args.output.exists() and not args.overwrite:
        print(f"[skipped_existing] case_id={args.case_id} output={args.output}")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the H2D microbenchmark")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("--device must be a CUDA device")
    torch.cuda.set_device(device)

    load_started = time.perf_counter()
    backend = LocalDiskBackend(args.kv_dir)
    entry = backend.get(args.cache_id)
    if entry is None:
        raise FileNotFoundError(
            f"cache_id={args.cache_id!r} is not present in {args.kv_dir}"
        )
    disk_load_ms = (time.perf_counter() - load_started) * 1000.0
    if len(entry.kv_by_layer) != args.expected_layers:
        raise RuntimeError(
            f"Expected {args.expected_layers} layers, got {len(entry.kv_by_layer)}"
        )
    pin_setup_ms = pin_entry(entry) if args.memory == "pinned" else 0.0
    if args.memory == "pageable" and any(
        key.is_pinned() or value.is_pinned()
        for key, value in entry.kv_by_layer.values()
    ):
        raise RuntimeError("Pageable condition contains a pinned source tensor")

    # Set the reuse target position to "original position + position_shift",
    # simulating the entry being reused in a different request with a
    # different prefix length. When position_shift == 0, target_start ==
    # entry.source_start and reuse_slice() skips the RoPE stage entirely;
    # otherwise it triggers rerotate_k_for_target_positions to
    # relative-rotate the key by the position delta.
    target_start = entry.source_start + args.position_shift
    target_end = target_start + entry.length
    if args.position_shift >= args.max_position:
        raise ValueError(
            f"position_shift={args.position_shift} exceeds RoPE cache "
            f"max_position={args.max_position}"
        )
    block_ids, num_blocks = build_block_mapping(
        target_start, target_end, args.block_size
    )
    kv_caches = allocate_kv_caches(
        entry,
        num_blocks=num_blocks,
        block_size=args.block_size,
        device=device,
    )
    connector = build_connector(
        entry,
        kv_caches=kv_caches,
        block_size=args.block_size,
        head_size=args.head_size,
        max_position=args.max_position,
        rope_theta=args.rope_theta,
        device=device,
    )
    # source_offset=0 means the whole cached entry is reused (not a
    # probe-gated partial tail reuse) — the full entry is moved to
    # [target_start, target_end).
    plan = CSKLoadPlan(
        req_id=args.case_id,
        cache_id=entry.cache_id,
        mode=CSKCacheMode.REUSE,
        start=target_start,
        end=target_end,
        token_ids=tuple(entry.token_ids),
        source_offset=0,
    )
    byte_count = entry_nbytes(entry)
    profiling = args.profiling == "on"

    for warmup_index in range(args.warmup):
        run_iteration(
            iteration=-(args.warmup - warmup_index),
            measured=False,
            profiling=profiling,
            connector=connector,
            entry=entry,
            plan=plan,
            block_ids=block_ids,
            byte_count=byte_count,
            device=device,
        )
    validate_scatter(connector, entry, plan, block_ids)

    case_record = {
        "record_type": "case",
        "case_id": args.case_id,
        "profiling": args.profiling,
        "memory": args.memory,
        "position_shift": args.position_shift,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "cache_id": entry.cache_id,
        "tokens": entry.length,
        "bytes": byte_count,
        "layers": len(entry.kv_by_layer),
        "disk_load_ms": disk_load_ms,
        "pin_setup_ms": pin_setup_ms,
        "block_size": args.block_size,
        "num_physical_blocks": num_blocks,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "pid": os.getpid(),
    }
    # Write to a temp file first, flushing every line, then atomically
    # rename to the final path once done, so the sweep script never reads a
    # half-written JSONL left behind by a mid-run crash.
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        output.write(json.dumps(case_record, sort_keys=True) + "\n")
        measured_wall: list[float] = []
        for iteration in range(args.repetitions):
            record = run_iteration(
                iteration=iteration,
                measured=True,
                profiling=profiling,
                connector=connector,
                entry=entry,
                plan=plan,
                block_ids=block_ids,
                byte_count=byte_count,
                device=device,
            )
            assert record is not None
            record.update(
                {
                    "case_id": args.case_id,
                    "profiling": args.profiling,
                    "memory": args.memory,
                    "position_shift": args.position_shift,
                }
            )
            if args.sample_gpu_clock:
                # Sampled after the timed region above, so nvidia-smi's own
                # latency never leaks into wall/CUDA timing.
                record.update(sample_gpu_clock(device.index or 0))
            measured_wall.append(float(record["end_to_end_wall_ms"]))
            output.write(json.dumps(record, sort_keys=True) + "\n")
            output.flush()
    temporary.replace(args.output)
    print(
        f"[completed] case_id={args.case_id} repetitions={args.repetitions} "
        f"median_ms={statistics.median(measured_wall):.3f} output={args.output}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[failed] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
