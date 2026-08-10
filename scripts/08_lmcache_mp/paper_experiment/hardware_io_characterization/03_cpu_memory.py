#!/usr/bin/env python3
"""Measure NUMA-local, cross-NUMA, and pageable-to-pinned CPU copies."""
from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import torch

from common import gib_per_second, load_config, numa_nodes, percentile, write_test_result


def bind(cpus: list[int]) -> None:
    if not cpus:
        raise ValueError("cannot bind to an empty CPU set")
    # 用于设置当前进程可以在哪些 CPU 核心上运行，也叫 CPU 亲和性绑定
    # 0：表示当前进程。
    # set(cpus)：允许使用的 CPU 逻辑核心编号集合。
    os.sched_setaffinity(0, set(cpus))


def allocate_on(cpus: list[int], size_bytes: int, value: int) -> np.ndarray:
    bind(cpus)
    array = np.empty(size_bytes, dtype=np.uint8)
    array.fill(value)
    return array


def time_numpy_copy(
    source: np.ndarray,
    destination: np.ndarray,
    cpus: list[int],
    warmups: int,
    repetitions: int,
) -> list[float]:
    bind(cpus)
    for _ in range(warmups):
        np.copyto(destination, source)
    durations: list[float] = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        np.copyto(destination, source)
        durations.append((time.perf_counter_ns() - started) / 1_000_000)
    if int(destination[0]) != int(source[0]):
        raise RuntimeError("CPU copy checksum guard failed")
    return durations


def summarize(name: str, size_bytes: int, durations: list[float], **fields: Any) -> dict[str, Any]:
    p50 = percentile(durations, 0.50)
    return {
        "path": name,
        "status": "ok",
        "bytes": size_bytes,
        "p50_ms": p50,
        "p95_ms": percentile(durations, 0.95),
        "bandwidth_gib_s": gib_per_second(size_bytes, p50),
        "samples_ms": durations,
        **fields,
    }

# 测三种 CPU 内存复制路径的延迟和带宽
# 在同一个 NUMA 节点、跨 NUMA 节点，以及复制到 CUDA 锁页内存时，CPU 复制速度分别有多快？
def main() -> None:
    config = load_config()
    settings = config["dram"]
    size_bytes = int(settings["copy_size_mib"]) * 1024**2
    pinned_size = int(settings["pinned_copy_size_mib"]) * 1024**2
    warmups = int(settings["warmups"])
    repetitions = int(settings["repetitions"])
    if min(size_bytes, pinned_size, repetitions) <= 0 or warmups < 0:
        raise ValueError("invalid DRAM benchmark configuration")
    original_affinity = sorted(os.sched_getaffinity(0))
    allowed = set(original_affinity)
    nodes = {
        node: [cpu for cpu in cpus if cpu in allowed]
        for node, cpus in numa_nodes().items()
    }
    nodes = {node: cpus for node, cpus in nodes.items() if cpus}
    if not nodes:
        raise RuntimeError("no NUMA node contains a CPU allowed by this process")
    node_ids = sorted(nodes)
    rows: list[dict[str, Any]] = []
    try:
        # 实验一：NUMA 本地内存复制
        # Linux 通常采用 first-touch 内存分配策略：哪个 NUMA 节点的 CPU 第一次写入内存页，物理内存页通常就分配到哪个节点。
        # 所以这里相当于：
        # NUMA 0 CPU：
        # NUMA 0 内存 source → NUMA 0 内存 destination
        # 本地 DRAM 复制性能
        local_node = node_ids[0]
        # 为什么要先绑定 CPU？
        # 单纯调用：
        # np.empty(...)
        # 只会建立虚拟地址，未必立即分配所有物理内存页。
        # 而：array.fill(value)
        # 会真正访问并写入每个内存页，触发物理内存分配。由于此时进程已经绑定到指定 NUMA 节点的 CPU，Linux 通常就会把内存页分配在该 CPU 所属 NUMA 节点中。
        source = allocate_on(nodes[local_node], size_bytes, 17)
        destination = allocate_on(nodes[local_node], size_bytes, 0)
        durations = time_numpy_copy(
            source, destination, nodes[local_node], warmups, repetitions
        )
        rows.append(
            summarize(
                "numa_local_memcpy",
                size_bytes,
                durations,
                source_node=local_node,
                destination_node=local_node,
                execution_node=local_node,
            )
        )

        if len(node_ids) >= 2:
            # 实验二：跨 NUMA 内存复制
            # 在 NUMA 0 的 CPU 上创建并初始化 remote_source；
            # 在 NUMA 1 的 CPU 上创建并初始化 local_destination；
            # 把执行复制的进程绑定到 NUMA 1；
            # 在 NUMA 1 CPU 上执行复制。
            # NUMA 1 CPU 从 NUMA 0 内存读取 source
            #   ↓ 跨 CPU 互联
            # NUMA 1 CPU 向 NUMA 1 本地内存写入 destination
            source_node, destination_node = node_ids[:2]
            remote_source = allocate_on(nodes[source_node], size_bytes, 29)
            local_destination = allocate_on(nodes[destination_node], size_bytes, 0)
            durations = time_numpy_copy(
                remote_source,
                local_destination,
                nodes[destination_node],
                warmups,
                repetitions,
            )
            rows.append(
                summarize(
                    "cross_numa_memcpy",
                    size_bytes,
                    durations,
                    source_node=source_node,
                    destination_node=destination_node,
                    execution_node=destination_node,
                )
            )
        else:
            rows.append(
                {
                    "path": "cross_numa_memcpy",
                    "status": "unavailable",
                    "reason": "only one NUMA node is visible",
                }
            )

        try:
            # 实验三：普通内存复制到锁页内存
            bind(nodes[local_node])
            # Pageable memory
            # 这是普通的 CPU 内存。操作系统可以将其换出到磁盘，也可以移动其物理页。
            pageable = np.empty(pinned_size, dtype=np.uint8)
            pageable.fill(43)
            # 锁页内存，也叫 page-locked memory：
            # 内存页被锁在物理内存中；
            # 不会被操作系统换出；
            # GPU DMA 可以直接访问；
            # 通常用于加速 CPU 到 GPU 的数据传输。
            pinned = torch.empty(pinned_size, dtype=torch.uint8, pin_memory=True)
            pinned_view = pinned.numpy()
            # 这个实验测的是：
            # 普通 CPU 内存 → CPU 锁页内存
            # 只测试 CPU 内存复制，并没有执行 CPU→GPU 传输。
            durations = time_numpy_copy(
                pageable, pinned_view, nodes[local_node], warmups, repetitions
            )
            rows.append(
                summarize(
                    "pageable_to_pinned_memcpy",
                    pinned_size,
                    durations,
                    source_node=local_node,
                    destination_node=local_node,
                    execution_node=local_node,
                )
            )
        except (RuntimeError, OSError) as error:
            rows.append(
                {
                    "path": "pageable_to_pinned_memcpy",
                    "status": "unavailable",
                    "reason": str(error),
                }
            )
    finally:
        os.sched_setaffinity(0, set(original_affinity))
    path = write_test_result(
        "03_cpu_memory",
        config,
        {
            "numa_cpu_map": {str(node): cpus for node, cpus in nodes.items()},
            "measurements": rows,
        },
    )
    print(f"[completed] {path}")


if __name__ == "__main__":
    main()
