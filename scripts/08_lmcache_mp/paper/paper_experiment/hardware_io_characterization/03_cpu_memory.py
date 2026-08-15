#!/usr/bin/env python3
"""Measure NUMA-local, cross-NUMA, and pageable-to-pinned CPU copies."""

from __future__ import annotations

import ctypes
import ctypes.util
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from common import (
    gib_per_second,
    load_config,
    numa_nodes,
    percentile,
    write_test_result,
)


class NumaPlacementError(RuntimeError):
    pass


def bind(cpus: list[int]) -> None:
    if not cpus:
        raise ValueError("cannot bind to an empty CPU set")
    # 用于设置当前进程可以在哪些 CPU 核心上运行，也叫 CPU 亲和性绑定
    # 0：表示当前进程。
    # set(cpus)：允许使用的 CPU 逻辑核心编号集合。
    os.sched_setaffinity(0, set(cpus))


def load_numa() -> ctypes.CDLL:
    # libnuma 是 Linux 上操作 NUMA memory policy 的系统库。Python 标准库没有
    # numa_alloc_onnode、move_pages 等接口，所以这里用 ctypes 直接调用 C API。
    # find_library 返回当前系统实际可加载的 soname，例如 "libnuma.so.1"，避免
    # 把某一种发行版上的绝对路径写死在实验代码中。
    library_path = ctypes.util.find_library("numa")
    if library_path is None:
        raise RuntimeError("libnuma is required to fix CPU memory placement")

    # use_errno=True 会让 ctypes 保存 C 函数调用后的 errno。这样 move_pages
    # 返回 -1 时，Python 才能通过 ctypes.get_errno() 得到 EPERM、EINVAL 等
    # 真实系统错误，而不是只看到一个没有原因的失败返回值。
    library = ctypes.CDLL(library_path, use_errno=True)

    # ctypes 默认不知道 C 函数参数和返回值的类型。如果不显式声明，64 位
    # 指针可能被当成普通整数错误处理。因此下面逐个声明本实验使用的接口签名。
    library.numa_available.argtypes = []
    library.numa_available.restype = ctypes.c_int

    # void *numa_alloc_onnode(size_t size, int node)
    # 它返回一段带指定 NUMA node policy 的虚拟地址。后续实际分配物理页时，
    # Linux 应按这个 policy 放置，而不只是依赖“哪个 CPU 先访问”的 first-touch。
    library.numa_alloc_onnode.argtypes = [ctypes.c_size_t, ctypes.c_int]
    library.numa_alloc_onnode.restype = ctypes.c_void_p

    # numa_alloc_onnode 分配的地址不受 Python 垃圾回收管理，必须和 numa_free 配对。
    library.numa_free.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    library.numa_free.restype = None

    # long move_pages(pid, count, pages, nodes, status, flags)
    # 当 nodes=NULL 时，它只查询 pages 中每一页当前所在的 NUMA node，
    # 不会执行迁移；查询结果逐页写入 status 数组。
    library.move_pages.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]
    library.move_pages.restype = ctypes.c_int

    # 返回负数表示当前机器或内核没有 NUMA policy 支持。此时如果继续运行，
    # 就只能退回不可靠的 first-touch 测法，所以实验选择直接失败。
    if library.numa_available() < 0:
        raise RuntimeError("NUMA memory policy is unavailable on this system")
    return library


@dataclass
class NumaBuffer:
    """管理一块由 libnuma 分配、由 NumPy 访问的 CPU 内存。

    普通 NumPy 数组拥有自己的内存，生命周期结束时由 Python 自动释放。这里的
    内存来自 C 库，array 只是原始地址上的零拷贝视图，并不拥有底层内存。因此
    还要保存 address 和 size_bytes，以便退出 with 时显式调用 numa_free。
    """

    library: ctypes.CDLL
    address: int
    size_bytes: int
    array: np.ndarray

    @classmethod
    def allocate(
        cls,
        library: ctypes.CDLL,
        cpus: list[int],
        node: int,
        size_bytes: int,
        value: int,
    ) -> "NumaBuffer":
        # CPU affinity 和 memory placement 是两件不同的事：
        # - bind(cpus) 决定下面的初始化代码在哪组 CPU 上执行；
        # - numa_alloc_onnode(..., node) 决定内存采用哪个 NUMA node policy。
        # 两者同时固定，才能明确描述“哪个 CPU 正在访问哪个节点的内存”。
        bind(cpus)
        pointer = library.numa_alloc_onnode(size_bytes, node)
        if not pointer:
            raise MemoryError(
                f"numa_alloc_onnode failed: node={node}, bytes={size_bytes}"
            )

        # C API 返回 void*。先保留其整数地址，再在该地址上构造 ctypes byte
        # array，最后让 NumPy 建立零拷贝视图。这里没有重新分配或复制数据，
        # 后续 np.copyto 访问的就是 libnuma 分配的那块内存。
        address = int(pointer)
        raw_buffer = (ctypes.c_uint8 * size_bytes).from_address(address)
        array = np.ctypeslib.as_array(raw_buffer)

        # numa_alloc_onnode 主要建立虚拟地址和 policy，物理页通常仍是按需分配。
        # fill 会触碰整个数组，让每个页真正落到物理内存中，同时写入 checksum
        # 使用的已知值。后面的 move_pages 会进一步确认内核是否遵循了 policy。
        array.fill(value)
        return cls(library, address, size_bytes, array)

    def close(self) -> None:
        # address=0 同时作为“已经释放”的标记，避免意外重复调用造成 double-free。
        # 一旦释放，调用方就不能再访问之前的 NumPy array 视图。
        if self.address:
            self.library.numa_free(
                ctypes.c_void_p(self.address), ctypes.c_size_t(self.size_bytes)
            )
            self.address = 0

    def __enter__(self) -> "NumaBuffer":
        return self

    def __exit__(self, *_args: object) -> None:
        # 即使计时、checksum 或 placement 检查抛出异常，with 也会调用这里，
        # 因此大块 NUMA 内存不会因为实验中途失败而泄漏。
        self.close()


def verify_placement(
    library: ctypes.CDLL,
    array: np.ndarray,
    expected_node: int,
    label: str,
) -> dict[str, Any]:
    # move_pages 的查询单位是虚拟内存页，不是 NumPy element。常见 x86 Linux
    # 页面为 4 KiB，但实验不应硬编码这个值，因此从当前系统动态读取。
    page_size = os.sysconf("SC_PAGE_SIZE")
    start = int(array.ctypes.data)

    # 数组首地址通常按页对齐，但 pinned allocator 等实现不保证一定如此。
    # first_page 向下对齐到覆盖数组起点的页面，page_count 再向上覆盖数组末尾，
    # 从而检查数组跨越的每一个页面，而不是只抽查首尾两个页。
    first_page = start - start % page_size
    end = start + int(array.nbytes)
    page_count = (end - first_page + page_size - 1) // page_size

    # pages 是传给 C API 的 void* 地址数组。256 MiB / 4 KiB 等于 65,536 页。
    # 这一步发生在正式计时区间之外，因此不会计入 np.copyto 的复制延迟。
    pages = (ctypes.c_void_p * page_count)(
        *(first_page + index * page_size for index in range(page_count))
    )
    statuses = (ctypes.c_int * page_count)()
    ctypes.set_errno(0)

    # pid=0 表示当前进程；nodes=None 对应 C 语言的 NULL，语义是“只查询”；
    # flags=0 也不请求迁移。这里不会主动修正错误 placement，因为偷偷修正
    # 会掩盖 benchmark 自己的内存放置问题。
    result = library.move_pages(0, page_count, pages, None, statuses, 0)
    if result < 0:
        error_number = ctypes.get_errno()
        raise NumaPlacementError(
            f"move_pages query failed for {label}: "
            f"[{error_number}] {os.strerror(error_number)}"
        )
    counts: dict[int, int] = {}
    for status in statuses:
        # 查询成功时，status[i] 是第 i 页当前所在的 node 编号；负数则是该页的
        # -errno，例如页面不可访问。这里将逐页结果聚合，方便写入 JSON 复查。
        if status < 0:
            raise NumaPlacementError(
                f"move_pages page query failed for {label}: status={status}"
            )
        counts[status] = counts.get(status, 0) + 1

    # 这里要求严格相等，而不是“多数页正确”。只要一个页出现在其他节点，
    # 本次复制就不再是定义清楚的纯本地/纯跨 NUMA 路径。fail closed 可以避免
    # 自动 NUMA balancing 产生的混合结果被误当成有效论文数字。
    if counts != {expected_node: page_count}:
        raise NumaPlacementError(
            f"NUMA placement changed for {label}: expected node "
            f"{expected_node}, observed {counts}"
        )
    return {
        "expected_node": expected_node,
        "page_size_bytes": page_size,
        "page_count": page_count,
        "node_page_counts": {
            str(node): count for node, count in sorted(counts.items())
        },
    }


def time_numpy_copy(
    source: np.ndarray,
    destination: np.ndarray,
    cpus: list[int],
    warmups: int,
    repetitions: int,
) -> list[float]:
    # 先把当前进程限制在指定 NUMA node 的 CPU 集合上。np.copyto 内部可能使用
    # libc/SIMD 实现，但执行线程仍只能在该 affinity 集合内调度。
    bind(cpus)

    # warmup 不进入统计。它用于消除首次调用、CPU 频率爬升和冷指令路径造成的
    # 启动偏差；内存页归属则由调用方在 warmup 前后独立验证。
    for _ in range(warmups):
        np.copyto(destination, source)
    durations: list[float] = []
    for _ in range(repetitions):
        # perf_counter_ns 是单调高精度时钟，不受系统时间校准影响。计时区间只
        # 包住一次 memcpy，不包含 NUMA 页查询、分配、释放或 JSON 写入。
        started = time.perf_counter_ns()
        np.copyto(destination, source)
        durations.append((time.perf_counter_ns() - started) / 1_000_000)

    # 这是轻量正确性保护，不是完整数据校验。source 在分配时被填成固定值，
    # 如果 destination[0] 不相等，说明复制路径至少没有正确写入目标缓冲区。
    if int(destination[0]) != int(source[0]):
        raise RuntimeError("CPU copy checksum guard failed")
    return durations


def summarize(
    name: str, size_bytes: int, durations: list[float], **fields: Any
) -> dict[str, Any]:
    p50 = percentile(durations, 0.50)
    return {
        "path": name,
        "status": "ok",
        "bytes": size_bytes,
        "p50_ms": p50,
        "p95_ms": percentile(durations, 0.95),
        # 这里报告 payload bytes / p50 time，即“有效复制带宽”。一次 memcpy
        # 同时产生 DRAM 读取和写入流量，但本指标不把二者相加，因此不能直接
        # 当作内存控制器的总线带宽。
        "bandwidth_gib_s": gib_per_second(size_bytes, p50),
        "samples_ms": durations,
        **fields,
    }


def measure_pageable_copy(
    library: ctypes.CDLL,
    name: str,
    size_bytes: int,
    nodes: dict[int, list[int]],
    source_node: int,
    destination_node: int,
    execution_node: int,
    warmups: int,
    repetitions: int,
    source_value: int,
) -> dict[str, Any]:
    # source 和 destination 各自带明确的 NUMA policy：
    # - 本地实验中，两者以及执行 CPU 都位于同一个 node；
    # - 跨 NUMA 实验中，source 固定在 node 0，destination 与执行 CPU 固定
    #   在 node 1。此时数据路径就是 node 1 CPU 远程读取 node 0 source，
    #   再将结果写入 node 1 的本地 destination。
    with (
        NumaBuffer.allocate(
            library, nodes[source_node], source_node, size_bytes, source_value
        ) as source,
        NumaBuffer.allocate(
            library, nodes[destination_node], destination_node, size_bytes, 0
        ) as destination,
    ):
        # before/after 使用完全相同的全页检查。before 证明实验起点正确；
        # after 证明 warmup 和 repetitions 期间没有被 automatic NUMA balancing
        # 改写成另一条路径。两份页归属快照都会随测量结果写入 JSON。
        def placement(phase: str) -> dict[str, Any]:
            return {
                "source": verify_placement(
                    library, source.array, source_node, f"{name} source {phase}"
                ),
                "destination": verify_placement(
                    library,
                    destination.array,
                    destination_node,
                    f"{name} destination {phase}",
                ),
            }

        placement_before = placement("before")
        durations = time_numpy_copy(
            source.array,
            destination.array,
            nodes[execution_node],
            warmups,
            repetitions,
        )
        placement_after = placement("after")
    return summarize(
        name,
        size_bytes,
        durations,
        source_node=source_node,
        destination_node=destination_node,
        execution_node=execution_node,
        placement_before=placement_before,
        placement_after=placement_after,
    )


# 测三种 CPU 内存复制路径的延迟和带宽
# 在同一个 NUMA 节点、跨 NUMA 节点，以及复制到 CUDA 锁页内存时，CPU 复制速度分别有多快？
def main() -> None:
    # 参数仍统一来自 config.yaml，脚本本身不要求记忆命令行参数。两个 size
    # 分开配置，是因为 pageable→pinned 需要 CUDA pinned allocator，通常没有
    # 必要为了测 CPU staging 一次申请和 NUMA memcpy 相同的 256 MiB 锁页内存。
    config = load_config()
    numa = load_numa()
    settings = config["dram"]
    size_bytes = int(settings["copy_size_mib"]) * 1024**2
    pinned_size = int(settings["pinned_copy_size_mib"]) * 1024**2
    warmups = int(settings["warmups"])
    repetitions = int(settings["repetitions"])
    if min(size_bytes, pinned_size, repetitions) <= 0 or warmups < 0:
        raise ValueError("invalid DRAM benchmark configuration")

    # 保存进入脚本时的 affinity。测试过程中会依次绑定 NUMA 0、NUMA 1，
    # finally 必须恢复原值，避免脚本结束后影响同一进程中的清理代码或调用方。
    original_affinity = sorted(os.sched_getaffinity(0))
    allowed = set(original_affinity)

    # sysfs 给出机器上每个 NUMA node 的全部 CPU，但容器、作业调度器或 taskset
    # 可能只允许当前进程使用其中一部分。这里取交集，避免绑定到无权限的 CPU。
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
        # libnuma 为 source 和 destination 设置显式内存策略，并在计时前后
        # 查询全部物理页归属。因此这里相当于：
        # NUMA 0 CPU：
        # NUMA 0 内存 source → NUMA 0 内存 destination
        # 本地 DRAM 复制性能
        local_node = node_ids[0]
        rows.append(
            measure_pageable_copy(
                numa,
                "numa_local_memcpy",
                size_bytes,
                nodes,
                local_node,
                local_node,
                local_node,
                warmups,
                repetitions,
                17,
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
            rows.append(
                measure_pageable_copy(
                    numa,
                    "cross_numa_memcpy",
                    size_bytes,
                    nodes,
                    source_node,
                    destination_node,
                    destination_node,
                    warmups,
                    repetitions,
                    29,
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
            # Pageable memory 是普通 CPU 内存：操作系统可以换出或迁移其物理页。
            # 本实验没有用 np.empty，而是让 NumaBuffer 把它固定到 local_node，
            # 这样 pageable source 的初始位置和计时后的实际位置都可以验证。
            #
            # 锁页内存，也叫 page-locked memory：
            # - 内存页被锁在物理内存中，不会在 DMA 期间被换出或更换物理地址；
            # - GPU 的 DMA engine 因而能直接把它作为稳定的 H2D 数据源；
            # - pageable 数据发往 GPU 时，CUDA 往往需要先复制到内部 pinned
            #   staging buffer，所以真实系统里 pageable→pinned 本身就是一段开销。
            with NumaBuffer.allocate(
                numa, nodes[local_node], local_node, pinned_size, 43
            ) as pageable_buffer:
                # torch.empty(..., pin_memory=True) 通过 PyTorch/CUDA pinned allocator
                # 申请锁页内存。zero_ 不属于计时，它既初始化 checksum 目标，也会
                # 触碰每一个页，使 move_pages 能查询到完整的物理页归属。
                pinned = torch.empty(pinned_size, dtype=torch.uint8, pin_memory=True)
                pinned.zero_()
                pinned_view = pinned.numpy()

                # pinned tensor 仍然位于 CPU 地址空间，所以可以建立 NumPy 视图并
                # 用 move_pages 查询 node。这里要求 pageable 与 pinned 都在 GPU
                # 所在的 local_node，避免把 NUMA 惩罚混入 staging 基线。
                placement_before = {
                    "source": verify_placement(
                        numa,
                        pageable_buffer.array,
                        local_node,
                        "pageable source before",
                    ),
                    "destination": verify_placement(
                        numa, pinned_view, local_node, "pinned destination before"
                    ),
                }
                # 这个实验测的是：
                # 普通 CPU 内存 → CPU 锁页内存
                # 只测试 GPU DMA 之前的 CPU staging，并没有执行 CPU→GPU 传输；
                # 真正的 PCIe H2D 由 04_pcie_h2d.py 单独测量。
                durations = time_numpy_copy(
                    pageable_buffer.array,
                    pinned_view,
                    nodes[local_node],
                    warmups,
                    repetitions,
                )
                placement_after = {
                    "source": verify_placement(
                        numa,
                        pageable_buffer.array,
                        local_node,
                        "pageable source after",
                    ),
                    "destination": verify_placement(
                        numa, pinned_view, local_node, "pinned destination after"
                    ),
                }
            rows.append(
                summarize(
                    "pageable_to_pinned_memcpy",
                    pinned_size,
                    durations,
                    source_node=local_node,
                    destination_node=local_node,
                    execution_node=local_node,
                    placement_before=placement_before,
                    placement_after=placement_after,
                )
            )
        except NumaPlacementError:
            # 页归属错误会破坏实验定义，不能降级成 unavailable 后继续写 JSON，
            # 因此原样抛出，让本次运行失败并保留上一次结果文件。
            raise
        except (RuntimeError, OSError) as error:
            # 某些机器没有可用 CUDA/pinned allocator。它不影响前两项 NUMA
            # memcpy，因此只把该路径记录为 unavailable，而不是丢弃全部结果。
            rows.append(
                {
                    "path": "pageable_to_pinned_memcpy",
                    "status": "unavailable",
                    "reason": str(error),
                }
            )
    finally:
        # restore 发生在 JSON 写入前，并且异常路径同样执行。
        os.sched_setaffinity(0, set(original_affinity))

    # write_test_result 使用临时文件 + os.replace 原子替换。只有三项测量完成且
    # placement 检查通过后才会走到这里；失败不会留下半截 JSON。
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
