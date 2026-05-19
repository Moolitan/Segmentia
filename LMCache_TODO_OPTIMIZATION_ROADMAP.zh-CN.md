# LMCache 内存、显存与数据调度 TODO 优化路线图

这份文档整理了 LMCache 代码中作者留下的、和 **内存/显存优化**、**KV cache 数据搬运**、**多级存储调度**、**prefetch/lookup** 相关的 TODO/FIXME。它不是简单 TODO 清单，而是面向后续研发的技术债分析和优化路线图。

## 1. 范围和筛选原则

本次扫描重点看：

- `lmcache/`
- `csrc/`

刻意排除：

- vendored `vllm/` 目录下的 TODO
- operator/Kubernetes 模板中的 TODO
- 测试文件里仅用于 skip 的 TODO
- 和格式、兼容性、纯类型标注无关的低相关 TODO

筛选关键词包括：

- memory, mem, allocator, allocation, buffer
- GPU, CUDA, XPU, H2D, D2H
- KV cache, cache, chunk, layerwise
- batch, batching, batched
- prefetch, lookup, retrieve, store, load
- eviction, fragmentation, ref count
- P2P, PD, NIXL, GDS, Redis, S3
- copy, permute, cat, zero-copy

这份文档里的行号是基于当前工作区扫描结果，后续代码变动后可能会漂移，建议用文件路径和 TODO 文本定位。

## 2. 总体判断

LMCache 当前已经有比较完整的多级 KV cache 架构：GPU connector、CacheEngine、StorageManager、本地 CPU/disk、远端 connector、P2P/PD/NIXL、layerwise 和 CacheGen 都已经具备骨架。

但是 TODO 透露出的主要问题是：

1. 很多“batched”接口还是 Python for-loop 版批处理。
2. GPU KV cache 搬运路径里仍有额外临时 tensor、`torch.cat`、slot mapping 复制等开销。
3. CacheEngine 仍有不少逐 chunk allocate、remove、contains、ref-count 处理。
4. layerwise、async loading、prefetch、多 location retrieve 之间还没有完全打通。
5. CPU/disk allocator 和 eviction 还不能很好处理碎片、批量淘汰、引用计数和并发写入。
6. P2P/PD/NIXL/S3/Redis/GDS 的异步完成、polling、zero-copy 和 location-aware 调度还比较初级。
7. CacheGen 压缩路径仍存在额外拷贝、layout 变换和 kernel 层可优化点。

按收益和落地难度看，优先级最高的是：

- GPU connector 真批量搬运
- CacheEngine 批量 allocate / batched cleanup / batched lookup
- async lookup + prefetch + layerwise 的统一调度
- CPU/disk allocator 碎片和 eviction 策略
- P2P/PD/NIXL 的异步完成与 location-aware 调度

## 3. 优先级概览

| 优先级 | 方向 | 为什么重要 | 代表 TODO |
| --- | --- | --- | --- |
| P0 | GPU connector 真批量 H2D/D2H | KV load/store 位于 TTFT 关键路径。现在多个 batched API 只是循环调用单 chunk 路径，导致 Python overhead、kernel launch、同步和临时分配都随 chunk 数增长。 | `lmcache/v1/gpu_connector/gpu_connectors.py:478`, `:488`, `:1083`, `:1443`, `:1954`, `:1962` |
| P0 | CacheEngine 批量 allocate/cleanup/contains | 即使 GPU kernel 真批量了，如果上层仍逐 chunk allocate、remove、ref-count，整体延迟仍会被 Python 和锁开销限制。 | `lmcache/v1/cache_engine.py:550`, `:619`, `:993`, `:1360`, `:1507` |
| P1 | async lookup、prefetch、layerwise、多 location retrieve | 这些能力决定 LMCache 是否能把远端/磁盘 I/O 和 GPU transfer 隐藏在计算后面。 | `lmcache/v1/storage_backend/storage_manager.py:510`, `:578`, `:711`; `lmcache/v1/cache_engine.py:1155` |
| P1 | CPU/disk allocator、碎片和 eviction | under memory pressure 时，allocation stall 和错误淘汰会直接影响命中率和延迟稳定性。 | `local_cpu_backend.py:617`, `:726`; `local_disk_backend.py:154`, `:327` |
| P1 | P2P/PD/NIXL 调度 | 多实例/分离式 prefill 场景下，remote hit 是否有价值取决于 transfer 是否能异步、批量、可预测地完成。 | `p2p_backend.py:294`, `:337`; `pd_backend.py:289`, `:455`; `nixl_channel.py:444` |
| P2 | CacheGen/serde copy reduction | 压缩能减少存储和传输压力，但 encode/decode 和 layout 转换如果太贵，会抵消收益。 | `cachegen_encoder.py:380`; `naive_serde/cachegen_encoder.py:39`; `naive_serde/cachegen_decoder.py:64` |

## 4. GPU connector 批量搬运与显存拷贝优化

### 4.1 相关 TODO

主要位置：

- `lmcache/v1/gpu_connector/gpu_connectors.py:478`
  - TODO: `batched_to_gpu` 需要优化成真正 batching。
- `lmcache/v1/gpu_connector/gpu_connectors.py:488`
  - TODO: `batched_from_gpu` 需要优化成真正 batching。
- `lmcache/v1/gpu_connector/gpu_connectors.py:1083`
  - TODO: 减少 `batched_to_gpu` 和 `batched_from_gpu` 中重复操作。
- `lmcache/v1/gpu_connector/gpu_connectors.py:1443`
  - TODO: 优化掉 `torch.cat`。
- `lmcache/v1/gpu_connector/gpu_connectors.py:1954`
  - TODO: SGLang connector 的 `batched_to_gpu` 需要 real batching。
- `lmcache/v1/gpu_connector/gpu_connectors.py:1962`
  - TODO: SGLang connector 的 `batched_from_gpu` 需要 real batching。
- `lmcache/v1/gpu_connector/xpu_connectors.py:222`
  - TODO: XPU connector 同样需要 real batching。

### 4.2 当前问题

当前多个 connector 的 batched API 长得像批处理，但本质还是：

```python
for memory_obj, start, end in zip(memory_objs, starts, ends):
    self.to_gpu(memory_obj, start, end, **kwargs)
```

或：

```python
for memory_obj, start, end in zip(memory_objs, starts, ends):
    self.from_gpu(memory_obj, start, end, **kwargs)
```

这会导致：

- 每个 chunk 单独进入 Python 调用栈。
- 每个 chunk 可能触发一次或多次 CUDA kernel launch。
- slot mapping 每个 chunk 单独 slice、复制或拼接。
- 临时 GPU buffer 和 tensor 视图管理重复执行。
- stream synchronization 粒度过细。
- layerwise 模式下 chunk 数 × layer 数会放大这些开销。

其中 `gpu_connectors.py:1443` 的 `torch.cat` 是一个很具体的热点。注释里已经说明：`slot_mapping[start:end]` 通常只是 view，但 `torch.cat(...)` 会创建一个新的连续 tensor，并把多个 slice 拷进去。这会产生额外显存分配和额外拷贝。

### 4.3 为什么这影响 TTFT

LMCache 的收益来自避免 prefill 计算，但命中后仍然要把 KV cache 搬回 serving engine 的 GPU KV cache。对于短 prompt、多 chunk、partial hit、layerwise retrieval 这类场景，真实 KV 数据搬运不一定是唯一瓶颈，调度和元数据处理也可能变成瓶颈。

假设一次请求命中 16 个 chunk，每个 chunk 32 tokens。如果 batched API 是 loop：

- 16 次 Python 调用
- 16 次 slot mapping slice 处理
- 16 次 kernel launch 或 transfer 调用
- 16 次可能的 MemoryObj/tensor 检查

如果再乘以 layerwise 的 layer 数，调度开销会非常明显。

### 4.4 建议实现方向

#### 方案 A: descriptor-based batched transfer

引入 transfer descriptor 数组，每个 descriptor 描述一个 chunk 的搬运：

```text
TransferDescriptor {
  src_ptr,
  dst_ptr,
  slot_mapping_ptr,
  token_count,
  layer_start,
  layer_count,
  memory_format,
  block_size,
  head_size
}
```

然后让 CUDA op 一次消费 descriptor 数组，而不是 Python 循环多次调用单 chunk op。

优点：

- Python 调用次数降到 1 次。
- kernel launch 数可以降低。
- 多 chunk metadata 可一次性传给 kernel。
- 更容易在 kernel 内做 coalescing 和按 layout 优化。

风险：

- 需要改 `lmc_ops.multi_layer_kv_transfer` 或新增 sibling op。
- descriptor 的生命周期和 device/host 放置需要设计。
- 不同 connector 的 KV format 差异需要抽象。

#### 方案 B: 先消除 `torch.cat` 和重复 slot mapping 拷贝

如果短期不改 kernel，可以先做低风险优化：

- 在 connector 中维护可复用 slot mapping workspace。
- 对相同 batch size/token count 复用 buffer。
- 避免每次 `torch.cat` 触发新分配。
- 对 `slot_mapping.to(device)` 做预分配拷贝，而不是每次生成新 tensor。

这和 vLLM adapter 中的 TODO 可以联动：

- `lmcache/integration/vllm/vllm_v1_adapter.py:846`
- `lmcache/integration/vllm/vllm_v1_adapter.py:1099`
- `lmcache/integration/vllm/vllm_v1_adapter.py:1202`

这些地方都写了需要预分配 slot mappings buffer。

#### 方案 C: load/store 共用 batch preparation

`gpu_connectors.py:1083` 提到 `batched_to_gpu` 和 `batched_from_gpu` 有重复操作。可以抽一个内部 helper，例如：

```python
BatchTransferPlan = self._prepare_batch_transfer(memory_objs, starts, ends, slot_mapping)
```

这个 plan 负责：

- 检查 starts/ends 长度。
- 计算每个 chunk 的 token_count。
- 准备 slot mapping 描述。
- 准备 temporary GPU buffer 视图。
- 计算 offsets。

然后 load/store 只负责方向不同。

### 4.5 验证建议

需要 microbenchmark，不然优化容易只改了代码形状：

- 固定总 token 数，改变 chunk 数。
- 固定 chunk 数，改变每 chunk token 数。
- 比较 non-layerwise 和 layerwise。
- 比较 CPU pinned memory、GPU temporary buffer、direct device memory。
- 统计 kernel launch 数、CUDA malloc 数、H2D/D2H 带宽、TTFT。

## 5. CacheEngine 和 StorageManager 的批量化

### 5.1 相关 TODO

主要位置：

- `lmcache/v1/cache_engine.py:550`
  - store 时逐 chunk allocate，TODO 是未来 batched allocate。
- `lmcache/v1/cache_engine.py:619`
  - 隐式依赖 `batched_put` 做 `ref_count_down`，引用计数管理需要更清晰。
- `lmcache/v1/cache_engine.py:993`
  - retrieve 后逐 chunk 循环 cleanup，TODO 是用 batched operation 替换。
- `lmcache/v1/cache_engine.py:995`
  - `remove_after_retrieve` 逻辑需要重构。
- `lmcache/v1/cache_engine.py:1155`
  - layerwise retrieve 未来支持从多个 location 取。
- `lmcache/v1/cache_engine.py:1360`
  - layerwise 模式下支持 `batched_contains`。
- `lmcache/v1/cache_engine.py:1369`
  - layerwise lookup 可以优化为只检查某一层 key。
- `lmcache/v1/cache_engine.py:1477`
  - move 逻辑里需要减少 loop。
- `lmcache/v1/cache_engine.py:1507`
  - async lookup + prefetch 需要支持 layerwise。

StorageManager 对应位置：

- `lmcache/v1/storage_backend/storage_manager.py:215`
  - StorageManager 需要扩展 caching policy 和 eviction policy。
- `lmcache/v1/storage_backend/storage_manager.py:487`
  - 需要确保返回的 MemoryObj 都来自 allocator backend。
- `lmcache/v1/storage_backend/storage_manager.py:510`
  - `get_non_blocking` 需要纳入 prefetch。
- `lmcache/v1/storage_backend/storage_manager.py:549`
  - write-back 逻辑应该重构进 caching policy module。
- `lmcache/v1/storage_backend/storage_manager.py:578`
  - async loading 和 layerwise 需要兼容。
- `lmcache/v1/storage_backend/storage_manager.py:593`
  - 支持 write-back policy。
- `lmcache/v1/storage_backend/storage_manager.py:711`
  - 当前 retrieval 假设 prefix-based，需要支持非前缀或中间 chunk 缺失。
- `lmcache/v1/storage_backend/storage_manager.py:1053`
  - remove 需要处理非 CPU backend。
- `lmcache/v1/storage_backend/storage_manager.py:1123`
  - clear 需要处理非 CPU backend。

### 5.2 当前问题

CacheEngine 表面上已经传递 list：

- `memory_objs`
- `starts`
- `ends`
- `keys`

但是内部许多步骤仍然逐 chunk：

1. 每个 chunk 调一次 `storage_manager.allocate(...)`。
2. 每个 chunk 单独追加 event。
3. retrieve 后每个 chunk 单独 remove。
4. 每个 MemoryObj 单独 ref-count down。
5. layerwise contains 需要拆成所有 layer key，再查所有 key。

这意味着即便底层存储 backend 支持 batch，上层也没有完全利用。

### 5.3 影响

逐 chunk 逻辑的影响不是单点，而是贯穿整个请求生命周期：

- chunk 越小，overhead 越明显。
- layerwise 模式下，chunk 数和 layer 数叠加。
- under memory pressure 时，逐个 allocate 更容易反复触发 eviction。
- remove-after-retrieve 时，cleanup 成为额外同步点。
- ref-count ownership 不清晰时，异步 backend 容易引入泄漏或提前释放。

### 5.4 建议实现方向

#### 方向 A: 请求级 CacheTransferPlan

在真正 allocate/transfer 之前，先构造请求级 plan：

```text
CacheTransferPlan {
  chunks: [
    {
      key,
      start,
      end,
      num_tokens,
      hit_location,
      target_location,
      shape,
      dtype,
      layer_ids,
      transfer_direction
    }
  ]
}
```

这个 plan 可以统一服务：

- allocate
- contains
- get
- from_gpu/to_gpu
- put
- remove-after-retrieve
- prefetch
- write-back

#### 方向 B: 使用 batched_allocate

当前 `StorageManager` 已有 `batched_allocate` 接口，但 `CacheEngine.store` TODO 说明还没真正用上。可以先把相同 shape/dtype/fmt 的 chunk 合并申请：

```python
memory_objs = self.storage_manager.batched_allocate(
    kv_shapes,
    kv_dtypes,
    batch_size=len(chunks),
    fmt=self.fmt,
)
```

需要处理的问题：

- 最后一个 chunk token 数可能小于 full chunk。
- layerwise 和 non-layerwise shape 不同。
- 分配失败时要返回部分成功还是整体失败。
- eviction 是否以 batch 目标大小为单位。

#### 方向 C: 明确 MemoryObj ownership

`cache_engine.py:619` 的 TODO 很关键。现在调用方“隐式依赖” `batched_put` 做 `ref_count_down`。这类隐式 ownership 在异步 backend 里很危险。

建议定义明确协议：

```text
batched_put(keys, memory_objs) -> BatchedPutResult

BatchedPutResult {
  consumed_refs: list[bool],
  pending_refs: list[bool],
  failed_keys: list[key]
}
```

或者更简单：

- 调用方永远负责释放自己的 reference。
- backend 如果异步使用 MemoryObj，必须自己 `ref_count_up`，完成后自己 `ref_count_down`。

第二种规则更容易理解。

#### 方向 D: layerwise contains 优化

`cache_engine.py:1369` 提到只检查某一层 key。这取决于一个重要 invariant：

- 如果所有 layer 的 key 总是一起创建、一起 evict、一起 move，那么查一层即可。
- 如果不同 layer 可能独立存在，那么必须返回 per-layer availability。

需要先明确 layerwise key 的一致性语义。否则简单优化可能引入错误命中。

### 5.5 验证建议

重点测试：

- batch allocate 部分失败。
- 最后一个 chunk size 不同。
- layerwise 某一层缺失。
- `remove_after_retrieve=True`。
- async loading 下 MemoryObj ref-count 不泄漏。
- retrieve location 分散在 CPU/disk/P2P。

## 6. CPU 内存、磁盘空间、碎片和淘汰

### 6.1 相关 TODO

内存管理：

- `lmcache/v1/memory_management.py:631`
  - TODO: 考虑缓存 `get_size()`。
- `lmcache/v1/memory_management.py:638`
  - TODO: `byte_array` 可以考虑其他方式，可能减少 copy。
- `lmcache/v1/memory_management.py:772`
  - TODO: 需要记录 token 数。
- `lmcache/v1/memory_management.py:1645`
  - TODO: allocate 时 metadata 设置有点冗余。
- `lmcache/v1/memory_management.py:1657`
  - TODO: debug ops 需要 flag 控制。
- `lmcache/v1/memory_management.py:1810`
  - FIXME: NIXL 相关 memory leak 应该在别处处理。
- `lmcache/v1/memory_management.py:1851`
  - TODO: byte buffer batched allocation loop 可优化。

LocalCPUBackend：

- `lmcache/v1/storage_backend/local_cpu_backend.py:233`
  - TODO: `batched_submit_put_task` 需要 batching。
- `lmcache/v1/storage_backend/local_cpu_backend.py:617`
  - TODO: `num_candidates` 需要根据估计优化，碎片使精确估计困难。
- `lmcache/v1/storage_backend/local_cpu_backend.py:726`
  - TODO: batched allocation 时同样需要优化 eviction candidate 数。
- `lmcache/v1/storage_backend/local_cpu_backend.py:745`
  - TODO: 通过 `batched_remove` 支持 batched allocate 时，usage tracking 等能力还不支持。
- `lmcache/v1/storage_backend/local_cpu_backend.py:874`
  - TODO: clear 时 token 统计和 remove 不是原子操作，可能不准确。

LocalDiskBackend：

- `lmcache/v1/storage_backend/local_disk_backend.py:37`
  - TODO: 处理重复 prefetch 同一个 cache。
- `lmcache/v1/storage_backend/local_disk_backend.py:154`
  - TODO: 需要 disk space allocator 来避免 fragmentation。
- `lmcache/v1/storage_backend/local_disk_backend.py:327`
  - TODO: 当前没有考虑 fragmentation。
- `lmcache/v1/storage_backend/local_disk_backend.py:369`
  - TODO: enable real batching。
- `lmcache/v1/storage_backend/local_disk_backend.py:524`
  - TODO: disk memory object 需要 ref count。
- `lmcache/v1/storage_backend/local_disk_backend.py:531`
  - TODO: 如果 freed memory object 立即被复用，当前逻辑可能有问题。
- `lmcache/v1/storage_backend/local_disk_backend.py:563`
  - TODO: 处理 loading 失败。

### 6.2 当前问题

CPU 和 disk backend 都在做容量控制，但目前偏“总量统计”，不是真正 allocator 级别的空间管理。

CPU 侧：

- allocator 能分配 MemoryObj。
- eviction 通过 cache policy 找 candidate。
- candidate 数现在经常是 1。
- 如果碎片严重，可能多次 evict 才能满足一个大 allocation。

Disk 侧：

- 通过 `current_cache_size + required_size > max_cache_size` 控制容量。
- 没有显式空闲区间管理。
- 文件删除后空间由文件系统处理，但 backend 不知道碎片形态。
- repeated prefetch 和 async save 可能有竞态。

### 6.3 影响

在高负载下，这些 TODO 可能表现为：

- allocation latency 抖动。
- eviction 过于保守，一次只淘汰一个 chunk。
- eviction 过于激进时降低命中率。
- disk 明明有总空间，但局部 layout 或 I/O pattern 不理想。
- async write 中的 MemoryObj 被提前释放或复用。
- clear/remove 统计不准，影响 metrics 和调度决策。

### 6.4 建议实现方向

#### 方向 A: MemoryObj metadata 缓存 size

把 size 分清楚：

- logical size: 有效 KV 数据大小。
- physical size: 实际分配大小，可能包含 alignment。
- token count: 这个 MemoryObj 覆盖多少 token。

建议放进 `MemoryObjMetadata`：

```text
MemoryObjMetadata {
  shape,
  dtype,
  fmt,
  logical_size_bytes,
  physical_size_bytes,
  num_tokens
}
```

这样可以减少热路径反复计算，也能改善 clear/metrics。

#### 方向 B: byte-targeted batch eviction

当前 eviction candidate 经常是 `num_candidates = 1`。更合理的是：

```text
get_evict_candidates(target_bytes, max_candidates, respect_pinned=True)
```

cache policy 可以返回足够释放目标大小的一组 key。

这需要 policy 能看到：

- 每个 key 的 physical size。
- ref count。
- pinned 状态。
- layerwise key 是否需要整组淘汰。

#### 方向 C: disk allocator

Disk backend 应该有独立 allocator，负责：

- 文件命名/placement。
- free range 或 segment 管理。
- alignment。
- fragmentation stats。
- reserve/commit/rollback。

异步 save 前先 reserve，写成功后 commit，失败后 rollback。

#### 方向 D: async write ref-count 协议

`local_disk_backend.py:531` 的 TODO 指出：如果 MemoryObj ref-count down 后立即被复用，可能出问题。

建议规则：

- 异步任务入队前 `ref_count_up`。
- 真正写入完成并复制完 metadata 后 `ref_count_down`。
- 如果 backend 只是使用 `memoryview` 指向源 buffer，必须保证写完成前源 buffer 不会复用。
- 如果做了内部 copy，则可以提前释放源 MemoryObj，但要明确记录。

### 6.5 验证建议

需要构造压力测试：

- 小 chunk 和大 chunk 混合 allocate/free。
- pinned chunk 阻止 eviction。
- ref_count > 1 时尝试 eviction。
- async disk save 后立即复用源 MemoryObj。
- repeated prefetch 同一 key。
- clear 时同时有写入或读取。

## 7. 数据调度、prefetch、lookup 和 location-aware retrieval

### 7.1 相关 TODO

- `lmcache/v1/cache_engine.py:1155`
  - TODO: layerwise retrieve 未来支持 multi-location。
- `lmcache/v1/cache_engine.py:1507`
  - TODO: async lookup + prefetch 增加 layerwise 支持。
- `lmcache/v1/storage_backend/storage_manager.py:510`
  - TODO: get_non_blocking 纳入 prefetch。
- `lmcache/v1/storage_backend/storage_manager.py:578`
  - TODO: async loading 和 layerwise 兼容。
- `lmcache/v1/storage_backend/storage_manager.py:711`
  - TODO: 当前 retrieval 假设 prefix-based，非 prefix 或中间 chunk miss 需要优化。
- `lmcache/v1/storage_backend/local_disk_backend.py:37`
  - TODO: repeated prefetch。
- `lmcache/v1/multiprocess/server.py:195`
  - TODO: stale prefetch jobs periodic cleanup。
- `lmcache/v1/cache_controller/controllers/kv_controller.py:380`
  - TODO: prefix chunks 被 evict 但 suffix chunk 还在时，当前实现处理不了。
- `lmcache/v1/cache_controller/controllers/kv_controller.py:384`
  - TODO: 当前 lookup 不考虑 KV chunks location，只返回最长 prefix 的 instance。
- `lmcache/v1/cache_controller/controllers/kv_controller.py:402`
  - TODO: 改进 matching logic，返回多个结果。
- `lmcache/v1/cache_controller/worker.py:547`
  - TODO: 当前 move 只支持 local disk 到 local CPU。
- `lmcache/v1/cache_controller/worker.py:553`
  - TODO: prefetch 和 move 需要对齐。

### 7.2 当前问题

现在 retrieval 逻辑基本偏 prefix-based：

- 从前往后找连续命中的 chunk。
- 假设 suffix 更容易被 evict。
- 中间 chunk miss 后，后面的 chunk 即使存在也不一定被利用。
- controller lookup 返回的信息比较粗，偏最长前缀。
- layerwise retrieve 目前要求所有 retrieved keys 来自同一 location。

这对简单 prefix sharing 有效，但对更复杂场景不够：

- RAG 中 prompt 中间部分复用。
- 多轮对话里部分历史被压缩/移动/evict。
- 某些 chunk 在 CPU，某些在 disk，某些在 P2P。
- layerwise 数据按层或按 chunk 分散在不同 location。
- prefetch 已经发起但还没完成。

### 7.3 影响

调度不够细会导致两类问题：

1. 有 cache 但用不上。
   - 中间 miss 后面的 hit 被放弃。
   - P2P 或远端命中因为 location 信息不足无法排序。

2. 用了 cache 但不划算。
   - 远端取回太慢，不如 recompute。
   - prefetch 没有和 layerwise compute 重叠。
   - 多 location retrieve 被 assert 限制。

### 7.4 建议实现方向

#### 方向 A: lookup 返回 segment，而不是只返回最长 prefix

建议 controller 和 StorageManager 的 lookup 都返回分段结果：

```text
[
  {
    range: [0, 256),
    keys: [...],
    location: "LocalCPUBackend",
    instance: "A",
    state: "ready",
    estimated_latency_ms: 0.2
  },
  {
    range: [256, 512),
    keys: [...],
    location: "P2PBackend",
    instance: "B",
    state: "remote_ready",
    estimated_latency_ms: 1.5
  },
  {
    range: [768, 1024),
    keys: [...],
    location: "LocalDiskBackend",
    instance: "A",
    state: "prefetchable",
    estimated_latency_ms: 3.0
  }
]
```

CacheEngine 再根据阈值决定：

- 取哪些 segment。
- 哪些 segment prefetch。
- 哪些 segment recompute。
- 是否允许 missing middle。

#### 方向 B: prefetch future 标准化

现在不同 backend 的 async/prefetch 返回值不统一。建议定义：

```text
PrefetchResult {
  lookup_id,
  key,
  location,
  memory_obj,
  ready,
  error,
  bytes,
  ownership
}
```

这样 CacheEngine 可以统一等待：

- local disk prefetch
- GDS prefetch
- P2P transfer
- PD transfer
- NIXL transfer

#### 方向 C: layerwise + multi-location

layerwise retrieve 需要支持：

- layer 0 来自 CPU，layer 1 来自 disk。
- chunk 0 来自 CPU，chunk 1 来自 P2P。
- 部分 layer 已 ready，部分 layer in-flight。

这要求 retrieval plan 能表达二维结构：

```text
layer_id x chunk_id -> location/state/future
```

短期可以先支持 chunk-level multi-location，不支持 layer-level multi-location。这样风险低一些。

#### 方向 D: prefetch 和 move 统一

controller worker 里的 TODO 指出 move 和 prefetch 没对齐。实际上二者都可以看成：

```text
ensure key exists at target location
```

区别只是：

- prefetch 不一定删除 old location。
- move 可能删除 old location。
- copy 会保留 old location。

可以抽象成：

```text
RelocationPlan {
  source_location,
  target_location,
  do_copy,
  priority,
  async_mode
}
```

### 7.5 验证建议

需要场景测试：

- prefix hit + suffix miss。
- middle chunk miss + later chunks hit。
- CPU/disk/P2P 同时存在同一 key。
- layerwise retrieve across locations。
- prefetch 发起后请求取消，stale job cleanup。
- controller 返回多个 candidate，CacheEngine 选择最快。

## 8. P2P、PD、NIXL 和分布式传输

### 8.1 相关 TODO

P2P：

- `lmcache/v1/storage_backend/p2p_backend.py:294`
  - TODO: 实现 local lookup cache。
- `lmcache/v1/storage_backend/p2p_backend.py:337`
  - TODO: 可以在 controller lookup 或 tier 3 lookup 后更新 local cache。
- `lmcache/v1/storage_backend/p2p_backend.py:406`
  - TODO: local CPU backend 没必要走 async 调用，async overhead 可避免。
- `lmcache/v1/storage_backend/p2p_backend.py:455`
  - TODO: 支持更多 backend。

PD：

- `lmcache/v1/storage_backend/pd_backend.py:170`
  - TODO: async ZMQ context，提高异步性。
- `lmcache/v1/storage_backend/pd_backend.py:289`
  - TODO: batched allocate 降低分配 overhead。
- `lmcache/v1/storage_backend/pd_backend.py:404`
  - TODO: batched submit put task 未来改 async。
- `lmcache/v1/storage_backend/pd_backend.py:448`
  - TODO: 和 transfer channel 解耦。
- `lmcache/v1/storage_backend/pd_backend.py:455`
  - TODO: 考虑 real async。
- `lmcache/v1/storage_backend/pd_backend.py:462`
  - TODO: transfer 完成后的 ref-count 管理可能应该移到 transfer channel。
- `lmcache/v1/storage_backend/pd_backend.py:534`
  - TODO: busy-loop allocation 应该属于 memory allocator，而不是 backend。

NIXL：

- `lmcache/v1/storage_backend/nixl_storage_backend.py:852`
  - TODO: async NIXL operations 需要 callback support。
- `lmcache/v1/transfer_channel/nixl_channel.py:444`
  - TODO: tune hyperparameters。
- `lmcache/v1/transfer_channel/nixl_channel.py:495`
  - TODO: tune hyperparameters。
- `lmcache/v1/transfer_channel/nixl_channel.py:537`
  - TODO: tune hyperparameters。
- `lmcache/v1/distributed/l2_adapters/nixl_store_l2_adapter.py:371`
  - TODO: transfer polling sleep 需要调优。
- `lmcache/v1/distributed/l2_adapters/nixl_store_l2_adapter.py:378`
  - TODO: support eviction。
- `lmcache/v1/distributed/l2_adapters/nixl_store_l2_adapter.py:605`
  - TODO: optimize lock usage。
- `lmcache/v1/distributed/l1_manager.py:223`
  - TODO: TTLLock.lock 支持 count 参数，避免 Python for-loop。
- `lmcache/v1/distributed/l1_manager.py:344`
  - TODO: TTLLock.unlock 支持 count 参数。
- `lmcache/v1/distributed/memory_manager.py:158`
  - TODO: Lazy allocator expansion 前 RDMA registration 是否可行需要测试。

### 8.2 当前问题

这些 TODO 共同指向一个问题：分布式数据传输缺少统一的 async completion 和 ownership 模型。

当前可能出现：

- transfer channel 里 hardcode polling sleep。
- backend 自己决定何时 ref-count down。
- NIXL async 操作没有 callback。
- PD transfer 名义上 batched，但内存分配和完成管理仍同步。
- P2P 每次 lookup 都可能访问 controller，没有 local lookup cache。
- lock/unlock 对每个单位循环调用，Python overhead 不必要。

### 8.3 影响

对分离式 prefill 或多实例共享来说，remote cache hit 的收益取决于：

- lookup 多快。
- remote allocation 多快。
- transfer 能不能和计算 overlap。
- transfer completion 能不能准确通知上层。
- 数据是否能写回 local cache，避免下次重复 remote fetch。

如果这些路径同步或 polling 太粗糙，remote hit 可能反而拖慢请求。

### 8.4 建议实现方向

#### 方向 A: transfer result / callback 标准化

所有远端 transfer 最好都返回统一 future：

```text
TransferFuture {
  keys,
  memory_objs,
  source,
  destination,
  bytes,
  state,
  error,
  on_complete
}
```

NIXL/PD/P2P/GDS 都可以挂在这个抽象下。

#### 方向 B: polling 策略下沉

`wait_time = 0.001`、`await asyncio.sleep(0.01)` 这类超参不应该散落在 backend 里。建议：

- transfer channel 配置 polling policy。
- 支持 busy-poll、sleep-poll、callback 三种模式。
- 根据 object size 或 SLA 动态选择。

#### 方向 C: P2P local lookup cache

P2P TODO 很明确。local lookup cache 可以缓存：

```text
chunk_hash -> {peer, location, timestamp, confidence}
```

失效策略：

- controller eviction event。
- TTL。
- transfer failure。
- peer disconnect。

#### 方向 D: local backend 避免 async overhead

`p2p_backend.py:406` 说明 local CPU backend 走 async 可能没必要。可以给 backend 增加 capability：

```text
backend.supports_fast_sync_contains
backend.supports_async_contains
```

调度器按 capability 选择路径。

### 8.5 验证建议

- P2P local lookup cache hit/miss latency。
- controller lookup QPS 压测。
- NIXL polling interval sweep。
- PD sync vs async transfer 对 TTFT 的影响。
- NIXL callback 是否正确释放 MemoryObj。
- peer disconnect 后 local lookup cache 是否失效。

## 9. 远端 connector、S3、Redis、GDS 和 zero-copy

### 9.1 相关 TODO

Redis：

- `lmcache/v1/storage_backend/connector/redis_connector.py:25`
  - TODO: 使用 `redis.asyncio`。
- `lmcache/v1/storage_backend/connector/redis_connector.py:299`
  - TODO: 找办法 inplace get。
- `lmcache/v1/storage_backend/connector/redis_connector.py:304`
  - TODO: 更好处理 consistency issue。
- `lmcache/v1/storage_backend/connector/redis_connector.py:306`
  - TODO: metadata 和 KV cache 聚合到一个 key 可能更好。
- `lmcache/v1/storage_backend/connector/redis_connector.py:545`
  - TODO: background sweeper 可能更适合性能。

LM server connector：

- `lmcache/v1/storage_backend/connector/lm_connector.py:26`
  - TODO: 该 class 性能优化，考虑 C/C++/Rust 做通信和反序列化。
- `lmcache/v1/storage_backend/connector/lm_connector.py:56`
  - TODO: `receive_all` 应该是 async。
- `lmcache/v1/storage_backend/connector/lm_connector.py:61`
  - TODO: 支持 compressed memory format 后会用 format 字段。
- `lmcache/v1/storage_backend/connector/lm_connector.py:138`
  - TODO: `get` 应该是 async function。

S3：

- `lmcache/v1/storage_backend/connector/s3_connector.py:178`
  - TODO: `_get_object_size` 用 async 优化。
- `lmcache/v1/storage_backend/connector/s3_connector.py:308`
  - TODO: 更细粒度 data partition。
- `lmcache/v1/storage_backend/connector/s3_connector.py:327`
  - TODO: 支持 offset 以启用 zero-copy，具体需要 shared memory offset。

GDS：

- `lmcache/v1/storage_backend/gds_backend.py:43`
  - TODO: 读取 4KB metadata block 时避免触发 read-ahead。
- `lmcache/v1/storage_backend/gds_backend.py:719`
  - TODO: prefetch interface 确定后需要修改。
- `lmcache/v1/storage_backend/gds_backend.py:847`
  - TODO: 当前只是 dummy wrapper around prefetch。

### 9.2 当前问题

远端 connector 的共性问题：

- metadata 和 KV bytes 分开存储时，容易出现一致性问题。
- get 路径往往需要先 allocate，再把 bytes 拷进 MemoryObj。
- async 支持不一致。
- in-place receive / zero-copy 不完善。
- S3/GDS 这类 backend 的 range/prefetch/offset 能力尚未完全接入统一调度。

### 9.3 影响

这些问题主要影响：

- 大 KV chunk 的远端读取 latency。
- CPU 内存额外 copy。
- remote cache consistency。
- prefetch 是否能精确加载需要的范围。
- GDS 是否能发挥 direct-to-GPU 的优势。

### 9.4 建议实现方向

#### Redis

- 优先统一 metadata + KV bytes 的写入协议，减少 split-brain。
- 若保留两个 key，需要 background sweeper 清理孤儿 metadata 或孤儿 KV。
- 探索 RESP/native client 的 in-place read，避免 Python bytes 中转。

#### S3

- 支持 range GET，用于只取需要的 KV segment。
- 将 shared memory offset 暴露给 S3 write callback。
- `_get_object_size` async 化，避免阻塞调度线程。

#### GDS

- 先定义 prefetch contract：
  - prefetch 是否 reserve memory？
  - 返回 Future 还是 bool？
  - completion 后 MemoryObj ownership 属于谁？
- 然后再实现真正 direct-to-GPU prefetch。

## 10. CacheGen、serde 和 CUDA kernel 优化

### 10.1 相关 TODO

Python serde：

- `lmcache/storage_backend/serde/cachegen_encoder.py:78`
  - TODO: helper 可以优化，不需要当前这种 concat。
- `lmcache/storage_backend/serde/cachegen_encoder.py:380`
  - TODO: `permute` 很贵，需要低层更好方式。
- `lmcache/storage_backend/serde/cachegen_basics.py:141`
  - TODO: 也许用 NumPy array，方便 `tobytes()` / `frombuffer()`。
- `lmcache/v1/storage_backend/naive_serde/cachegen_encoder.py:39`
  - TODO: serialize 中很多 memory copy 可以避免。
- `lmcache/v1/storage_backend/naive_serde/cachegen_encoder.py:52`
  - TODO: 直接在 gpu connector 里做 serialization，避免 copy。
- `lmcache/v1/storage_backend/naive_serde/cachegen_decoder.py:64`
  - TODO: deserialize 中很多 memory copy 可以避免。

CUDA：

- `csrc/mem_kernels.cu:237`
  - TODO: HND format 需要专用 kernel 改善 memory coalescing。
- `csrc/ac_enc.cu:30`
  - TODO: little endian 可以直接写 4 bytes。
- `csrc/ac_enc.cu:367`
  - TODO: block 负责 256 channel，这个 256 也许可配置。
- `csrc/ac_enc.cu:377`
  - TODO: 输入 token 太多时 shared memory 不够，当前边界约 256 tokens。
- `csrc/ac_enc.cu:420`
  - TODO: 可用 PackedAccessor32 访问 tensor，处理非 contiguous tensor。
- `csrc/ac_dec.cu:119`
  - TODO: 用 packed 32-bit read 替代 8-bit read。
- `csrc/ac_dec.cu:169`
  - TODO: 实现 binsearch。
- `csrc/ac_dec.cu:257`
  - TODO: 用 packed 32-bit read 替代 8-bit read。
- `csrc/ac_dec.cu:306`
  - TODO: 实现 binsearch。

### 10.2 当前问题

CacheGen 的目标是压缩 KV cache，降低存储和传输量。但目前 encode/decode 路径还有额外成本：

- Tensor layout 需要 `permute`。
- Python 层可能 `torch.cat` 或构造临时 list。
- MemoryObj 先 copy 到 CUDA tensor，再 serialize。
- deserialize 也可能多次中转。
- CUDA 编解码 kernel 的访存粒度和 shared memory 使用还可优化。

### 10.3 影响

压缩只有在下面不等式成立时才划算：

```text
encode_time + compressed_transfer_time + decode_time
<
uncompressed_transfer_time
```

如果 encode/decode 里有大量额外 copy 或 layout transform，小 chunk 或低带宽收益场景下可能反而变慢。

### 10.4 建议实现方向

- 把 layout transform 下沉到 CUDA kernel。
- store 方向融合 GPU connector extraction 和 CacheGen encode。
- load 方向融合 CacheGen decode 和 GPU connector writeback。
- 用 reusable workspace 替代 Python 临时对象和 `torch.cat`。
- 对 token 数超过 shared memory 边界的输入做分段 encode。
- 对 packed read/write 和 binsearch 做 kernel-level benchmark。

### 10.5 验证建议

每个压缩 benchmark 至少同时报告：

- compression ratio。
- encode latency。
- decode latency。
- 总 store latency。
- 总 retrieve latency。
- GPU temporary allocation 数。
- D2D/H2D/D2H copy 次数。

## 11. 建议执行路线

### Phase 1: 先去掉最明显的逐 chunk overhead

目标：低风险、收益快。

任务：

- 预分配 vLLM slot mapping device buffer。
- 去掉或复用 `gpu_connectors.py:1443` 的 `torch.cat` workspace。
- 给 GPU connector 增加 batch transfer descriptor。
- `CacheEngine.store` 使用 `batched_allocate`。
- `batched_remove` 保留 usage tracking。

预期收益：

- Python overhead 降低。
- kernel launch 和临时 tensor 分配减少。
- 多 chunk 请求 TTFT 更稳定。

### Phase 2: 让 async + layerwise + prefetch 成为统一路径

目标：把 I/O、GPU transfer 和 compute overlap 起来。

任务：

- 定义 retrieval segment plan。
- 支持 multi-location retrieve。
- async lookup + prefetch 支持 layerwise。
- StorageManager prefetch 返回结构化 Future。
- stale prefetch cleanup。

预期收益：

- 磁盘/远端 cache hit 更容易隐藏延迟。
- layerwise 模式不再受单 location 限制。

### Phase 3: 改善 memory pressure 下的行为

目标：减少 allocation stall 和错误淘汰。

任务：

- MemoryObj metadata 缓存 size/token count。
- byte-targeted batch eviction。
- disk allocator 和 fragmentation stats。
- allocator 层统一 busy-loop/retry。
- async write ref-count 协议。
- NIXL memory leak owner 明确化。

预期收益：

- CPU/disk under pressure 更稳定。
- metrics 更准确。
- 异步写入和复用更安全。

### Phase 4: 分布式 transfer 和远端 backend 深化

目标：让 remote hit 真正可调度、可预测、可 overlap。

任务：

- NIXL/PD completion callback。
- polling policy 配置化和自适应。
- P2P local lookup cache。
- controller location-aware lookup。
- S3 range read 和 zero-copy offset。
- Redis metadata/KV 一致性协议。
- GDS prefetch contract 完整实现。

预期收益：

- P2P/PD/NIXL 命中延迟下降。
- 远端 storage 更适合生产流量。
- 多级 cache 策略更智能。

## 12. 关键设计问题

后续真正动手前，建议先明确这些问题：

1. layerwise key 的一致性 invariant 是什么？
   - 所有 layer 一起创建/evict/move？
   - 还是允许 per-layer availability？

2. StorageManager 是否应该统一拥有 admission/write-back policy？
   - 还是每个 backend 自己决定？

3. prefetch 是否应该提前 reserve memory？
   - 提前 reserve 可以避免完成后无内存。
   - 但可能占住内存导致其他请求受影响。

4. MemoryObj 引用计数协议如何统一？
   - 调用方释放？
   - backend 消费？
   - async transfer 持有？

5. controller lookup 是否应该返回完整 segment 列表？
   - 而不是最长 prefix。

6. remote hit 和 recompute 如何比较？
   - 需要 latency model。
   - 需要 backend bandwidth/queue depth。
   - 需要考虑 GPU 当前负载。

7. 是否需要统一 transfer descriptor？
   - GPU/CPU/disk/P2P/PD/NIXL 是否共享一个抽象？
   - 还是只在每层内部各自抽象？

## 13. 快速索引

| 文件 | 主题 | 优化信号 |
| --- | --- | --- |
| `lmcache/v1/gpu_connector/gpu_connectors.py` | GPU connector | 真 batching、去 `torch.cat`、减少 slot mapping 重复处理 |
| `lmcache/v1/gpu_connector/xpu_connectors.py` | XPU connector | 真 batching |
| `lmcache/integration/vllm/vllm_v1_adapter.py` | vLLM 适配 | 预分配 slot mapping buffer |
| `lmcache/v1/cache_engine.py` | CacheEngine | batched allocate、batched cleanup、layerwise prefetch、multi-location retrieve |
| `lmcache/v1/storage_backend/storage_manager.py` | StorageManager | prefetch、write-back policy、async layerwise、非 prefix retrieval |
| `lmcache/v1/storage_backend/local_cpu_backend.py` | CPU backend | eviction candidate 估计、usage tracking、准确 token accounting |
| `lmcache/v1/storage_backend/local_disk_backend.py` | disk backend | disk allocator、fragmentation、ref-count、repeated prefetch |
| `lmcache/v1/storage_backend/p2p_backend.py` | P2P | local lookup cache、cache update、减少 async overhead |
| `lmcache/v1/storage_backend/pd_backend.py` | PD | batched allocation、real async、transfer ownership |
| `lmcache/v1/transfer_channel/nixl_channel.py` | NIXL channel | polling 超参、completion semantics |
| `lmcache/v1/distributed/l2_adapters/nixl_store_l2_adapter.py` | NIXL L2 | polling、eviction、lock usage |
| `lmcache/v1/storage_backend/connector/redis_connector.py` | Redis | async、in-place get、一致性 |
| `lmcache/v1/storage_backend/connector/s3_connector.py` | S3 | async object size、range read、zero-copy offset |
| `lmcache/v1/storage_backend/gds_backend.py` | GDS | prefetch contract、metadata read-ahead |
| `lmcache/storage_backend/serde` | CacheGen serde | 避免 permute、concat、Python bytes 中转 |
| `lmcache/v1/storage_backend/naive_serde` | CacheGen v1 serde | 避免 serialize/deserialize copy |
| `csrc/mem_kernels.cu` | CUDA KV transfer | HND coalescing kernel |
| `csrc/ac_enc.cu`, `csrc/ac_dec.cu` | CacheGen CUDA | packed read/write、shared memory 边界、binsearch |

## 14. 推荐下一步

如果要从这份路线图里挑一个最适合先做的任务，我建议从 **GPU connector real batching 的前置工作** 开始：

1. 给 vLLM adapter 加 slot mapping 预分配 buffer。
2. 去掉 V3 connector 中 `torch.cat` 的临时分配。
3. 做一个 benchmark，量化 chunk 数增长时的 overhead。
4. 再决定是否投入 descriptor-based CUDA kernel。

这个方向足够靠近核心路径，收益直观，而且不需要先重构 controller 或远端 backend 语义。
