# Segmentia vLLM 修改位置与代码路径

更新时间：2026-07-07

本文记录当前 Segmentia 系统依赖的本地 vLLM 修改。代码位于：

```text
/home/wsh/vllm
```

这些修改不是上游 vLLM 的普通配置项，而是当前工作树中的本地补丁。启动实验时必须使用
`opencode` 环境里指向 `/home/wsh/vllm` 的 patched vLLM。

## 结论

当前 vLLM 改动分成四层：

| 层级 | 作用 | 关键文件 |
|---|---|---|
| 请求解析 | 从 OpenAI-compatible 请求的 `vllm_xargs.context_segment_cache` 解析 sources / targets。 | `/home/wsh/vllm/vllm/v1/request.py` |
| 调度与 prefix-cache 边界 | 在 target_start 前限制普通 prefix cache；在 target_start 处分配 private KV slots 并生成 injection metadata；在 source span 完成后生成 registration metadata。 | `/home/wsh/vllm/vllm/v1/core/sched/scheduler.py`, `/home/wsh/vllm/vllm/v1/core/kv_cache_manager.py`, `/home/wsh/vllm/vllm/v1/context_segment_cache/scheduler.py` |
| GPU worker KV 操作 | 从 `.pt` 或进程内 registry 读取 K/V，校验 token identity，按 request block table scatter 到目标 span；可保存 source span K/V。 | `/home/wsh/vllm/vllm/v1/context_segment_cache/worker.py`, `/home/wsh/vllm/vllm/v1/context_segment_cache/registry.py`, `/home/wsh/vllm/vllm/v1/context_segment_cache/slot_ops.py`, `/home/wsh/vllm/vllm/v1/context_segment_cache/rope.py` |
| 诊断 hook | 默认关闭的 attention probe、function-vector probe、post-prefill patch，用于机制实验，不是正常在线路径。 | `/home/wsh/vllm/vllm/v1/context_segment_cache/attention_probe.py`, `/home/wsh/vllm/vllm/v1/context_segment_cache/function_vector_probe.py`, `/home/wsh/vllm/vllm/v1/context_segment_cache/post_prefill_patch.py`, `/home/wsh/vllm/vllm/v1/attention/backends/flash_attn.py`, `/home/wsh/vllm/vllm/model_executor/models/qwen3.py` |

核心状态边界仍然是：

```text
服务重启边界 = (mode, task)
task 内按 invocation_index 递增 replay
后续 occurrence 通过 prefix cache 继承历史注入后的 KV
当前 request 只显式注入当前 target span
不对历史 skill span 重复显式注入
```

## 请求侧接口

脚本侧通过 `scripts/06_context_free_segment_cache/module/vllm_client.py` 把
`context_segment_cache` 放进 OpenAI 请求的扩展字段：

```python
payload["vllm_xargs"] = {
    "context_segment_cache": json.dumps(context_segment_cache)
}
```

vLLM 在 `/home/wsh/vllm/vllm/v1/request.py` 解析该字段。结构是：

```json
{
  "sources": [
    {
      "cache_id": "case-source-cache",
      "source_start": 100,
      "source_end": 140
    }
  ],
  "targets": [
    {
      "cache_id": "skill-cache",
      "mode": "rope",
      "target_start": 200,
      "target_end": 240
    }
  ]
}
```

解析后的 request 字段：

```text
request.context_segment_kv_enabled
request.context_segment_kv_sources
request.context_segment_kv_targets
```

`mode` 当前支持：

```text
disabled: 跳过该 target
direct:   直接复用离线 key/value
rope:     value 直接复用，key 做 RoPE 位置修正后复用
```

当前版本会对未知 mode 直接抛错，不再静默降级为 disabled。

## Scheduler 改动

### 1. ContextSegmentKVScheduler

文件：

```text
/home/wsh/vllm/vllm/v1/context_segment_cache/scheduler.py
```

职责：

- 启动时读取 `VLLM_CONTEXT_SEGMENT_KV_DIR`，把目录下已有 `.pt` 的 stem 作为
  `known_cache_ids`。
- source span 被采集后调用 `mark_cache_collected(cache_id)`，让同一进程后续请求可复用。
- `get_max_cache_hit_length(request)` 返回已加载 target 中最早的 `target_start`。
- `get_injection(request, num_computed_tokens)` 只在
  `num_computed_tokens == target_start` 时返回 injection。
- 如果 `num_computed_tokens >= target_end`，说明该 target 已被当前 task 的 prefix cache
  覆盖，scheduler 会记录日志并跳过显式注入。
- `get_source_spans_covered(...)` 在正常 prefill 已覆盖 source span 时返回 registration。

### 2. prefix cache 命中上限

文件：

```text
/home/wsh/vllm/vllm/v1/core/kv_cache_manager.py
```

`KVCacheManager.get_computed_blocks()` 增加：

```python
max_cache_hit_length: int | None = None
```

普通请求不传该参数，行为不变。ContextSegmentKV target 请求传入最早
`target_start`，让普通 prefix cache 最多命中到注入点之前的完整 block。

例子：

```text
block_size = 16
target_start = 35
prefix cache 最多命中 32 tokens
剩余 3 tokens 正常 prefill 到 target_start
随后执行 KV 注入
```

这避免了旧逻辑为了不跳过注入点而完全禁用 prefix cache。

### 3. injection-only 调度步

文件：

```text
/home/wsh/vllm/vllm/v1/core/sched/scheduler.py
```

当 request 到达 `target_start`：

```text
allocate_slots(num_new_tokens = target_length)
num_scheduled_tokens[request_id] = 0
context_segment_kv_metadata.injections.append(...)
request.num_computed_tokens = target_end
```

这里分配的是当前 request 私有 KV slots，不是 KVConnector 的 external computed tokens。
该 engine step 不产生 query token、不 forward、不采样，只让 worker 在这些 slots 中写入
外部 K/V。

`update_from_output` 中对 `num_scheduled_tokens == 0` 的请求直接 continue，避免把
injection-only step 当成正常模型输出。

### 4. registration 调度

同一个 scheduler 在正常 prefill step 后检查 source span：

```text
old_num_computed_tokens < source_end <= old_num_computed_tokens + num_new_tokens
```

满足时生成 `ContextSegmentKVRegistration`，包含：

```text
req_id
cache_id
source_start/source_end
当前 request 的 block_ids
source span token_ids
```

这些 metadata 会随 `SchedulerOutput.context_segment_kv_metadata` 传给 GPU worker。

## GPU worker 改动

文件：

```text
/home/wsh/vllm/vllm/v1/worker/gpu_model_runner.py
/home/wsh/vllm/vllm/v1/worker/gpu/model_runner.py
```

主要调用顺序：

```text
_update_states(...)
apply_injections(...)
如果本步只有 injection，则不运行模型 forward
正常 forward
collect_registrations(...)
post_prefill_patch.maybe_apply(...)
postprocess / sampling
```

`gpu_model_runner.py` 是当前常用路径；`worker/gpu/model_runner.py` 中也有相同方向的
Segmentia hook，用于另一个 v1 runner 路径。

启动 worker 时会：

```text
self.context_segment_kv_worker = ContextSegmentKVWorker(...)
如果 VLLM_CONTEXT_SEGMENT_KV_DIR 存在，则 get_global_registry().load_dir(...)
```

## KV registry 与 `.pt` 格式

文件：

```text
/home/wsh/vllm/vllm/v1/context_segment_cache/registry.py
```

`.pt` payload 格式：

```python
{
    "cache_id": str,
    "source_start": int,
    "source_end": int,
    "token_ids": list[int] | None,
    "kv_by_layer": {
        layer_name: (key_tensor, value_tensor)
    },
}
```

大体量 `.pt` 默认不放在仓库 `results/`，而放在外存：

```text
SEGMENTIA_OUTPUT_DIR=/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/06_context_free_segment_cache
```

相关环境变量：

```text
VLLM_CONTEXT_SEGMENT_KV_DIR       # 启动时加载可复用 KV
VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR  # prefill source span 后保存 KV
```

## KV 写入逻辑

文件：

```text
/home/wsh/vllm/vllm/v1/context_segment_cache/worker.py
/home/wsh/vllm/vllm/v1/context_segment_cache/slot_ops.py
```

`slot_ops.py` 把逻辑 token span 映射到物理 KV cache slots：

```text
logical token position
-> logical block = pos // block_size
-> block offset = pos % block_size
-> physical slot = block_ids[logical_block] * block_size + block_offset
```

`ContextSegmentKVWorker.apply_injections(...)` 对每个 injection：

1. 从 registry 取 `cache_id` 对应的 `ContextSegmentKVEntry`。
2. 校验 length 一致。
3. 校验 source `token_ids` 与 target span 当前 `request.all_token_ids` 完全一致。
4. 如果 `mode=rope`，调用 `rerotate_k_for_target_positions(...)` 对 key 做 RoPE 修正。
5. 对每层调用 `scatter_span(...)`，把 key/value 写入当前 request 私有 slots。

`ContextSegmentKVWorker.collect_registrations(...)` 对每个 registration：

1. 用当前 request 的 `block_ids` 从每层 KV cache 调 `gather_span(...)`。
2. 构造 `ContextSegmentKVEntry` 放入进程内 registry。
3. 如果设置了 `VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR`，保存为 `<cache_id>.pt`。

## RoPE 修正

文件：

```text
/home/wsh/vllm/vllm/v1/context_segment_cache/rope.py
```

`mode=rope` 只修正 key：

```text
key:   从 source absolute positions 反旋，再按 target absolute positions 重旋
value: 直接复用
```

因此 rope 复用要求 source token IDs 与 target token IDs 一致；当前 worker 会强制校验。

## post-prefill patch

文件：

```text
/home/wsh/vllm/vllm/v1/context_segment_cache/post_prefill_patch.py
```

这是 marker relay / oracle patch 诊断用 hook，默认关闭。启用条件：

```text
VLLM_SEGMENTIA_POST_PREFILL_PATCH_CURRENT=/path/to/current_marker.json
```

执行位置：

```text
正常 forward 后
collect_registrations 后
logits/sampling 前
```

它读取 marker，找到当前 request 中唯一的 registration anchor，使用该 registration 的
block table 覆盖 marker 指定的 target span。支持：

```text
component=kv
component=key
component=value
```

安全检查包括：

- registration cache id 和 anchor span 必须唯一匹配。
- source cache 必须已加载。
- source/target length 必须一致。
- 当前 pilot 要求 source absolute positions 与 target positions 一致，避免未修正 RoPE。
- registration token IDs 和 target token IDs 必须匹配。
- 成功后写 `status_path`，记录实际 engine request ID、client request ID、patch 层数等。

注意：post-prefill patch 只改 KV cache，不改当前 hidden state 和首 token logits。

## attention probe

文件：

```text
/home/wsh/vllm/vllm/v1/context_segment_cache/attention_probe.py
/home/wsh/vllm/vllm/v1/attention/backends/flash_attn.py
```

启用条件：

```text
VLLM_SEGMENTIA_ATTENTION_PROBE_CURRENT=/path/to/current_attention_probe.json
VLLM_SEGMENTIA_ATTENTION_PROBE_OUT_DIR=/path/to/dump_dir
VLLM_SEGMENTIA_ATTENTION_PROBE_LAYERS=all 或 "0,1,2"
VLLM_SEGMENTIA_ATTENTION_PROBE_MAX_WINDOW_TOKENS=4096
```

调用点：

```text
gpu_model_runner.begin_step(...)
-> attention_probe.begin_step(...)
FlashAttentionImpl.forward(...)
-> attention_probe.maybe_dump_attention(...)
```

probe 只对 marker 指定的 query rows 重新计算 `q @ K`，不要求 FlashAttention 输出完整
attention matrix，也不改变模型输出。对于 prompt 内 query，会把 key 长度截断到
`query_abs_position + 1`，避免诊断时读取未来 prompt token。

输出是 JSONL，主要包括：

```text
attention_region_mass_*.jsonl
attention_local_windows_*.jsonl
```

## function-vector probe

文件：

```text
/home/wsh/vllm/vllm/v1/context_segment_cache/function_vector_probe.py
/home/wsh/vllm/vllm/model_executor/models/qwen3.py
```

启用条件：

```text
VLLM_SEGMENTIA_FV_CAPTURE_CURRENT=/path/to/current_capture.json
VLLM_SEGMENTIA_FV_CAPTURE_OUT_DIR=/path/to/head_outputs
```

调用点：

```text
gpu_model_runner.begin_step(...)
-> function_vector_probe.begin_step(...)
Qwen3Attention.forward(...)
-> attention output
-> o_proj
-> function_vector_probe.maybe_capture(...)
```

probe 在指定 `readout_abs_position` 采集每层：

```text
raw_head_output
projected_head_output
reconstruction_max_abs_error
reconstruction_relative_l2_error
```

当前要求 tensor_parallel_size=1，因为它直接按完整 `o_proj.weight` 重构逐 head residual
contribution。

## 当前工作树状态

截至本文件更新时间，`/home/wsh/vllm` 中与本说明相关的工作树状态包括：

```text
modified:
  tests/v1/core/test_prefix_caching.py
  tests/v1/segkv/test_scheduler.py
  vllm/model_executor/models/qwen3.py
  vllm/v1/attention/backends/flash_attn.py
  vllm/v1/context_segment_cache/metadata.py
  vllm/v1/context_segment_cache/scheduler.py
  vllm/v1/context_segment_cache/worker.py
  vllm/v1/core/kv_cache_manager.py
  vllm/v1/core/sched/scheduler.py
  vllm/v1/request.py
  vllm/v1/worker/gpu/model_runner.py
  vllm/v1/worker/gpu_model_runner.py

untracked:
  context_segment_kv_replay_slowdown_changes.md
  tests/v1/segkv/test_post_prefill_patch.py
  tests/v1/segkv/test_token_identity.py
  vllm/v1/context_segment_cache/attention_probe.py
  vllm/v1/context_segment_cache/function_vector_probe.py
  vllm/v1/context_segment_cache/post_prefill_patch.py
```

不要把“当前 server 能 import 到这些文件”误认为它们已经进入 git commit。迁移或提交
vLLM 补丁时必须显式纳入这些 untracked 文件。

## 相关测试

当前测试文件：

```text
/home/wsh/vllm/tests/v1/core/test_prefix_caching.py
/home/wsh/vllm/tests/v1/segkv/test_scheduler.py
/home/wsh/vllm/tests/v1/segkv/test_token_identity.py
/home/wsh/vllm/tests/v1/segkv/test_post_prefill_patch.py
```

覆盖内容：

- `max_cache_hit_length` 只允许 prefix cache 命中到完整 block 边界。
- scheduler 只对已加载 cache id 生成 injection。
- 多 target 时 prefix-cache 上限取最早已加载 target。
- source/target token identity mismatch 会失败。
- post-prefill patch 可覆盖 `kv`、`key`、`value`，并写 status。
- registration anchor 可与 patch target 分离，但 source/target 位置和 token identity
  仍需满足当前 pilot 约束。

本文档只做静态整理；本轮未启动 vLLM，也未运行 pytest。

