# ContextSegmentKV vLLM 代码调用链解读

本文档解释 `scripts/05_context_segment_agent_kv/run_real_multurn_context_segment.py`
如何一步一步调用到修改过的 vLLM ContextSegmentKV 代码。

整体实现分成两个阶段：

1. **离线 source 收集**：对一段稳定文本，例如 `internal-comms/SKILL.md`，
   做一次正常 prefill，然后把这段文本对应的每层 K/V cache 收集并保存成
   `.pt` 文件。
2. **在线 target 注入**：真实 agent 后续通过 `skill` 工具读到同一段 skill
   文本时，不重新计算这段文本的 KV，而是把离线保存的 KV splice 到在线请求
   的目标 token 位置。

## 入口脚本

启动脚本是：

`scripts/05_context_segment_agent_kv/run_real_multurn_context_segment.sh`

关键位置：

- 第 16-18 行设置 cache 目录：
  - `VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR`：离线收集 KV 后保存 `.pt` 文件的位置。
  - `VLLM_CONTEXT_SEGMENT_KV_DIR`：vLLM 启动时加载已有 `.pt` 文件的位置。
- 第 63-69 行重启 vLLM，使 vLLM 进程能看到这些环境变量。
- 第 73-82 行调用 `run_real_multurn_context_segment.py`。

Python 主入口是：

`scripts/05_context_segment_agent_kv/run_real_multurn_context_segment.py`

关键位置：

- 第 30-34 行定义默认 benchmark root、默认 repo、默认缓存的 skill 名称。
- 第 411-431 行解析命令行参数。
- 第 442-443 行准备 benchmark workspace 和实际运行用的 `.agents/skills`。
- 第 447-454 行逐个读取 `SKILL.md`，并调用 `offline_prefill_segment(...)`
  做离线 KV 收集。
- 第 471-480 行加载真实多轮 runner、创建 OpenHands agent、给 LLM 绑定在线
  注入 wrapper。
- 第 483-489 行运行真实多轮 benchmark。
- 第 491-500 行写出结果 JSON，包括：
  - `offline_context_segments`
  - `context_segment_kv_events`
  - `llm_calls`

## 阶段一：离线收集 Source KV

### 1. 读取 Skill 文本

在 `run_real_multurn_context_segment.py` 第 447-454 行：

- 第 448 行定位 `<active_skills_dir>/<skill_name>/SKILL.md`。
- 第 449 行读取 skill 全文。
- 第 450 行构造稳定的 cache id，例如：
  `context-segment-internal-comms-v1`。
- 第 453 行调用 `offline_prefill_segment(...)`。

### 2. 计算离线 source token span

`offline_prefill_segment(...)` 定义在第 254-304 行。

- 第 269 行把 segment 文本包装成一条 chat message：

```python
{"role": "user", "content": segment_text}
```

- 第 270-278 行调用 `span_token_offsets_for_message_text(...)`，参数是
  `char_start=0` 和 `char_end=len(segment_text)`。

`span_token_offsets_for_message_text(...)` 定义在第 218-251 行。

它的作用是把 message 里的字符范围转换成整个 chat prompt 里的 token 范围。

- 第 236-241 行分别构造两个截断后的 messages：
  - 一个截到 `char_start`
  - 一个截到 `char_end`
- 第 243-250 行分别调用 vLLM `/tokenize`。
- 第 251 行返回 `(start, end)`，这两个值就是该文本片段在 vLLM chat template
  下的 token offset。

`tokenize_chat(...)` 定义在第 80-105 行：

- 第 94-104 行 POST 到 `/tokenize`。
- 第 97-102 行传入和 generation 一致的参数，包括：
  - `model`
  - `messages`
  - `add_generation_prompt`
  - `chat_template_kwargs={"enable_thinking": True}`

这里必须走 vLLM `/tokenize`，因为 ContextSegmentKV 注入依赖 token 级 span，
不能只靠字符位置。

### 3. 发送带 `sources` 的离线请求

回到 `offline_prefill_segment(...)`：

- 第 279-287 行构造 `sources` 配置，例如：

```json
{
  "sources": [
    {
      "cache_id": "context-segment-internal-comms-v1",
      "source_start": 5,
      "source_end": 331
    }
  ]
}
```

- 第 288-296 行调用 `chat_completion(...)`，其中 `max_tokens=1`，
  `context_segment_cache=cfg`。

`chat_completion(...)` 定义在第 108-141 行：

- 第 124-131 行构造 OpenAI-compatible `/v1/chat/completions` payload。
- 第 133-138 行把 ContextSegmentKV 配置放到：

```json
"vllm_xargs": {
  "context_segment_cache": "{\"sources\": [...]}"
}
```

- 第 140 行 POST 到 `/v1/chat/completions`。

从这里开始，请求进入修改过的 vLLM。

## vLLM：解析 Request 里的 ContextSegmentKV 参数

vLLM 侧解析逻辑在：

`/home/wsh/vllm/vllm/v1/request.py`

关键位置：

- 第 180-182 行初始化每个 request 的 ContextSegmentKV 状态。
- 第 183-187 行读取 `sampling_params.extra_args["context_segment_cache"]`，
  如果是字符串则 JSON decode。
- 第 194-203 行把 `sources` 解析成 `ContextSegmentKVSourceSpan`。
- 第 204-218 行把 `targets` 解析成 `ContextSegmentKVTargetSpan`。
- 第 219-221 行设置 `context_segment_kv_enabled`。
- 第 222-228 行：如果这个请求有 target，则设置
  `skip_reading_prefix_cache=True`。

最后一点很关键：在线注入时 scheduler 必须停在精确的 `target_start`。如果普通
prefix cache 一次性跳过了 `target_start`，splice 点就错过了。

相关 dataclass 在：

`/home/wsh/vllm/vllm/v1/context_segment_cache/metadata.py`

关键位置：

- 第 6-8 行定义 mode：
  - `disabled`
  - `rope`
- 第 11-19 行定义 `ContextSegmentKVSourceSpan`。
- 第 22-31 行定义 `ContextSegmentKVTargetSpan`。
- 第 34-59 行定义 `ContextSegmentKVInjection`，其中 `with_blocks(...)`
  会把 scheduler 分配到的 KV block id 绑定进去。
- 第 62-69 行定义 `ContextSegmentKVRegistration`，离线保存 KV 时使用。
- 第 72-75 行定义 `ContextSegmentKVSchedulerMetadata`，用于 scheduler 把
  injection/registration 元数据传给 GPU worker。

## 阶段一继续：Scheduler 判断何时可以注册 Source

ContextSegmentKV 的 scheduler helper 在：

`/home/wsh/vllm/vllm/v1/context_segment_cache/scheduler.py`

关键位置：

- 第 11-20 行初始化 `known_cache_ids`，会从
  `VLLM_CONTEXT_SEGMENT_KV_DIR/*.pt` 读取已有 cache id。
- 第 22-23 行：新 cache 收集完成后，标记为 known。
- 第 67-83 行实现 `get_source_spans_covered(...)`。

`get_source_spans_covered(...)` 的作用是：当当前 prefill step 已经覆盖了某个
source span 的 `source_end`，就返回这个 source span，告诉 scheduler 可以在本轮
forward 后收集 KV。

主 scheduler 在：

`/home/wsh/vllm/vllm/v1/core/sched/scheduler.py`

关键位置：

- 第 291-295 行创建 `ContextSegmentKVScheduler`。
- 第 382-384 行创建本轮的 `ContextSegmentKVSchedulerMetadata`。
- 第 597-620 行处理 RUNNING request 的 source registration。
- 第 999-1020 行处理新调度或恢复的 request 的 source registration。
- 第 1129-1135 行：只有 metadata 里有 injections 或 registrations 时，
  才把它挂到 `SchedulerOutput` 上。

离线 source registration 里包含：

- request id
- cache id
- source token span
- 当前 request 的 KV block ids
- `request.all_token_ids[source_start:source_end]`

## 阶段一继续：GPU Worker 收集并保存 KV

GPU runner 在：

`/home/wsh/vllm/vllm/v1/worker/gpu_model_runner.py`

关键位置：

- 第 3820-3826 行执行正常的 model forward。
- 第 3827-3830 行调用：
  `self.context_segment_kv_worker.collect_registrations(...)`

worker 实现在：

`/home/wsh/vllm/vllm/v1/context_segment_cache/worker.py`

`collect_registrations(...)` 的关键位置：

- 第 94-100 行：如果没有 registration metadata，直接返回。
- 第 101-106 行校验 block ids。
- 第 107-116 行对每一层调用 `gather_span(...)`，从当前 request 的 KV cache
  里取出 source span 对应的 K/V。
- 第 117-123 行构造 `ContextSegmentKVEntry`。
- 第 124 行放入进程内 registry。
- 第 125-133 行：如果设置了 `VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR`，则保存成
  `<cache_id>.pt`。

底层 slot 操作在：

`/home/wsh/vllm/vllm/v1/context_segment_cache/slot_ops.py`

关键位置：

- 第 4-12 行把 KV cache tensor 拆成 key cache 和 value cache。
- 第 15-22 行把 `[num_blocks, block_size, num_kv_heads, head_dim]` 展平成
  slot 维度。
- 第 25-33 行根据 block ids 把逻辑 token 位置映射到物理 KV slot。
- 第 36-51 行实现 `gather_span(...)`。
- 第 54-72 行实现 `scatter_span(...)`。

cache registry 在：

`/home/wsh/vllm/vllm/v1/context_segment_cache/registry.py`

关键位置：

- 第 8-18 行定义 `ContextSegmentKVEntry`，包含：
  - cache id
  - source span
  - 每层 KV tensor
  - 可选 token ids
- 第 21-29 行实现内存中的 `put(...)` 和 `get(...)`。
- 第 34-48 行用 `torch.save` 保存 `.pt`。
- 第 50-65 行加载单个 `.pt`。
- 第 67-79 行加载目录下所有 `.pt`。
- 第 82-89 行提供全局 registry。

## 阶段二：在线 Target 注入

在线阶段从真实 OpenHands agent 开始运行 benchmark turns 后开始。

### 1. 创建真实 Agent

`run_real_multurn_context_segment.py` 复用：

`scripts/03_14B_anthropic/run_multurn3.py`

关键位置：

- 第 347-363 行创建 OpenHands `LLM`，base URL 指向 vLLM OpenAI API。
- 第 365-367 行给 LLM 绑定 request collector，用于记录 prompt 和指标。
- 第 384-387 行加载 skills，并添加 `SkillTool`。
- 第 389-407 行创建 `Agent`。

多轮 benchmark loop 在 `run_multurn3.py`：

- 第 258-277 行逐轮发送 turn message 并运行 `conversation.run()`。
- 第 280-309 行记录每次 LLM call 的 prompt、tool calls、vLLM 指标。
- 第 422-453 行从 `<bench-root>/<repo>/turns/turn_*.txt` 加载所有 turn。

### 2. 给 `_transport_call` 加在线注入 Wrapper

`run_real_multurn_context_segment.py` 第 474-480 行调用
`attach_context_segment_injector(...)`。

`attach_context_segment_injector(...)` 定义在第 307-385 行。

关键位置：

- 第 327 行保存原始 `llm._transport_call`。
- 第 328-330 行初始化 patch 状态、注入事件列表、seen span 集合。
- 第 336-342 行检查当前 LLM 请求的 `messages`，查找每个已离线缓存的
  segment 原文。
- 第 343-345 行保证同一个 `(cache_id, msg_idx, char_start, char_end)` 只注入
  一次。
- 第 347-355 行用同样的 `/tokenize` 方法计算在线 target token span。
- 第 356-363 行构造 target injection，例如：

```json
{
  "cache_id": "context-segment-internal-comms-v1",
  "mode": "rope",
  "target_start": 6021,
  "target_end": 6347
}
```

- 第 368-374 行把它挂到：
  `kwargs["extra_body"]["vllm_xargs"]["context_segment_cache"]`
- 第 375-380 行记录到 `llm._context_segment_kv_events`。
- 第 382 行调用原始 transport function。

segment 查找逻辑在 `locate_segment(...)`，第 199-215 行：

- 第 208-210 行跳过 system message。
- 第 211-214 行在非 system message 中做 exact substring match。

对当前 `internal_comms_incident_update` 运行来说，唯一一次在线注入是：

```text
cache_id=context-segment-internal-comms-v1
mode=rope
target=[6021,6347)
tokens=326
num_messages=8
```

它发生在 agent 调用 `skill(name="internal-comms")` 后。下一次 LLM 请求里包含
完整 `internal-comms` skill 文本，因此 wrapper 找到这段文本并附加 target。

## vLLM 启动时加载已有 Cache

vLLM 启动时，GPU model runner 会把 `.pt` 文件加载到全局 registry。

位置：

`/home/wsh/vllm/vllm/v1/worker/gpu_model_runner.py`

关键位置：

- 第 6541-6550 行创建 `ContextSegmentKVWorker`，传入全局 registry 和 KV
  block size。
- 第 6551-6559 行读取 `VLLM_CONTEXT_SEGMENT_KV_DIR`，调用
  `get_global_registry().load_dir(...)`。

所以后续 run 可以直接复用之前保存的 `.pt` 文件，不一定需要重新离线收集，
前提是 cache 目录、模型、chat template、skill 文本保持一致。

## 阶段二继续：Scheduler 停在 Target 并分配 KV Slot

在线请求进入 vLLM 后，`request.py` 已经把 target 配置解析到
`request.context_segment_kv_targets`。

ContextSegmentKV scheduler helper 在：

`/home/wsh/vllm/vllm/v1/context_segment_cache/scheduler.py`

`get_injection(...)` 定义在第 25-47 行：

- 第 28 行：没有启用 ContextSegmentKV 的 request 直接返回 `None`。
- 第 30-32 行读取 target spans。
- 第 33-35 行要求 target cache id 已知。
- 第 36-37 行要求 `num_computed_tokens == target.target_start`。
- 第 38-46 行返回 `ContextSegmentKVInjection`。

`cap_before_target_span(...)` 定义在第 49-65 行。

它的作用是防止 chunked prefill 一次跨过 `target_start`。如果某个 prefill chunk
会从 target 前面跨到 target 后面，就把 `num_new_tokens` 截断到刚好停在
`target_start` 前。

主 scheduler 使用它的位置：

`/home/wsh/vllm/vllm/v1/core/sched/scheduler.py`

对 RUNNING request：

- 第 393-396 行调用 `get_injection(...)`，判断当前 request 是否刚好停在
  target start。
- 第 397-403 行分配 KV slot，注意这里 `num_new_tokens=0`，
  `num_external_computed_tokens=segment_kv_injection.length`。
- 第 407-413 行绑定 block ids，并把 injection metadata 加到本轮 metadata。
- 第 414 行把 `request.num_computed_tokens` 推进到 `target_end`。

对 WAITING 或 resumed request：

- 第 764-767 行同样调用 `get_injection(...)`。
- 第 768-777 行为 external KV 分配 slot。
- 第 780-802 行把 request 放入 running 状态，绑定 injection metadata，并把
  `num_computed_tokens` 推进到 `target_end`。

注入这一步不会产生 logits，也不会采样 token。第 1551-1557 行明确跳过
`num_tokens_scheduled == 0` 的正常 token 处理，因为这一步只是把外部 KV splice
进 cache。

## 阶段二继续：GPU Worker 把离线 KV 写入在线 KV Cache

GPU runner 在 model forward 前应用 injection。

位置：

`/home/wsh/vllm/vllm/v1/worker/gpu_model_runner.py`

关键位置：

- 第 3598-3600 行更新 request 状态，并读取
  `scheduler_output.context_segment_kv_metadata`。
- 第 3602-3607 行调用：
  `self.context_segment_kv_worker.apply_injections(...)`
- 第 3608-3617 行把 runner 侧 request state 的 `num_computed_tokens` 更新到
  `injection.target_end`。

实际写 KV 的代码在：

`/home/wsh/vllm/vllm/v1/context_segment_cache/worker.py`

`apply_injections(...)` 关键位置：

- 第 36-37 行：没有 injection 就直接返回。
- 第 39-44 行按 `cache_id` 从 registry 里取离线 KV；如果没加载到就报错。
- 第 45-53 行校验 block ids 和长度。
- 第 54-58 行把离线 K/V tensor 移到当前 KV cache 所在 device。
- 第 59-72 行：如果 mode 是 `rope`，对 key 做 RoPE 位置修正。
- 第 73-81 行调用 `scatter_span(...)`，把 key/value 写入在线 request 的目标
  KV slot。
- 第 82-91 行打印日志：

```text
ContextSegmentKV: applied cache_id=... source=[...,...) target=[...,...) tokens=...
```

RoPE 修正在：

`/home/wsh/vllm/vllm/v1/context_segment_cache/rope.py`

关键位置：

- 第 8-12 行找到模型里的 rotary embedding 模块。
- 第 15-44 行实现 `rerotate_k_for_target_positions(...)`。
- 第 21-27 行只允许移动到相同或更靠后的位置。
- 第 34-41 行根据 `target_start - source_start` 计算位置偏移，并旋转 key。

## 当前运行的证据

当前 `internal_comms_incident_update` 的结果文件：

`results/05_context_segment_agent_kv/internal_comms_incident_update/multiturn_sequence_traces.json`

记录到 3 个离线 segment：

- `context-segment-internal-comms-v1`：source `[5,331)`，326 tokens
- `context-segment-slack-gif-creator-v1`：source `[5,2081)`，2076 tokens
- `context-segment-brand-guidelines-v1`：source `[5,533)`，528 tokens

同一个 JSON 里记录到 1 次在线注入：

- `context-segment-internal-comms-v1`：target `[6021,6347)`，326 tokens

`log/vllm.log` 里有对应证据：

```text
ContextSegmentKV: loaded 3 cache(s) from .../kv_cache:
['context-segment-brand-guidelines-v1',
 'context-segment-internal-comms-v1',
 'context-segment-slack-gif-creator-v1']

ContextSegmentKV: applied cache_id=context-segment-internal-comms-v1
mode=rope source=[5,331) target=[6021,6347) tokens=326
```

这说明：

- vLLM 启动时确实加载了 3 个离线 cache。
- 在线运行中实际应用了 `internal-comms` 的 KV 注入。
- `slack-gif-creator` 和 `brand-guidelines` 虽然被加载，但这次
  `internal_comms_incident_update` benchmark 没有让 agent 读取它们的完整 skill
  文本，所以没有触发这两个 cache 的在线注入。

## 关键约束和风险点

- source span 和 target span 的 token 长度必须一致。`worker.py` 第 49-53 行
  会检查长度。
- 当前实现不会校验 target token ids 是否和离线保存的 `token_ids` 完全一致。
  如果 `.pt` 是旧的，但 token 长度碰巧相同，当前检查不会发现 stale cache。
- 有 target 的请求会跳过本次 prefix-cache read，因为 scheduler 必须精确停在
  `target_start`，见 `request.py` 第 222-228 行。
- 一旦 skill 文本进入 agent 历史，后续轮次通常依赖普通 vLLM prefix cache，
  不会反复做 ContextSegmentKV 注入。
- `rope` 模式只支持把离线 span 移动到相同或更靠后的 token 位置，见
  `rope.py` 第 21-27 行。

