# CSKCache Connector Migration Development

## 总开发目标

把当前 Segmentia / Context Segment KV Cache 的复用逻辑迁移到
`/home/wsh/openhands_code_research/CSKCache` 下，并评估是否通过 vLLM
`KVConnectorBase_V1` connector 接入，而不是继续把所有逻辑直接写在
`/home/wsh/vllm/vllm/v1/context_segment_cache/` 中。

## 开发阶段总览

| 阶段 | 名称 | 目标 | 当前进度 | 剩余 |
|---|---|---|---|---|
| 1 | 结构摸底 | 对照 `CSKCache` 当前文件和 `/home/wsh/LMCache/lmcache` 重点模块。 | 已完成初步静态分析。 | 需要后续读 vLLM connector 调用点细节。 |
| 2 | 方案判断 | 判断 CSK 是否适合走 vLLM connector。 | 已形成修正建议：可以用 connector，但不能依赖 oracle `target_start/target_end`；第一版不要全量移植 LMCache。 | 需要用户确认 segment 自动发现与 connector 调度语义。 |
| 3 | 最小骨架 | 实现 CSKCache vLLM connector 最小可 import / 可初始化骨架。 | 已完成初版。 | 需要真实 vLLM 空跑启动验证。 |
| 4 | CSK segment 语义接入 | 自动发现请求中的可复用 segment occurrence，并把 lookup/load/save/slot mapping 接入 connector metadata。 | 已完成 A 版 registry-derived token matcher、load plan 和 RoPE key 相对位置修正；B 版 prompt metadata 留 TODO。 | 需要 scheduler boundary hook、save path 和真实模型 RoPE parity 验证。 |
| 5 | 验证 | 静态编译、connector import、空跑启动，再做小规模真实 replay。 | 已完成静态语法、matcher、slot ops 和 connector import 验证。 | 未启动 vLLM，未跑真实 replay。 |
| 6 | 旧 vLLM hook 禁用 | 禁用 `/home/wsh/vllm` 主执行路径中的旧 `context_segment_cache` 调度、worker 注入和 probe hook。 | 已完成主链路注释；旧模块目录保留归档。 | 需要后续把 CSKCache connector 接入 vLLM 启动配置后做空跑验证。 |
| 7 | Probe-gated injection | 对 skill span 先重算少量 probe token，按 RoPE 复用 KV 与真实重算 KV 的残差选择纯复用或 anchor 兜底重算。 | 首版已实现：CSKCache 新增 `v1/compute`，connector 状态机、worker meta 回传和 vLLM scheduler 小 hook 已接入。 | 需要真实 vLLM replay 验证、阈值标定、多 worker metric 聚合语义确认和质量/TTFT 对照。 |

## 当前分析结论

`CSKCache/cskcache/integration/vllm/` 已实现初版 vLLM connector，并补齐
CSKCache v1 核心模块：

```text
CSKCache/README.md
cskcache/__init__.py
cskcache/integration/__init__.py
cskcache/integration/vllm/__init__.py
utils.py
v1_adapter.py
v1_connector.py
cskcache/v1/__init__.py
cskcache/v1/matcher.py
cskcache/v1/metadata.py
cskcache/v1/registry.py
cskcache/v1/rope.py
cskcache/v1/slot_ops.py
```

LMCache 中和 vLLM connector 迁移最相关的是：

```text
/home/wsh/LMCache/lmcache/integration/vllm/lmcache_connector_v1.py
/home/wsh/LMCache/lmcache/integration/vllm/vllm_v1_adapter.py
/home/wsh/LMCache/lmcache/integration/vllm/utils.py
/home/wsh/LMCache/lmcache/v1/cache_engine.py
/home/wsh/LMCache/lmcache/v1/gpu_connector/gpu_connectors.py
/home/wsh/LMCache/lmcache/v1/storage_backend/storage_manager.py
/home/wsh/LMCache/lmcache/v1/token_database.py
```

不建议第一版全量搬运 LMCache 的 server、CLI、controller、multiprocess、observability
和多后端 storage 体系。CSKCache 的第一版目标应先是“能作为 vLLM connector 接入，
并能从请求 token 序列或 prompt 组装元数据中自动发现可复用 segment occurrence 后完成
KV load/save”，而不是复刻 LMCache。

推荐架构不是回到旧 `context_segment_cache`，也不是全量复刻 LMCache，而是：

```text
CSKCache package:
  负责 segment index、token matching、cache entry 选择、是否复用、load/save metadata。

vLLM connector:
  作为 CSKCache 与 vLLM scheduler/worker 的标准桥。

small vLLM hook:
  只提供 CSKCache 需要的调度边界能力，例如不要让 chunked prefill 跨过下一个
  segment start。vLLM 不理解 skill 语义。
```

旧 `context_segment_cache` 适合机制实验，因为它能用 oracle `target_start/target_end`
精确控制注入；但作为系统实现，它把策略、segment 边界和 vLLM 内部执行耦合得太紧。
CSKCache 应该把这部分策略移出 vLLM，只在 vLLM 里保留必要的低层执行接口。

## 当前风险

- vLLM connector 原生语义是 prefix-level external KV cache；当前 Segmentia 实验补丁
  使用 explicit `target_start/target_end` span injection，这是 replay / 机制诊断中的
  oracle 边界，不是实际系统可以假设的输入。
- 可部署 CSKCache 不能要求调用方预先知道 skill 的 token start/end。系统必须从请求
  token 序列、prompt 组装元数据，或可复查的 segment index 中自动发现可复用 segment
  occurrence，再把发现结果转成 scheduler / worker 可执行的 load 计划。
- Agent API 层通常每轮都会重新发送完整 chat prompt；历史 user/assistant/tool 内容
  虽然在之前请求中出现过，但仍会作为当前请求的前缀再次出现。vLLM prefix cache 可以在
  token 完全一致且服务未重启/缓存未清空时复用这些历史前缀。CSKCache 的 load 边界应以
  当前 request 的 `num_computed_tokens` 为准：如果 prefix cache 已经覆盖到 skill 之后，
  本轮无需显式 load；如果只覆盖到 skill 之前，则在自动发现的 skill start 处 load。
- 复用决策归 CSKCache，不归 vLLM。vLLM 不应理解 skill 语义或判断是否值得复用；但
  CSKCache 的决策必须翻译成 vLLM 可执行的调度信号，例如“本轮先 prefill 到某个
  segment start”“从当前 computed 位置 load 多少 tokens”“这些 tokens 写入哪些
  KV slots”。换言之，vLLM 只负责执行 block allocation、slot mapping 和 forward
  边界，策略和 cache entry 选择都属于 CSKCache。
- LMCache 也是这个边界：`lmcache_connector_v1.py` 只把 vLLM connector hook 转发给
  LMCache adapter；`vllm_v1_adapter.py#get_num_new_matched_tokens()` 在 scheduler 侧
  查询 LMCache lookup client，返回“除了 vLLM 本地 prefix cache 已算 token 外，外部
  LMCache 还能连续提供多少 token”；`build_connector_meta()` 再把 token_ids、
  slot_mapping、LoadSpec/SaveSpec 发给 worker；`start_load_kv()` 在 worker 侧按
  slot_mapping 写入 vLLM paged KV cache。LMCache 不让 vLLM 理解语义，但其主路径仍是
  prefix/chunk 命中，不是任意 skill 语义段。
- Connector 的 `get_num_new_matched_tokens()` 只能告诉 vLLM 从当前
  `num_computed_tokens` 之后额外命中多少连续 token；agent 当前待复用 skill 通常位于
  prompt 尾部、assistant generation marker 之前，但它仍然不是从 request 开头开始。
  因此 CSKCache 需要先让 scheduler 正常 prefill 到自动发现的 `segment_start`，然后再
  load 该 segment，而不是让一次 chunked prefill 直接跨过 segment。
- 原有 `context_segment_cache` worker 直接在实验指定的 `target_start` 处分配
  injection-only step；标准 connector 不天然提供“中间 span 注入点暂停”语义。CSKCache
  若走 connector，第一版必须先定义如何从 token matching / prompt metadata 得到这个
  暂停点；这里的暂停点是系统自动发现出来的，不是用户提供 oracle span。
- 如果第一版要保留 task 内 prefix-cache 继承历史注入，仍需明确服务重启边界为
  `(mode, task)`，task 内按 `invocation_index` replay，且不重复显式注入历史 span。
- 旧 `/home/wsh/vllm/vllm/v1/context_segment_cache/` 模块当前保留不删除，作为机制实验归档
  和后续迁移参考；但 vLLM 主执行路径中的 import、scheduler hook、worker injection、
  registration、post-prefill patch、attention probe 和 function-vector probe 已注释掉。
  后续长期方向是把可复用策略放到 `CSKCache`，只在 vLLM 保留 connector 需要的最小执行接口。

## 2026-07-08 只读复核

本轮阅读了 `lmcache_cacheblend_cacheslide_notes.md` 和 `CSKCache` 当前代码。
结论是：当前 `CSKCache` 实现更接近“vLLM connector + segment exact matching +
paged KV scatter + RoPE-only key correction”的最小原型，不包含 LMCache
CacheBlend 的 GPU layerwise mini-forward、diff/topk 覆盖修正，也不包含 CacheSlide
的 attention-inline WCA 融合逻辑。

当前代码数据流为：

```text
CSKCACHE_KV_DIR / kv_connector_extra_config
  -> registry.load_dir() 读取 .pt cache entry
  -> SegmentCatalog.from_entries() 从 token_ids 构造 exact-match catalog
  -> get_num_new_matched_tokens() 按 num_computed_tokens 查找 ready occurrence
  -> update_state_after_alloc() 保存 vLLM 分配的 block ids
  -> build_connector_meta() 把 CSKLoadPlan + block ids 传给 worker
  -> start_load_kv() 逐层读取 entry KV，必要时重旋转 key，再 scatter 到 paged KV cache
```

当前关键缺口仍然是：

- scheduler boundary hook 尚未实现；如果 occurrence 在 `num_computed_tokens` 之后，
  当前代码只记录 `_pending_boundaries` 并返回 0，不能阻止 chunked prefill 跨过 segment。
- prompt-builder metadata path 尚未实现；segment discovery 仍依赖 registry entry
  的 exact token subsequence matching。
- save path 仍是空实现，不能通过 CSKCache 收集 canonical segment KV。
- 没有真实 vLLM end-to-end parity / replay 验证。
- 没有 CacheBlend / CacheSlide 风格的 model-aware correction；当前只有 key 的 RoPE
  相对位置修正，value 直接复用。

本轮未修改 `CSKCache` 代码，未启动 vLLM，未运行实验。

## 2026-07-08 Probe-Gated Injection 初步方案

用户提出的 `inject_skill_span(skill)` 更接近 CacheBlend-style 架构：在 connector /
worker 侧基于真实模型计算少量 token 的 KV probe，再决定是否复用离线 KV，而不是直接
把 WCA 写进 attention forward。

对照 `/home/wsh/LMCache/lmcache` 后确认分层应保持如下边界：

```text
LMCache:
  integration/vllm/     负责接 vLLM connector hook、创建和调用 blender
  v1/compute/blend/     负责 CacheBlend 的 residual/topk/blend 策略
  v1/compute/models/    负责访问 vLLM model/layers 做 layerwise mini-forward

CSKCache:
  integration/vllm/     只负责把 vLLM scheduler/worker 状态转成 CSK plan/meta
  v1/compute/gate/      放 probe residual、threshold gate、decision metadata
  v1/compute/reuse/     放 partial KV slice、RoPE-shift、scatter/gather 组合逻辑
```

因此不能把 probe gate / residual 计算堆进
`CSKCache/cskcache/integration/vllm/v1_adapter.py`。`v1_adapter.py` 只应该调用
`cskcache.v1.compute` 暴露出的 planner/corrector API。

首版建议不要在 connector `start_load_kv()` 内部嵌套调用 `model.forward()` 来计算
`full_forward_recompute(skill)`。更稳的实现是让 vLLM scheduler 把 probe / anchor
作为正常 prefill 小步执行：

```text
DISCOVER skill span [s,e)
  -> normal prefill 到 s
  -> normal prefill [s,s+m) 作为 probe
  -> gather probe 的真实 recompute KV
  -> 与离线 KV 的 RoPE-shift 后 probe KV 计算残差 d
  -> 若 d <= tau，external-load [s+m,e)
  -> 若 d > tau，normal prefill [s+m,s+k)，再 external-load [s+k,e)
```

这样 probe / anchor 的 KV 已经自然写入 vLLM paged KV cache，不需要先写入
`kv_rope_reuse[s:e]` 再覆盖 probe。external load 只负责跳过尚未真实 prefill 的尾部。

需要新增或修改的关键能力：

- scheduler boundary hook：至少能把当前请求的 chunked prefill 截断在 `s`、`s+m`、
  以及触发兜底时的 `s+k`。
- partial load plan：当前 `CSKLoadPlan` 假设加载完整 entry；probe/anchor 方案需要能加载
  `source_offset -> target_start` 的部分 span，例如 `[m, len)` 或 `[k, len)`。
- probe KV gather：用 `slot_ops.gather_span()` 或 `save_kv_layer()` 从真实 prefill 后的
  paged KV cache 收集 `[s,s+m)` 的全层 K/V。
- RoPE-shift temp KV：只把 probe 所需的离线 K/V 切片搬到 GPU，按
  `target_start - (entry.source_start + source_offset)` 对 key 做相对旋转，再与
  recompute KV 比较。
- metric logging：同时记录 K/V 分开后的 mean、max-layer、p95 或 per-layer 列表，先不只
  固定一个 `d`，用标定结果决定 gate 用 mean 还是 max。

当前建议的内存/加载策略：

- skill markdown / token_ids 较小，可用 host-memory LRU 常驻，用于 prompt expansion、
  span 定位和 token identity 校验。
- 大 KV 不默认常驻 GPU；先保留在外存或 host memory，按请求、按层、按 slice 传到 GPU。
- RoPE 平移优先在 GPU 上按层/按 slice 执行，然后立即 scatter 或参与 residual 计算；
  不为不同 target position 持久化多份已平移 KV。

本节设计已完成首版实现，但尚未启动 vLLM，未运行真实 replay 实验。

### 拟实现状态机

首版状态机基于 request-local `req_id -> CSKProbeState`：

```text
IDLE
  find occurrence [s,e)
  if computed < s: ask scheduler cap at s
  if computed == s: ask scheduler prefill m probe tokens

PROBE_PREFILLED
  gather [s,s+m) recompute KV from paged cache
  build RoPE-shifted offline probe KV
  compute residual metrics
  if d <= tau: plan external load [s+m,e)
  else: ask scheduler prefill [s+m,s+k)

ANCHOR_PREFILLED
  plan external load [s+k,e)

LOADED
  clear per-request state
```

这个状态机必须配一个 vLLM scheduler 小 hook。原因是当前 vLLM 只在
`request.num_computed_tokens == 0` 时调用 connector 的
`get_num_new_matched_tokens()` 查询外部 KV；中间 skill span 如果没有
`cap_before_boundary` 能力，scheduler 可能一次 chunked prefill 跨过 `[s,e)`，
CSKCache 就没有机会在 `s`、`s+m` 或 `s+k` 处介入。

### 首版不做的内容

- 不在 `start_load_kv()` 内部嵌套调用完整 `model.forward()`；probe/anchor 通过 vLLM
  正常 prefill 得到。
- 不实现 CacheSlide attention-inline WCA。
- 不实现 LMCache 那种完整 layerwise mini-forward blender；后续若需要，可在
  `cskcache/v1/compute/models/` 追加，但不是本阶段最小实现。
- 不让大 KV 长期常驻 GPU；按 partial span / layer 切片加载并即时 scatter。

### 2026-07-08 实现记录

本轮实现了 probe-gated injection 的首版代码路径：

```text
CSKCache/cskcache/v1/compute/
  __init__.py
  gate.py
  reuse.py

CSKCache/cskcache/v1/metadata.py
CSKCache/cskcache/integration/vllm/utils.py
CSKCache/cskcache/integration/vllm/v1_adapter.py
CSKCache/cskcache/integration/vllm/v1_connector.py
CSKCache/README.md
/home/wsh/vllm/vllm/v1/core/sched/scheduler.py
```

实现后的数据流：

```text
1. cap_num_new_tokens()
   发现 registry-derived segment occurrence [s,e)，把正常 prefill 截断在 s、
   s+m 或 s+k。

2. 正常 vLLM prefill probe [s,s+m)
   build_connector_meta() 发送 CSKProbeMeta；worker 的 save_kv_layer() 从 paged KV
   gather recompute probe KV。

3. worker-side residual gate
   cskcache.v1.compute.gate 分层计算 K/V residual，记录 mean/max/per-layer，
   build_connector_worker_meta() 回传 CSKProbeDecision。

4. scheduler-side decision
   update_connector_output() 接收 decision。若通过，下一步 load [s+m,e)；若失败，
   先继续 prefill anchor [s+m,s+k)，再 load [s+k,e)。

5. in-process load
   scheduler duck-typed 调用 get_inprocess_load_tokens()，只分配 KV slots、不调度
   query token；worker start_load_kv() 按 partial source_offset 切片，GPU 上做 RoPE
   key 平移，然后 scatter 到 paged KV cache。
```

新增配置：

```text
cskcache.probe_enabled: bool, default False
cskcache.probe_tokens: int, default 4
cskcache.anchor_tokens: int, default 32
cskcache.probe_tau: float, default 0.15
cskcache.gate_metric: str, default max
```

当前实现边界：

- 默认 `probe_enabled=False`，不改变旧 direct reuse 行为；启用后才走 probe-gated path。
- `gate_metric=max` 表示使用 `max(k_max_layer, v_max_layer)` 作为门控值；同时保留 K/V
  mean、max 和 per-layer 记录用于标定。
- probe/anchor 不是 connector 内部 mini-forward，而是 vLLM 正常 prefill。
- vLLM scheduler hook 是 duck-typed：只有 connector 实现 `cap_num_new_tokens` /
  `get_inprocess_load_tokens` 时才生效，对其他 connector 透明。
- 多 worker 聚合当前通过 `CSKProbeWorkerMetadata.aggregate()` 简单拼接 decisions；
  scheduler 侧只在 `WAIT_PROBE` 状态消费第一轮有效 decision。TP/PP 下 metric 是否需要
  更严格聚合仍需真实验证。

本轮验证：

- 使用源码 `compile()` 检查 CSKCache 修改文件和
  `/home/wsh/vllm/vllm/v1/core/sched/scheduler.py`：通过。
- 使用 `opencode` 环境做 import/gate smoke：
  `CSKProbeAccumulator` 同值 K/V residual gate 通过，`CSKCacheConnectorV1` 可 import。
- `git diff --check` 检查 CSKCache 修改和 vLLM scheduler 修改：通过。
- 未启动 vLLM server，未运行 decode/replay 实验。

## 历史修改与验证

修改：

- 新增本 development 文档，记录 CSKCache connector 迁移分析状态。
- 更新 `CSKCache/README.md`，记录初版范围、KV entry metadata、vLLM动态加载方式和旧
  `context_segment_cache` 处理策略。
- 新增 `CSKCache/cskcache/v1/metadata.py`：定义 `CSKCacheMode`、segment、
  occurrence、KV entry 和 load plan。
- 更新 `CSKCache/cskcache/v1/matcher.py`：A版基于 registry entry token_ids 做
  exact matching，不再从 JSON 文件加载 catalog；TODO(B)：后续用 prompt-builder
  metadata 作为主来源。
- 更新 `CSKCache/cskcache/v1/registry.py`：加载旧 `.pt` 格式 KV entry，并暴露
  `entries()` 供 matcher 构建内存 segment index。
- 新增 `CSKCache/cskcache/v1/slot_ops.py`：按 vLLM block ids 对 paged KV cache
  gather/scatter。
- 更新 `CSKCache/cskcache/v1/rope.py`：实现 CSK RoPE key 相对位置修正。
  same-position 直接返回；different-position 使用 vLLM rotary helper 对 cached key
  施加 `target_start - source_start` 的相对旋转，value 不做旋转。
- 新增 `CSKCache/cskcache/integration/vllm/v1_connector.py`：实现
  `CSKCacheConnectorV1(KVConnectorBase_V1)` 动态加载入口。
- 新增 `CSKCache/cskcache/integration/vllm/v1_adapter.py`：实现 scheduler侧
  token matching / load plan / metadata，以及 worker侧从 registry scatter KV。
- 新增 package `__init__.py` 文件，保证 `PYTHONPATH=.../CSKCache` 后可以 import。
- 注释 `/home/wsh/vllm/vllm/v1/request.py` 中旧 `sampling_params.extra_args["context_segment_cache"]`
  解析逻辑，避免请求对象再携带旧 oracle span 配置。
- 注释 `/home/wsh/vllm/vllm/v1/core/sched/scheduler.py` 中旧
  `ContextSegmentKVScheduler` import、初始化、injection-only step、source registration、
  cache-hit 截断和 scheduler output metadata 传递。
- 注释 `/home/wsh/vllm/vllm/v1/core/sched/output.py` 中旧
  `context_segment_kv_metadata` 字段。
- 注释 `/home/wsh/vllm/vllm/v1/core/kv_cache_manager.py` 中旧
  `max_cache_hit_length` 参数逻辑，恢复普通 prefix-cache hit 上限语义。
- 注释 `/home/wsh/vllm/vllm/v1/worker/gpu_model_runner.py` 和
  `/home/wsh/vllm/vllm/v1/worker/gpu/model_runner.py` 中旧 worker 初始化、KV registry
  加载、在线注入、registration 收集、post-prefill patch 和 injection-only 请求过滤。
- 注释 `/home/wsh/vllm/vllm/model_executor/models/qwen3.py` 中旧 function-vector capture hook。
- 注释 `/home/wsh/vllm/vllm/v1/attention/backends/flash_attn.py` 中旧 attention probe hook。

验证：

- 静态检查 `CSKCache` 当前文件结构。
- 静态阅读 LMCache vLLM connector、adapter、utils、cache engine、GPU connector 和
  vLLM connector factory/base 接口。
- 根据用户纠正，修正设计记录：`target_start/target_end` 只代表旧实验接口，不代表
  真实系统可用信息；CSKCache 迁移核心应是 segment occurrence 自动发现。
- 根据用户澄清，补充边界：是否复用、复用哪个 segment、对应哪个 cache entry 是
  CSKCache 的职责；vLLM 只需要接收可执行调度/slot 写入计划。
- 静态核对 LMCache 代码后补充：LMCache 确实由 connector 告诉 vLLM 要额外 load 多少
  external KV tokens，但该接口主语义是当前 computed 位置之后的连续 prefix/chunk 命中。
- 根据用户追问，明确迁移方向：优先 connector 化 CSKCache，但保留必要的小型 vLLM
  scheduler hook；不建议继续沿用旧 `context_segment_cache` 作为长期系统形态。
- 使用 `opencode` 环境执行源码 `compile()`，12个CSKCache Python文件语法检查通过。
- 使用合成 registry entries 验证 matcher：多 occurrence、ready occurrence 和 no-match
  情况通过。
- 使用小 tensor 验证 `slot_ops.scatter_span/gather_span` 往返一致。
- 使用
  `PYTHONPATH=/home/wsh/openhands_code_research/CSKCache:/home/wsh/vllm`
  成功 import `CSKCacheConnectorV1`；过程中 torch/vLLM 报无 NVML/CUDA runtime
  警告，但模块加载成功。
- 对 `CSKCache/cskcache/v1/rope.py` 执行源码 `compile()`，语法检查通过。
- 使用 fake rotary embedding 验证 `rerotate_k_for_target_positions()`：
  same-position 返回原 tensor；source->target 正 delta 后再 target->source 负 delta
  可以数值恢复；非 RoPE pass-through 维度保持不变。验证过程中 vLLM import 报无
  NVML/CUDA runtime 警告，但函数检查通过。
- 未生成遗留 `__pycache__`。
- 使用 `PYTHONDONTWRITEBYTECODE=1` 对上述 8 个 vLLM 主链路文件执行源码 `compile()`，
  语法检查通过。
- 使用 `rg` 检查上述主链路文件中 `context_segment_cache`、`ContextSegment`、
  `context_segment_kv`、`VLLM_CONTEXT_SEGMENT`、`VLLM_SEGMENTIA`、
  `post_prefill`、`attention_probe`、`function_vector_probe`，没有发现未注释的旧调用；
  剩余命中均在以 `# context_segment_cache:` 标记的注释块中。
- 本轮已修改 `CSKCache` 代码和 `/home/wsh/vllm` 主链路注释；未启动 vLLM，未运行实验。
