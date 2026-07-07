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
| 4 | CSK segment 语义接入 | 自动发现请求中的可复用 segment occurrence，并把 lookup/load/save/slot mapping 接入 connector metadata。 | 已完成 A 版 token catalog matcher 和 load plan；B 版 prompt metadata 留 TODO。 | 需要 scheduler boundary hook、save path 和 RoPE parity。 |
| 5 | 验证 | 静态编译、connector import、空跑启动，再做小规模真实 replay。 | 已完成静态语法、matcher、slot ops 和 connector import 验证。 | 未启动 vLLM，未跑真实 replay。 |

## 本轮分析结论

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
  负责 segment catalog、token matching、cache entry 选择、是否复用、load/save metadata。

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
  token 序列、prompt 组装元数据，或可复查的 segment catalog 中自动发现可复用 segment
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

## 本轮修改与验证

修改：

- 新增本 development 文档，记录 CSKCache connector 迁移分析状态。
- 新增 `CSKCache/README.md`，记录初版范围、catalog格式、vLLM动态加载方式和旧
  `context_segment_cache` 处理策略。
- 新增 `CSKCache/cskcache/v1/metadata.py`：定义 `CSKCacheMode`、segment、
  occurrence、KV entry 和 load plan。
- 新增 `CSKCache/cskcache/v1/matcher.py`：A版 token catalog exact matching，
  并在注释中标记 TODO(B)：后续用 prompt-builder metadata 作为主来源。
- 新增 `CSKCache/cskcache/v1/registry.py`：加载旧 `.pt` 格式 KV entry。
- 新增 `CSKCache/cskcache/v1/slot_ops.py`：按 vLLM block ids 对 paged KV cache
  gather/scatter。
- 新增 `CSKCache/cskcache/v1/rope.py`：保留 CSK RoPE 边界；same-position 直接返回，
  different-position 明确要求后续从旧 vLLM rope实现迁移。
- 新增 `CSKCache/cskcache/integration/vllm/v1_connector.py`：实现
  `CSKCacheConnectorV1(KVConnectorBase_V1)` 动态加载入口。
- 新增 `CSKCache/cskcache/integration/vllm/v1_adapter.py`：实现 scheduler侧
  token matching / load plan / metadata，以及 worker侧从 registry scatter KV。
- 新增 package `__init__.py` 文件，保证 `PYTHONPATH=.../CSKCache` 后可以 import。

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
- 使用合成 token catalog 验证 matcher：多 occurrence、ready occurrence 和 no-match
  情况通过。
- 使用小 tensor 验证 `slot_ops.scatter_span/gather_span` 往返一致。
- 使用
  `PYTHONPATH=/home/wsh/openhands_code_research/CSKCache:/home/wsh/vllm`
  成功 import `CSKCacheConnectorV1`；过程中 torch/vLLM 报无 NVML/CUDA runtime
  警告，但模块加载成功。
- 未生成遗留 `__pycache__`。
- 本轮已修改 `CSKCache` 代码；未修改 `/home/wsh/vllm`，未启动 vLLM，未运行实验。
