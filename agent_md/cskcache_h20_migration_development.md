# CSKCache H20 实现兼容迁移 Development

## 总开发目标

将已在 H20 环境使用的 `cskcache_h20/` 有效能力兼容、扩展并迁移到正式
Python package `CSKCache/cskcache/`。迁移以正式 package 的现有生命周期和测试为
基线，不整目录覆盖，不复制运行残留；H20 新能力通过显式配置启用，现有 Qwen3、
ticket TTL、多请求独立 Host buffer 和 profiling 入口保持兼容。

## 开发阶段总览

| 阶段 | 名称 | 目标 | 当前进度 | 剩余 |
|---|---|---|---|---|
| 1 | H20 差异审计 | 区分有效源码、实验语义变化和拷贝残留 | 已完成 | 无 |
| 2 | 模型结构扩展 | 支持 native decoder、Mistral 和 Pixtral text decoder | 已完成并通过 CPU 测试 | 真实 Pixtral/Mistral GPU forward |
| 3 | 多 Worker 与 TTL 兼容 | 支持 metadata 路径模板，同时保留可配置 ticket 超时 | 已完成并通过 CPU 测试 | TP 环境逐 rank Catalog 验证 |
| 4 | Host 末对象常驻 | 可选保留最近一个完整 Skill 的 Pinned Host buffers | 已完成并通过 CPU 测试 | H20 上验证真实 SSD 读取次数与 TTFT |
| 5 | Profiling 输出收敛 | trace file 与 stdout 可独立控制，避免默认重复日志 | 已完成并通过子进程测试 | 真实服务日志核对 |
| 6 | Package 回归 | 执行正式 CSKCache 全量 CPU 测试与静态检查 | 已完成，160 tests passed | 无 |
| 7 | 硬件验收 | 验证 H20 workload、模型、TP 和 Pinned/SSD 数据路径 | 未执行，按仓库约束留给用户 | 用户运行真实服务和实验 |

## 差异审计结论

`cskcache_h20/` 与迁移前的 `CSKCache/cskcache/` 大部分源码一致。有效差异只有：

1. LMCache layerwise model adapter，增加 Mistral decoder 和 Pixtral wrapper；
2. `cskcache_metadata_path` 的 `{worker_id}` / `{world_size}` 展开；
3. 可选保留最近一个 Host-resident cache object；
4. profiling trace 存在时默认不重复写 stdout；
5. H20 版本删除了 ticket TTL。

`.DS_Store`、`runtime.py.orig`、`__pycache__/`、`.pyc` 和名称以 ` 2` 结尾的空目录
属于拷贝或运行残留，没有迁移。`cskcache_h20/` 本身未修改或删除。

H20 对 TTL 的处理没有直接照搬。正式 package 原先用 TTL 回收异常 ticket，并已有
过期回归测试；完全删除会造成现有服务语义倒退。当前实现将其扩展为兼容配置：

- 未配置时仍为 `60.0` 秒；
- `csk_prefetch_handle_ttl_seconds: null` 时不创建 deadline，复现 H20 无 TTL 行为；
- 正数继续使用现有 `MetadataManager.expire()` 和 storage cancel 路径；
- 零或负数仍 fail closed。

## 模型适配数据流

新增 `cskcache/integrations/lmcache/model_adapters/`，把模型结构判断移出
`worker.py`：

```text
VLLMModelTracker.get_model(engine_name)
  -> ModelAdapterRegistry.bind(outer_model, blender)
  -> 按 outer model 类名选择 wrapper resolver
     PixtralForConditionalGeneration -> outer_model.language_model
  -> 按 decoder 类名选择 builder
     Llama/Qwen2/Qwen3 -> LMCache infer_model_from_vllm()
     Mistral           -> LMCLlamaModel
  -> 验证 layerwise_model.vllm_model 是同一个 decoder 实例
  -> 验证 decoder.model.layers 非空
  -> 返回 LayerwiseModelBinding
  -> worker 使用 binding.layerwise_model / binding.num_layers
```

registry key 是 vLLM Python 类名，不按模型路径做模糊匹配。未知 wrapper/decoder、
Pixtral 缺少 `language_model`、Pixtral 解析出非 Mistral decoder、builder 绑定到另一模型
实例或 decoder 没有 layer，都会直接失败。Pixtral 只复用语言 decoder 的 KV，不触碰
vision tower。

## 多 Worker metadata 路径

`LMCacheRuntimeBridge` 读取 `cskcache_metadata_path` 后，只展开实际出现的占位符：

```text
/path/catalog-{worker_id}-of-{world_size}.json
  -> engine.metadata.worker_id / world_size
  -> /path/catalog-2-of-4.json
```

路径没有占位符时不要求 engine metadata 提供这两个属性，保持单 Worker 旧入口兼容。
路径使用了占位符但对应属性不存在时直接报错，不读取带未解析花括号的错误路径。每个
worker 仍创建独立 `MetadataManager` 和 `StorageManager`；本轮没有增加跨 worker Catalog
聚合、同步或 Host buffer 共享。

## Host 常驻对象状态边界

新配置 `csk_retain_last_host_object` 默认 `false`。关闭时仍是原来的
`ticket -> 独立 SSD load -> 独立 Pinned buffers -> release`，支持多个 inflight ticket。

开启后，状态属于单个 worker 的 `StorageManager`：

```text
首次对象 A:
  ticket A1 -> SSD/local-disk load -> READY -> release
  -> buffers 转为 resident(cache_object_id=A)

下一次对象 A:
  ticket A2 -> resident hit -> 直接 READY
  -> 不创建 Future，不读取 SSD，不重新 acquire buffers

切换到对象 B:
  释放 resident A -> 正常加载 B -> release 后 resident B

cancel resident-hit ticket:
  ticket 结束，resident buffers 保留

StorageManager.close():
  停止新 load -> 处理 live ticket -> 每组物理 buffers 恰好释放一次
  -> 清空 resident object
```

常驻命中 key 是完整的 `cache_object_id`，没有按 Skill name 聚合或去重。该模式明确要求
ticket 串行；存在任一 live ticket 时再提交会报错。这样避免一个 resident tuple 同时被
两个请求使用或在 H2D 尚未结束时被替换。它只影响 external Skill KV 的 Pinned Host
生命周期，不清理或复用 vLLM prefix cache，也不跨服务重启保留。

## Profiling 输出

`CSKCACHE_PROFILE=1` 仍是总开关。输出规则为：

| `CSKCACHE_PROFILE_TRACE_PATH` | `CSKCACHE_PROFILE_STDOUT` | 行为 |
|---|---|---|
| 未设置 | 未设置 | 保持旧行为，写 stdout |
| 已设置 | 未设置 | 只写 JSONL trace |
| 已设置 | `1` | 同时写 JSONL 和 stdout |
| 任意 | `0` | 不写 stdout |

profile event schema、event name、request ID、时间戳和原子 append 逻辑没有改变。

## 本轮修改文件

- `CSKCache/cskcache/integrations/lmcache/model_adapters/`：新增 wrapper/decoder registry；
- `CSKCache/cskcache/integrations/lmcache/worker.py`：消费 validated model binding；
- `CSKCache/cskcache/integrations/lmcache/base.py`：增加 optional TTL 与 resident 配置；
- `CSKCache/cskcache/integrations/lmcache/runtime.py`：解析 worker 路径、TTL 和 resident 配置；
- `CSKCache/cskcache/runtime/request_manager.py`：`None` TTL 不创建 deadline；
- `CSKCache/cskcache/storage/manager.py`：单 resident object 生命周期；
- `CSKCache/cskcache/profile.py`：trace/stdout 路由；
- `CSKCache/tests/test_lmcache_model_adapters.py`：模型 registry 聚焦测试；
- `CSKCache/tests/test_lmcache_integration.py`：runtime 配置与路径模板测试；
- `CSKCache/tests/test_request_manager.py`：TTL disabled 生命周期测试；
- `CSKCache/tests/test_storage_manager.py`：resident hit、切换、并发和释放测试；
- `CSKCache/tests/test_profile.py`：独立子进程 profiling 输出测试；
- `agent_md/cskcache_h20_migration_development.md`：本开发记录。

## 验证结果

已完成的轻量验证：

- model adapter 与 worker 聚焦测试：`9 passed`；
- metadata/request/runtime 聚焦测试：`44 passed`；
- storage/request/LMCache integration 聚焦测试：`46 passed`；
- profiling 子进程测试：`3 passed`；
- 正确测试导入路径下的 `CSKCache/tests` 全量回归：`160 passed`；
- 相关 Python 文件 `python -m py_compile`：通过；
- `git diff --check -- CSKCache/cskcache CSKCache/tests`：通过。

全量测试首次 collection 使用的 `PYTHONPATH` 缺少仓库根目录与
`CSKCache/example`，导致四个 paper/example 测试模块无法导入；补齐为
`.:CSKCache:CSKCache/example:LMCache:vllm` 后同一全量集合全部通过。该首次失败不是
代码回归。

本轮没有启动 vLLM，没有执行 decode，没有读取真实 SSD cache pool，也没有运行 GPU、
H20 或 TP 实验。

## 当前风险与硬件 go/no-go

以下仍是尚未验证事项，不能写成主仓代码已在当前机器复现 H20 结果：

1. Pixtral/Mistral adapter 沿用 H20 已运行实现，但当前工作区只验证 registry 与对象绑定，
   尚未验证真实 `LMCLlamaModel.compute_layer()`、sliding-window attention 和输出质量；
2. TP 使用路径模板时，每个 rank 的 Catalog、模型 fingerprint、layer 数和 KV geometry 必须
   真实一致；本轮没有生成或复制 per-rank Catalog；
3. resident 模式只适用于串行请求实验。并发服务保持该配置关闭，不能把显式拒绝并发解释
   为通用 Host cache policy；
4. `null` TTL 会让异常 ticket 一直存活到显式 cancel/release 或服务关闭，只应在 H20
   workload 确实需要无超时时启用；
5. profiling 改变的是日志路由，不改变事件内容；现有依赖 stdout 抓 profile 的脚本若同时
   设置 trace path，需要显式加 `CSKCACHE_PROFILE_STDOUT=1`。

硬件验收的 go 条件是：目标模型能绑定正确 text decoder；每个 worker 读取自己的 Catalog；
同一对象第二个串行 ticket 出现 `csk_host_resident_hit` 且没有第二次
`csk_host_read_start`；换对象和服务关闭时 Pinned buffers 无泄漏；校准 forward、H2D、
PagedKV commit 和最终输出均完成且没有 fallback。任一条件失败都应停止全量实验并保留
日志，不继续扩展 workload。
