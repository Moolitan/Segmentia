# CSKCache 架构导读

## 0. 一分钟通俗版（不假设你了解 KV cache 内部）

大模型回答问题前，要先把 prompt「读」一遍，这一步（prefill）会为每个 token 算出一份
中间状态叫 **KV**，存在显存里，后面生成时反复用。Agent 场景里，同一个「技能」文本
（一段工具说明、一段 system 规则）会在成千上万次请求里重复出现——每次都重算它的 KV 很浪费。

**CSKCache 的作用**：把每个技能的 KV 提前算好存起来，之后遇到同一个技能就直接把 KV
搬进显存，跳过重算。难点是：同一个技能放在不同上下文里，KV 会有偏差，直接复用可能答错。
CSKCache 的**招牌机制**是 **probe 门控**：先只重算技能开头几个 token，和存下来的 KV 比一比，
够像就直接复用剩下的，不够像就多补算一段再复用——用极小的重算换取正确性。

这份文档讲 CSKCache 的代码怎么组织：一个**不依赖 vLLM、可单独 import 的缓存中间件**，
外加一层薄薄的 vLLM 适配。

## 1. 总览：五层 + 一层适配

```
cskcache/
  v1/
    core/          ② 引擎（大脑，vLLM 无关）
    token/         ② token exact-match 离线诊断工具
    kv_transfer/   ③ KV 在「离线条目 ↔ 显存分页缓存」之间搬运
    storage/       ④ CPU + 磁盘两级存储 + LRU 淘汰
    compute/       ⑤ probe 门控残差 + 决策（招牌机制）
    metadata.py    共享数据结构（entry / reuse signal / load plan / 搬运载体）
  integration/
    vllm/          ① 唯一 import vLLM 的地方：把 vLLM 调用翻译给引擎
```

设计原则：**引擎只吃普通 Python 数据**（`list[int]` token、`int` 帧位置、
`dict` reuse signal）和 torch 张量，**不 import vLLM**。因此整个中间件可以脱离
vLLM 构造和单元测试；vLLM 只是它的一个前端。

只想快速理解主流程，按此顺序读：

```
cskcache/v1/core/cache_engine.py          # 所有决策
cskcache/v1/token/token_database.py       # 离线 exact-match 诊断工具
cskcache/v1/storage/storage_manager.py    # 多级存储路由 + 淘汰
cskcache/v1/kv_transfer/gpu_connector.py  # scatter/gather + RoPE 校正
cskcache/v1/compute/gate.py               # probe 门控决策
cskcache/integration/vllm/v1_adapter.py   # vLLM 翻译层
```

## 2. 各层职责

### ① 集成适配层 `integration/vllm/`
- `v1_connector.py`：`CSKCacheConnectorV1`，vLLM `KVConnectorBase_V1` 入口薄壳，逐个钩子转发。
- `v1_adapter.py`：`CSKCacheConnectorV1Impl`，从 vLLM `Request` / `KVCacheBlocks` /
  `ForwardContext` 抽出普通数据 → 调引擎 → 把引擎产出的普通载体包成 vLLM 的
  `KVConnectorMetadata` / `KVConnectorWorkerMetadata` 信封。**唯一 import vLLM 的层。**
- `utils.py`：`load_vllm_config` → `CSKCacheConfig.from_vllm`。

### ② 引擎 `core/`（大脑）
- `config.py`：`CSKCacheConfig`，`from_dict / from_env / from_vllm` 三种构造，vLLM 无关。
- `probe_state.py`：`CSKProbePhase`（`NEED_PROBE→WAIT_PROBE→NEED_ANCHOR/NEED_LOAD→DONE`）
  与 `CSKProbeState`。
- `cache_engine.py`：`CSKCacheEngine`，承载全部决策：
  - 调度侧：`get_num_new_matched_tokens / cap_prefill_before_reuse /
    get_boundary_reuse_load_tokens / update_state_after_alloc / build_meta /
    on_worker_decisions / on_finished`
  - worker 侧：`register_kv_caches / load / capture_probes / decide_probes`
    （device 活委托 kv_transfer；gate 累积留在引擎）

### ② 令牌层 `token/`
- `token_database.py`：`SegmentCatalog`（用 KMP 做精确 token 子序列匹配）+
  `find_best_reuse`。这是离线诊断工具，不是 production 复用路径。线上请求没有
  `kv_transfer_params["cskcache"]` reuse signal 时，引擎不扫描 prompt，直接让 vLLM
  正常 prefill。

### ③ KV 搬运层 `kv_transfer/`
- `gpu_connector.py`：`KVConnectorInterface` + `VLLMPagedGPUConnector`。包装
  `slot_ops`（scatter/gather）、`compute.reuse`（切片 + 目标位置 K 校正）、`rope`
  （相对旋转）。把「显存分页布局」的细节从引擎里隔离出来。

### ④ 存储层 `storage/`
- `abstract_backend.py`：`StorageBackendInterface`（contains/get/put/remove/size）。
- `local_cpu_backend.py`：内存热层（泛化原 registry）。
- `local_disk_backend.py`：`.pt` 冷层，惰性加载，`.json` sidecar 记账。
- `cache_policy/lru.py`：LRU 淘汰（可扩展为 LFU/MRU/FIFO）。
- `storage_manager.py`：`StorageManager`，CPU 命中优先、未命中查磁盘并提升、put 超预算
  按 LRU 向磁盘溢出。默认 `cpu_max_bytes=None` 即全内存，行为与旧 registry 完全一致。
- `v1/registry.py` 是委托 `StorageManager` 的兼容 shim（`CSKCacheRegistry` /
  `get_global_registry`）。

### ⑤ 计算层 `compute/`（招牌机制）
- `gate.py`：`CSKProbeAccumulator`（逐层累积 `1 - cos` 残差）→ `CSKProbeDecision`
  （`gate_value ≤ τ` 则 `passed`）。
- `reuse.py`：`prepare_reuse_slice`（切片 + 需要时按目标位置校正 K）。

## 3. 一条完整请求路径（vLLM v1 + probe 门控）

```
调度进程：
  cap_prefill_before_reuse()      # 把 prefill chunk 截到技能段起点 / probe_end / anchor_end
  get_boundary_reuse_load_tokens # 到边界时声明「就地加载，不 forward」，生成 CSKLoadPlan
  update_state_after_alloc  # 记录 vLLM 分配的物理 block_ids
  build_meta()              # 打包 CSKReqMeta（load） / CSKProbeMeta（probe）发往 worker

worker 进程（围绕 forward）：
  register_kv_caches()      # 记录各层分页 KV 张量
  load()                    # forward 前：reuse_slice → scatter 注入 clean KV
  （模型 forward；probe 段按普通 prefill 真算）
  capture_probes()          # 每层：gather 真 KV，与 clean KV 逐层比残差
  decide_probes()           # forward 后：出 CSKProbeDecision 回传调度进程

调度进程：
  on_worker_decisions()     # passed→NEED_LOAD 直接加载尾部；failed→NEED_ANCHOR 先补算
```

非 probe 的**直接复用**路径更短：技能段起点正好在帧位置时，`get_num_new_matched_tokens`
直接返回可复用长度，worker 只走 `load()`（scatter），不需要 probe / worker meta。

## 4. 按目标改代码

- 改「命中/定位」：`core/cache_engine.py` 的 reuse signal 解析与调度边界。
- 改「probe 门控算法」：`compute/gate.py`（残差/阈值/指标）。
- 改「KV 搬运性能」：`kv_transfer/gpu_connector.py` + `slot_ops.py`。
- 改「存储层级/淘汰」：`storage/storage_manager.py` + `storage/cache_policy/`。
- 改「vLLM 集成行为」：`integration/vllm/v1_adapter.py`。

## 5. 作为库使用（无需 vLLM）

```python
from cskcache import CSKCacheEngine, CSKCacheConfig, StorageManager, CSKCacheEntry

storage = StorageManager.with_disk("/data/skills_kv", cpu_max_bytes=8 << 30)  # 8 GiB 热层
storage.put(entry)  # entry: CSKCacheEntry(cache_id, token_ids, kv_by_layer, ...)
engine = CSKCacheEngine(CSKCacheConfig(probe_enabled=True), storage, block_size=16)
n, _ = engine.get_num_new_matched_tokens("req-1", token_ids, num_computed_tokens=0)
```
