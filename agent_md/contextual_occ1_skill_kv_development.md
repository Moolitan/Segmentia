# Contextual Occurrence-1 Skill KV Development

## 总开发目标

为 Segmentia 增加一套上下文化 skill KV 对照源。每个 `(task, skill)` 使用该
skill 在对应真实 trace 中第一次出现时的完整前缀计算 K/V，保留 system
prompt、tools 和此前对话历史的影响，并与最小脚手架生成的 clean skill KV
严格区分。

## 开发阶段总览

| 阶段 | 名称 | 目标 | 当前进度 | 剩余 |
|---|---|---|---|---|
| 1 | 来源定义 | 明确按 `(task, skill)` 采集 occurrence 1，不跨 task 去重。 | 已完成。 | 无。 |
| 2 | Prefill 实现 | 从真实 invocation 构造 source span并保存 task-specific KV。 | 已完成。 | 无。 |
| 3 | 隔离调度 | 每个 task 重启 vLLM，task 内按首次 invocation顺序采集。 | 已完成。 | 无。 |
| 4 | 静态验证 | 验证脚本可导入、CLI、shell语法、顺序和cache ID唯一。 | 已完成。 | 无。 |
| 5 | 真实采集 | 启动带 `VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR` 的 vLLM并生成 `.pt`。 | 已完成12个task-specific source。 | 无。 |
| 6 | 跨 task occurrence-1 对照 | 用三个同名skill case比较target重算与source-task contextual KV复用。 | 双臂采集和首次thinking分析已完成。 | 增加clean context-free第三臂并做重复性验证。 |

## 当前实现

新增：

```text
scripts/06_context_free_segment_cache/prefill_contextual_skill_kv.py
scripts/06_context_free_segment_cache/run_prefill_contextual_skill_kv.sh
```

数据流：

```text
module/config.py::SKILL_TOKEN_LOCATIONS
  -> 每个 task 的每个 skill
  -> invocation_indices[0] 定位首次真实 invocation
  -> token_spans[0] 定位包含两个 context_segment 标签的 skill span
  -> 使用真实 system prompt、tools 和完整 trace messages执行 prefill
  -> 保存 task-specific occurrence-1 KV
```

默认大体量输出目录：

```text
$SEGMENTIA_OUTPUT_DIR/contextual_occ1_skill_kv/
```

该路径由 `module/config.py::DEFAULT_CONTEXTUAL_KV_DIR` 统一定义。Python
prefill和shell wrapper均读取该配置，不各自维护硬编码默认值。

cache ID：

```text
contextual-occ1-skill-{task}-{skill}
```

相同 skill 在不同 task 中不会覆盖，因为真实前缀上下文不同。

服务生命周期：

```text
for task:
  restart vLLM
  clear prefix cache and in-process KV registry
  collect this task's skills by invocation_indices[0]
```

同一 task 内不重启，使后续首次出现的 skill 可以继承该 task 更早 invocation
自然形成的真实 prefix cache；不同 task 之间不共享 prefix cache。每个 task
写入独立的 `manifests/{task}.json`。

## 校验与失败语义

脚本在发送任何 KV collection 请求前完成全量预检：

- `invocation_indices[0]` 必须指向存在的 invocation。
- 首次 invocation 中必须恰好出现一次目标 skill。
- 从真实消息重新计算出的 token span 必须等于 `token_spans[0]`。
- occurrence 1 的 source token IDs 必须与同 task 后续 occurrences 完全一致。
- 目标 `.pt` 或 manifest 已存在时立即失败，要求使用空目录并重启 vLLM，
  避免 registry 已知 cache ID 导致误以为完成重新采集。
- 每次 collection 后目标 `.pt` 必须实际存在，否则立即失败。

脚本不做自动重试，也不会跳过失败 case。任何 source 定义或保存失败都会终止，
防止生成不可比较的部分数据集。

## 当前限制与风险

- 尚未运行真实 vLLM prefill，因此 `.pt` 数值和 manifest 尚未验证。
- `--kv-dir` 是客户端预期输出路径；服务端实际写入位置由
  `VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR` 决定，两者必须相同。
- 当前未修改 `raw_decode_token_sequences`。现有 `rope` arm仍加载
  `offline_skill_kv` 中的 clean KV，不会自动使用本脚本产物。
- 后续 contextual replay 必须使用 task-specific cache ID，不能继续使用
  `context-free-skill-{skill}`。

## 三个跨 task occurrence-1 对照

新增：

```text
scripts/06_context_free_segment_cache/cross_occurrence_controller/
  raw_decode_token_sequences/
  run_cross_task_contextual_occ1_reuse.py
  run_cross_task_contextual_occ1_reuse.sh
```

固定case：

```text
doc_coauthoring_design_doc/doc-coauthoring
  -> mcp_server_and_spec/doc-coauthoring occurrence 1

internal_comms_incident_update/internal-comms
  -> slack_launch_pack/internal-comms occurrence 1

web_artifact_with_theme/web-artifacts-builder
  -> launch_poster_page_pack/web-artifacts-builder occurrence 1
```

每个case运行 `recompute` 和 `cross_contextual_rope` 两个mode。shell服务边界是
`(mode, target_task)`；跨task arm加载 contextual KV目录，只对当前target的
occurrence-1 span显式注入一次。三个方向均满足`target_start >= source_start`，
且source/target span长度相等。

默认使用`temperature=0`排除小样本采样方差。每个mode保存三份未经message
parser改写的raw token TXT和一份JSONL manifest。manifest记录source/target
task、cache ID、source/target span、采样参数、token数、SHA256、finish reason
和文件路径。当前只完成代码，未生成真实decode结果。

## 本轮修改与验证

修改：

- 新增 `prefill_contextual_skill_kv.py`。
- 新增 `run_prefill_contextual_skill_kv.sh`，按 task 重启 vLLM。
- Python source列表显式按 task 内 `invocation_indices[0]` 排序。
- `module/config.py` 新增 `DEFAULT_CONTEXTUAL_KV_DIR`；Python和shell共享该
  默认路径。
- 新增三个case的跨task occurrence-1 raw decode采集器和双mode shell。
- 新增本 development 文档。
- 未修改 `prefill_clean_skill_kv.py`和现有raw decode replay mode。

验证：

- `python -m py_compile
  scripts/06_context_free_segment_cache/prefill_contextual_skill_kv.py`：通过。
- `python scripts/06_context_free_segment_cache/prefill_contextual_skill_kv.py
  --help`：通过。
- 静态枚举得到12个 `(task, skill)` source和12个唯一 task-specific cache ID。
- `bash -n
  scripts/06_context_free_segment_cache/run_prefill_contextual_skill_kv.sh`：通过。
- 静态顺序检查确认多skill任务分别按 `2,11`、`2,11`、`2,6,11` 和
  `2,8,11` 的 invocation顺序采集。
- 跨task采集Python通过`py_compile`，CLI `--help`可加载，shell通过
  `bash -n`。
- 三个case静态检查通过：source/target长度分别为3313、338、719，target
  position均不早于source position，三份source `.pt`均存在。
- 合成response验证raw token提取、JSONL追加和manifest重新加载通过。
- 代码归入`cross_occurrence_controller/raw_decode_token_sequences/`后，修正
  `ROOT`和`MODULE_DIR`的目录层级；按实际入口重新验证Python导入、`--help`
  和shell语法通过。
- 首次shell实跑成功启动recompute服务，但Python入口被重复拼接目录而失败；
  已改为直接使用`$SCRIPT_DIR/run_cross_task_contextual_occ1_reuse.py`。
- 后续实跑已完整生成`doc-coauthoring -> mcp_server_and_spec` recompute结果：
  1049 tokens、`finish_reason=tool_calls`，TXT与manifest SHA256一致。再次运行因
  防覆盖检查停止，因此补充默认`RESUME=1`；已有case仅在文件、SHA256和采样
  参数均一致时跳过，否则失败。当前每个target task只有一个case，跳过不会
  缺失task内prefix状态。
- 本地直接执行`--resume`检查已对上述完整case输出`[skip-valid]`，未连接
  vLLM或重复decode。
- 用户完成三case双臂实验，6/6 raw TXT和2份manifest完整，全部
  `temperature=0`且`finish_reason=tool_calls`。
- 完整阅读六段`<think>`：3/3保持target主题和顶层tool name，但thinking和
  tool arguments均0/3精确匹配；未观察到source task主题直接复制。
- `internal-comms`发生明确退化：cross arm提前猜测brief和3P模板，最终Read
  path丢失task子目录；`web-artifacts-builder`保持页面目标但实现、路径和页面
  内容变化；`doc-coauthoring`保持最稳定且输出结构更贴近当前短spec要求。
- 新增结果summary、think comparison表、metric metadata、figure omission说明
  和source manifest；同步更新problem-exploration阶段导航。由于只有三个异质
  case，不生成可能暗示总体比例的聚合图。
