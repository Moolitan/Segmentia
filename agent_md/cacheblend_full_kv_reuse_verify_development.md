# CacheBlend Two-Pair Full-KV-Reuse Verify Development

## 总开发目标

实现一个最小、可复查的两 pair 对照：比较组合 prompt 全量重算与两个 chunk
分别预计算后零 token 重算的完整 KV reuse，并通过 `Cristiano` 与
`Cristiano Ronaldo` 两种表述，观察 Qwen3-14B 的实体关联和 `13 > 8` 数值
比较是否对完整复用表现出不同敏感性。

## 开发阶段总览

| 阶段 | 名称 | 目标 | 当前进度 | 剩余 |
|---|---|---|---|---|
| 1 | 实验语义冻结 | 固定三个 chunk、两个 case、两种 mode 和 full reuse 定义。 | 已完成。 | 无。 |
| 2 | Token/KV 数据流 | 独立采集三个 chunk KV并严格核对两个 case 的目标 token span。 | 已完成代码。 | 真实服务验证。 |
| 3 | 隔离执行 | collection 后按 `(case, mode)` 重启并完成四个 decode arm。 | 已完成 wrapper。 | 用户运行。 |
| 4 | 结果沉淀 | 保存四份原始响应并生成轻量 case/mode 对比。 | 目录和生成逻辑已完成。 | 运行后人工判读并更新 summary。 |
| 5 | 静态验证 | 编译、CLI、shell语法和路径检查。 | 已完成。 | 无。 |
| 6 | 单侧复用定位 | chunk 1重算、chunk 3复用，检查显式全名case的实体—数字绑定。 | 独立终端probe已完成代码。 | 用户运行并与双复用结果比较。 |
| 7 | 非零温度观察 | 四个arm使用temperature 0.6及Qwen3采样截断参数运行一次。 | 参数接口和独立run路径已完成代码。 | 用户运行；单seed不作稳定性结论。 |

## Claim 与 go/no-go

主张不是“CacheBlend 已有效”，而是更窄的机制验证：

```text
两个独立 chunk 在最终请求中零 token 重算时，`Cristiano` 短名 case 是否比
显式 `Cristiano Ronaldo` control 更容易丢失实体关联或跨 chunk 数值比较？
```

| 判据 | Go | No-Go |
|---|---|---|
| 工程正确性 | 三个 `.pt` 均生成；两个 reuse 请求各显示两个 target 注入；没有 token mismatch | 任一 KV 未保存、span 不唯一、token 不一致或注入未发生 |
| Recompute sanity | 两个 case 均明确回答 Messi，并使用 13 与 8 | 对应 recompute 不能答对时，该 case 无法诊断 reuse |
| Case内 full reuse | 对应 case 仍回答 Messi并使用正确实体与数字 | 选择错误、忽略一个 chunk，或 reasoning 暴露实体关联/比较失败 |
| Case间诊断 | `explicit_full_name` 成功而 `alias_only` 失败，支持实体表述是候选脆弱点 | 两者同时成功或同时失败，不能把差异定位到 alias |

单次成功只支持机制可行性，不支持可靠性、准确率或 CacheBlend 总体效果主张。

## 输入、核心结构与数据流

固定文本：

```text
Lionel Messi scored 13 goals at FIFA World Cups.
Cristiano scored 8 goals at FIFA World Cups.
Cristiano Ronaldo scored 8 goals at FIFA World Cups.
Who scored more goals at FIFA World Cups, Messi or Ronaldo?
```

Python 中 `CHUNKS` 是三个包含 `name/cache_id/text` 的记录，`CASES` 显式映射：

```text
alias_only        -> [chunk_1, chunk_2]
explicit_full_name -> [chunk_1, chunk_3]
```

每个组合 prompt 按 case 中的 chunk 顺序再接 query，不排序、不聚合、不去重。

采集流程：

```text
先对两个 case 的全部目标 span做预检
for chunk in [chunk_1, chunk_2, chunk_3]:
  将缓存 span 定义为 chunk 事实句及其后一个换行
  用 completion tokenizer 得到该完整 span 的 token IDs
  在 standalone chat prompt 中查找唯一完全匹配 span
  在 combined chat prompt 中查找唯一完全匹配 span
  核对两个 span 的 token IDs 完全一致
  发送只包含该 chunk 的独立 source request
  注册并保存该 span 的全层 K/V
```

采集时 prefix cache 关闭，所以后续 source request 不会继承前一个 chunk 的
状态。chunk 1 只采集一次并供两个 case 共用；KV 以 `cache_id` 查找，不按文本
模糊匹配。

decode 流程：

```text
recompute:
  当前 case 的 combined prompt 不携带 context_segment_cache targets

full_kv_reuse:
  target chunk_1 -> cacheblend-verify-chunk-1
  alias_only的第二个target -> cacheblend-verify-chunk-2
  explicit_full_name的第二个target -> cacheblend-verify-chunk-3
  两个 target 都使用 mode=rope
  vLLM 在各 target_start 停住 prefill并注入完整 span KV
  其余 prompt token（chat template、chunk 间换行和 query）正常 prefill
```

`rope` 只重新定位 key；value 原样复用。两个 chunk token 都不执行当前请求的
Transformer forward，因此属于零 token 重算。

单侧复用定位 probe：

```text
explicit_full_name:
  chunk 1不配置target -> 从prompt起点正常重算
  prefill推进到chunk 3的target_start
  chunk 3 -> cacheblend-verify-chunk-3, mode=rope
  query正常prefill
```

该 probe 只向终端打印，不写 raw JSON或覆盖现有四臂结果。默认生成上限提高到
1024 tokens，避免此前512-token thinking截断直接造成空answer。

缓存包含分隔边界的原因是 Qwen3 tokenizer 会把句末句号与后续换行合并为一个
token。若只缓存裸事实句，standalone 末尾 token 与组合 prompt 中对应 token
不同。把紧随事实句的一个换行纳入 span 后，standalone 和组合 prompt 能保持
完全相同的 token IDs；chunk 2 和 chunk 3 的 span 都在换行处结束，不包含
query token。

## 服务、缓存与失败恢复

服务重启边界：

```text
(cacheblend_verify, collect_independent_chunks)
(cacheblend_verify, alias_only, recompute)
(cacheblend_verify, alias_only, full_kv_reuse)
(cacheblend_verify, explicit_full_name, recompute)
(cacheblend_verify, explicit_full_name, full_kv_reuse)
```

这不是 Segmentia 多轮 trace replay：

- 没有 `invocation_index`。
- 没有 task 内历史轮次。
- 不通过 prefix cache 继承历史注入；prefix cache 在全部 stage 中关闭。
- 每个 full reuse 请求只显式注入当前 case 的两个 chunk span一次。

脚本不自动重试或跳过。输出采用防覆盖语义：默认 `pilot1` 中任一 collection、
decode 或 summary 产物已存在时终止。部分失败后应检查日志，并指定新的
`RUN_DIR`；若还要生成新的仓库内 summary，同时指定新的 `SUMMARY_OUTPUT`。

## 写入路径

代码：

```text
scripts/06_context_free_segment_cache/cacheblend_verify/
  run_cacheblend_verify.py
  run_cacheblend_verify.sh
  run_chunk1_recompute_chunk3_reuse.py
```

大体量与逐 run 产物：

```text
/mnt/Large_Language_Model_Lab_1/wsh/CacheBlend/output/
  two_chunk_full_reuse_verify/pilot1/
    kv/
    raw/alias_only/{recompute,full_kv_reuse}.json
    raw/explicit_full_name/{recompute,full_kv_reuse}.json
```

总结性轻量产物：

```text
results/problem_exploration/cacheblend_full_kv_reuse_verify/
  summary.md
  source_manifest.csv
  data/comparison.json  # 真实运行后生成
```

## 运行方式

由用户在 `opencode` 环境执行：

```bash
cd /home/wsh/openhands_code_research
conda activate opencode
bash scripts/06_context_free_segment_cache/cacheblend_verify/run_cacheblend_verify.sh
```

agent 本轮不启动 vLLM 或 decode。

非零温度观察默认配置：

```text
temperature = 0.6
top_p = 0.95
top_k = 20
min_p = 0.0
seed = 1111
```

这些参数只用于四个decode arm，KV collection不依赖生成采样。为避免覆盖已有
`pilot1`，默认使用：

```text
/mnt/Large_Language_Model_Lab_1/wsh/CacheBlend/output/
  two_chunk_full_reuse_verify/pilot_temp06/

results/problem_exploration/cacheblend_full_kv_reuse_verify/
  data/comparison_temp06.json
```

每个arm只有一个seed，该run只用于观察现象，不能判断随机稳定性。

## 当前问题与风险

- 本地 Qwen3 chat template已覆盖两个完整case：`alias_only`中chunk 1为
  `[8,22) -> [8,22)`、chunk 2为`[8,20) -> [22,34)`；
  `explicit_full_name`中chunk 1为`[8,22) -> [8,22)`、chunk 3为
  `[8,21) -> [22,35)`。服务端仍会在发送任何KV collection前重复检查。
- source request 与组合请求的上下文不同，这正是实验要测试的 full reuse
  context gap，不应通过额外上下文化消除。
- `Cristiano -> Ronaldo` 可能由模型参数知识完成；显式全名 control能缩小解释
  范围，但仍不是严格的知识来源归因实验。
- 真实运行后必须检查 vLLM 日志中的四个 `ContextSegmentKV: applied` 记录
  （两个 reuse arm各两个）；仅看到正确答案不能证明注入实际发生。

## 本轮修改与验证

新增：

- `run_cacheblend_verify.py`
- `run_cacheblend_verify.sh`
- 本 development 文档
- `results/problem_exploration/cacheblend_full_kv_reuse_verify/` 结果骨架

静态验证：

- `python -m py_compile .../run_cacheblend_verify.py`：通过。
- `python .../run_cacheblend_verify.py --help`：通过。
- `python .../run_cacheblend_verify.py collect --help`：通过。
- `bash -n .../run_cacheblend_verify.sh`：通过。
- 本地 Qwen3 chat template验证发现裸事实句因 `句号 + 换行` merge无法匹配
  组合 prompt；修复后 chunk 1 的source/target均为`[8,22)`，chunk 2 的
  source为`[8,20)`、target为`[22,34)`，四个位置均各自唯一匹配。
- 新增 chunk 3 和 `alias_only/explicit_full_name` 两个 case；Python compile、
  三个CLI和shell语法均通过。两个case的四个source/target组合已用本地Qwen3
  chat template验证token identity和唯一匹配，精确span见“当前问题与风险”。
- 单侧复用probe通过`py_compile`、`--help`、模块导入、case映射和chunk 3缓存
  路径检查；未连接或启动vLLM。
- 主实验decode新增`temperature/top_p/top_k/min_p` CLI并把采样参数写入raw
  JSON及comparison；shell默认传递`0.6/0.95/20/0.0`并使用独立
  `pilot_temp06`路径。

尚未验证：

- vLLM `/tokenize` 返回的 chat-template span与本地 tokenizer是否一致。
- 三个 KV 文件的真实保存与加载。
- 两个 case 共四个 arm 的模型回答。
- vLLM 日志中的四次 target 注入。
- chunk 1重算、chunk 3复用单侧probe的真实输出。
