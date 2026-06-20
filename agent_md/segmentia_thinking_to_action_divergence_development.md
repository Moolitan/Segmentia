# Segmentia Thinking-to-Action Divergence Development

## 开发阶段总览

| 阶段 | 名称 | 目标 | 当前进度 | 剩余 |
|---|---|---|---|---|
| 1 | 旧结论边界修正 | 将 `logprob_margin_diagnostic` 从“动作判决点诊断”收窄为 early hidden reasoning 轨迹脆弱性诊断 | 已完成 | 无 |
| 2 | 现有数据审计 | 判断 headline 与 margin 产物能否支撑 thinking/action 边界分析 | 已完成 | 无 |
| 3 | 新诊断设计确认 | 设计自由生成条件下的 thinking-to-action 诊断脚本与数据结构 | 已完成，用户确认只做 `recompute` vs `rope` | 无 |
| 4 | 新诊断实现 | 采集完整 reasoning/action/logprob，并定位动作边界 | 已完成代码实现与静态验证 | 等用户运行 |
| 5 | 运行与分析 | 用户运行长实验后，沉淀结果目录、图表、summary 和 manifest | 已完成首轮分析 | 可选：人工复核 B/C 类 case |
| 6 | 文档边界修正 | 解释 intent parser 粗粒度判定与 B 类分类的误判风险 | 已完成 | 后续人工复核 B/C 类时再更新 |
| 7 | B/C 复核与 candidate scoring 准备 | 生成人工复核入口和现有 top-k logprob 候选覆盖表 | 已完成离线脚本、表格和 summary 更新 | 人工填写复核列 |
| 8 | Candidate scoring summary | 将候选覆盖表聚合为 case-level 边界翻转结论 | 已完成；人工复核后 6 个 confirmed、2 个 partial、1 个 unclear、1 个 intent drift possible | 若 partial case 进入论文关键例子，再考虑 forced scoring |
| 9 | 最终 summary 整理 | 将当前所有数据、人工复核和 candidate scoring 结论整理为完整研究 summary | 已完成 | 后续进入 boundary 修复实验设计 |
| 10 | vLLM 注入代码复核 | 复查 `/home/wsh/vllm` 的 ContextSegmentKV 注入/复用链路是否符合当前实验假设 | 已完成静态审查；核心链路方向正确，但默认旧 GPU runner 混合 batch 存在 0-token 注入请求风险 | 修复旧 runner 混合 batch 过滤并补测试 |
| 11 | 阶段级 summary 同步 | 将 `problem_exploration/summary.md` 更新到人工复核、candidate scoring 和 vLLM 代码复核后的当前结论 | 已完成 | 无 |

## 总开发目标

研究 context-free skill KV 复用导致 action divergence 的传导机制：

```text
context-free KV reuse
  -> hidden reasoning / thinking 语义漂移或上下文缺失
  -> 动作边界处候选分数变化
  -> tool/text 或 Write/Edit/Read 分叉
```

核心问题不是简单跳过 thinking，而是在自由生成轨迹自然演化的条件下，分析 thinking 如何影响后续动作边界。

## 当前问题重述

旧的 `logprob_margin_diagnostic` 采集了每条 completion 前 256 个 token 的 top-k logprob，并在 `decision_window=128` 内统计最小 margin。复查发现：

- 该窗口从 completion 的 `token_index=0` 开始。
- 请求启用了 `enable_thinking=True`。
- 因此前 128 个 token 通常位于 `<think> ...` hidden reasoning 内。
- 旧分析主要说明 early thinking 轨迹脆弱，不能证明动作分叉点已经被定位。

## 现有数据审计

### Headline decode 产物

文件：

```text
results/problem_exploration/headline_semantic_action_gap/data/decode_outputs.jsonl
```

审计结果：

- 共 72 条记录：24 个 case × `recompute/direct/rope`。
- 每条都有 `reasoning`，可以用于 thinking 语义比较。
- 50/72 条有 `tool_calls`，41/72 条有 `content`。
- `max_tokens=4096`，完整输出覆盖动作结果。
- completion tokens 范围为 299 到 2308，中位数约 870。

结论：headline 产物可用于 thinking 语义诊断和 action label 对齐，但没有 token-level logprob，不能直接分析动作边界 margin。

### Logprob margin 产物

文件：

```text
results/problem_exploration/logprob_margin_diagnostic/data/logprob_margin_rows.jsonl
```

审计结果：

- 共 72 个 case/mode，每个 case/mode 256 个 token-level logprob。
- 70/72 条没有生成到 `</think>`。
- 只有 2/72 条在 logprob 采集范围内触达 `<tool_call>`。
- first-diff 中位 token index 为 25，34/48 个 reuse-vs-recompute 对比在 32 token 内分叉，主要是 `the`、`let`、`check`、`,` 等普通 hidden reasoning token。

结论：当前 logprob 产物不足以分析动作边界；它只能支撑 early hidden reasoning 轨迹脆弱性结论。

## 新诊断目标

下一步应新建研究内容目录：

```text
results/problem_exploration/thinking_to_action_divergence/
```

目标输出：

1. 自由生成条件下 `recompute` 与 `rope` 的完整 reasoning/action/logprob 数据。
2. 每个 case/mode 的 thinking 语义指标。
3. 每个 case/mode 的动作边界定位与 boundary margin。
4. recompute vs rope 的 thinking-to-action 分类表。
5. 至少一张总结图，展示 thinking similarity、action divergence 和 boundary margin 的关系。

## 拟定分析分层

### 第一层：thinking 语义诊断

比较 recompute/rope 的 `<think> ... </think>` 内容：

- overall semantic similarity。
- task-state grounding：是否正确提到当前用户要求、目标文件、已有 artifact、上一轮动作结果。
- next-action intent：从 thinking 中抽取下一步打算做什么。
- context omission / wrong carry-over：是否漏掉当前上下文，或把复用 skill 的早期状态带到当前 case。

### 第二层：动作边界诊断

保留自由生成轨迹，但只在真实动作边界附近统计 logprob：

- `</think>` 后第一个可见输出 token。
- text vs `<tool_call>`。
- function name token，例如 `Write/Edit/Read/Bash`。
- 多工具调用时的继续 tool call vs 结束边界。

### 第三层：thinking-to-action 关联

按 recompute vs rope 将 case 分成四类：

| 类别 | thinking 关系 | action 关系 | 解释重点 |
|---|---|---|---|
| A | 相似 | 一致 | 正常稳定 |
| B | 相似 | 分叉 | 高层语义保持但动作边界脆弱 |
| C | 不同 | 分叉 | thinking 上下文语义漂移传导到动作 |
| D | 不同 | 一致 | 动作选择对部分 reasoning drift 有鲁棒性 |

## 已实现内容

新增采集脚本：

```text
scripts/06_context_free_segment_cache/run_thinking_action_diagnostic.py
scripts/06_context_free_segment_cache/run_thinking_action_diagnostic.sh
```

采集配置：

```text
tasks = all headline tasks
occurrences = 2,3
modes = recompute,rope
temperature = 0
top_p = 1
max_tokens = 4096
top_logprobs = 10
enable_thinking = true
```

采集输出：

```text
results/problem_exploration/thinking_to_action_divergence/data/free_generation_rows.jsonl
results/problem_exploration/thinking_to_action_divergence/data/token_logprob_rows.jsonl
results/problem_exploration/thinking_to_action_divergence/tables/thinking_action_case_summary.csv
```

新增离线分析脚本：

```text
scripts/06_context_free_segment_cache/analyze_thinking_action_diagnostic.py
```

分析输出：

```text
results/problem_exploration/thinking_to_action_divergence/tables/thinking_pair_summary.csv
```

结果入口：

```text
results/problem_exploration/thinking_to_action_divergence/source_manifest.csv
```

第一版新诊断只做 `recompute` vs `rope`，不纳入 `direct`。`direct` 可作为后续补充，但不进入本轮主实验。

## 语义指标实现

常用词面指标：

```text
BLEU
ROUGE-L F1
chrF
token Jaccard
```

embedding 指标：

```text
embedding cosine
```

默认本地 embedding 模型：

```text
/mnt/Large_Language_Model_Lab_1/模型/rag_models/BAAI-bge-base-en-v1.5
```

实现约束：

- 默认不联网下载模型。
- 默认不自动安装依赖。
- 在 `opencode` 环境中优先使用已有 `transformers + torch + numpy` 加载本地 BGE 模型并做 mean pooling。
- 如果 embedding 模型或依赖不可用，分析脚本写入 `embedding_metric_status=unavailable`，并继续输出 BLEU、ROUGE-L、chrF、token Jaccard 与 task-grounding / intent 指标。

任务结构指标：

```text
intent_label
intent_match
mentioned_tool_overlap
mentioned_file_overlap
keyword_overlap
grounding_conflict
```

pair 分类：

```text
A_thinking_similar_action_same
B_thinking_similar_action_diverged
C_thinking_different_action_diverged
D_thinking_different_action_same
```

`intent_label` 和 `intent_match` 来自脚本对 thinking 文本的粗粒度启发式抽取，不是人工标注。当前很多样本会被归为 `multi_step`，因此 `intent_match=True` 只表示脚本没有检测到明显 next-action intent 差异，不能等价于人工确认两条 thinking 的下一步意图一致。B 类 case 仍需人工复核，以排除 recompute 与 rope thinking 实际已经指向不同动作、但被 parser 都标为 `multi_step` 的假阳性。

## B/C 复核与 candidate scoring 准备

新增离线脚本：

```text
scripts/06_context_free_segment_cache/build_boundary_candidate_review.py
scripts/06_context_free_segment_cache/summarize_boundary_candidate_scoring.py
```

输入：

```text
results/problem_exploration/thinking_to_action_divergence/tables/thinking_pair_summary.csv
results/problem_exploration/thinking_to_action_divergence/data/free_generation_rows.jsonl
results/problem_exploration/thinking_to_action_divergence/data/token_logprob_rows.jsonl
src/traces/_tools.json
```

输出：

```text
results/problem_exploration/thinking_to_action_divergence/tables/bc_case_manual_review.csv
results/problem_exploration/thinking_to_action_divergence/tables/boundary_candidate_scoring_plan.csv
results/problem_exploration/thinking_to_action_divergence/tables/candidate_scoring_summary.csv
```

实现原则：

- 只筛选 `B_thinking_similar_action_diverged` 和 `C_thinking_different_action_diverged`。
- 候选集合不固定写死为某几个工具名，而是由 observed divergence 的动作差异和当前真实工具列表派生。
- `<tool_call>` 候选使用 `tool_call_start_token_index`，文本候选使用 `visible_start_token_index`，工具名候选优先使用 `function_name_token_index`。
- 若候选不在现有 top-10 logprobs 中，标记 `needs_forced_scoring=True`，不猜测分数。

当前离线覆盖结果：

| 项目 | 数量 |
|---|---:|
| B/C case | 10 |
| manual-review rows | 20 |
| scoring-plan rows | 244 |
| observed divergence candidate rows | 52 |
| observed candidate rows already supported by top-k/generated token | 44 |
| observed candidate rows needing forced scoring | 8 |
| candidate-scoring summary rows | 10 |
| manual-confirmed boundary flip rows | 6 |
| partial boundary flip evidence rows | 2 |
| manual-unclear rows | 1 |
| intent-drift possible rows | 1 |
| rows needing forced scoring for qualitative label | 0 |
| rows pending manual intent review | 0 |

2026-06-19 对 `boundary_candidate_scoring_plan.csv` 的初步分析：

- 本段中的“翻转”指同一类动作边界处候选分数排序反向，不使用 `A > B` 这类容易被误解为动作转移的写法。
- function-name 翻转已有直接 top-k 证据：`internal_comms_incident_update/internal-comms/occ2`、`launch_poster_page_pack/canvas-design/occ3`、`mcp_server_and_spec/mcp-builder/occ3`、`slack_launch_pack/slack-gif-creator/occ3` 均显示 recompute 与 rope 在 observed tool-name 候选之间发生排序翻转。
- tool/text 翻转也已有直接 top-k 证据：`doc_coauthoring_design_doc/doc-coauthoring/occ3`、`launch_poster_page_pack/theme-factory/occ2`、`slack_launch_pack/slack-gif-creator/occ2` 等 case 显示 `<tool_call>` 与文本起始 token 的排序在 recompute/rope 间反向。
- 8/52 个 `needs_forced_scoring=True` 多数是反事实路径候选缺失，例如 text 路径下的 function-name 候选或 tool 路径下的另一路 text 起始 token。它们影响精确 margin，不应被直接解读为现有 top-k 无法支持定性排序分析。

2026-06-20 `candidate_scoring_summary.csv` 聚合结果：

- 共 10 行 B/C action-divergence boundary summary。
- 人工复核前：8 行为 `confirmed_boundary_flip`，2 行为 `partial_boundary_flip_evidence`。
- 人工复核后：6 行为 `manual_status=confirmed_similar` 且 `evidence_label=confirmed_boundary_flip`。
- 人工复核后：2 行为 `manual_status=confirmed_similar` 且 `evidence_label=partial_boundary_flip_evidence`。
- 人工复核后：1 行为 `manual_unclear`，1 行为 `intent_drift_possible`。
- 0 行需要 forced scoring 才能给出定性标签；如果 partial case 后续成为论文关键例子，再考虑补 forced scoring。

2026-06-20 `/home/wsh/vllm` ContextSegmentKV 注入代码复核：

- 请求侧 `vllm/v1/request.py` 会从 `sampling_params.extra_args["context_segment_cache"]` 解析 `sources` 与 `targets`，并支持 `mode=direct/rope/disabled`。非法 mode 现在会直接报错，避免静默降级。
- scheduler 侧会用 `get_max_cache_hit_length()` 限制普通 prefix cache 命中不能跨过最早可注入 target span；随后在 `num_computed_tokens == target_start` 时生成 injection metadata。
- 注入时不再把复用 span 当作外部已计算 token 跳过 block 分配，而是为 target span 正常分配物理 KV slots，然后把 `num_scheduled_tokens[req_id]` 记为 0，并把 scheduler 侧 `request.num_computed_tokens` 推到 `target_end`。
- worker 侧先校验 cache 是否存在、block ids 是否存在、长度是否一致、source/target token ids 是否完全一致；`rope` mode 只重旋 key，不改 value；`direct` mode 直接 scatter。
- 这条核心数据流符合当前实验假设：target span 在 block table 中有真实 slots，后续 token 可以 attention 到注入的 K/V。
- 发现一个默认路径风险：`vllm/v1/worker/gpu/model_runner.py` 已经删除 `num_scheduled_tokens == 0` 的注入-only 请求，避免 forward 看到 0-token request；但默认环境 `VLLM_USE_V2_MODEL_RUNNER=0` 会走 `vllm/v1/worker/gpu_model_runner.py`，该文件单独只有注入请求时会 early return，混合 batch 中却可能保留 0-token request 进入 `_prepare_inputs()` 和 attention metadata。这个风险应在继续跑批量实验前修复。
- 额外低风险改进：`validate_token_identity()` 当前能拒绝 token mismatch，但如果旧 `.pt` 文件缺 `token_ids` 会直接报错；这是安全行为，但需要确保后续 KV 都用新格式生成。也可以把 token-id 长度检查写得更显式，方便排错。

2026-06-20 阶段级 `results/problem_exploration/summary.md` 同步：

- 更新研究内容导航中的 `thinking_to_action_divergence/` 结论：从“9/10 自动 B 类”改为人工复核和 candidate scoring 后的 8/10 boundary-level 支持，其中 6 个完整强证据、2 个 partial evidence。
- 更新阶段总论：明确 10 个 B/C action-divergence boundary 的人工复核后分布为 6 strong、2 partial、1 unclear、1 intent drift possible。
- 新增 `/home/wsh/vllm` 注入代码复核边界：单请求链路成立；默认旧 GPU runner 混合 batch 0-token 注入请求过滤是后续系统设计风险。
- 新增 4.5 `Thinking-to-action` 小节，把旧 “待补 logprob/margin 诊断” 改为已完成的动作边界 top-k / candidate scoring 证据链。
- 更新下一阶段建议：优先进入 boundary halo / 局部 recompute / margin-aware 触发设计；forced scoring 不再是阻塞项。

## 尚未实现

- 尚未对 2 个 partial evidence case 做 forced scoring；当前判断为非必要，除非后续叙事需要精确 margin。
- 尚未做局部修复实验。

## 当前风险

- 如果 vLLM chat logprobs 对 tool call 的 function name 暴露不完整，需要改用 raw token stream 或额外 completion 格式来定位边界。
- 如果 4096 token logprob 采集成本过高，可以先只跑 headline 24 case 的 `recompute/rope` 单样本，不做重复采样。
- thinking 语义指标需要避免只看表面相似；必须显式检查 task-state grounding 和 next-action intent。当前 intent parser 较粗，B 类分类在人工复核前应表述为“现有自动指标下 thinking 相似”。
- `/home/wsh/vllm` 默认旧 GPU runner 的混合 batch 路径可能让注入-only 请求以 0 scheduled tokens 进入 forward batch。单请求实验可能不触发，但批量并发或同一步混合调度可能触发，应优先修复。

## 本轮修改与验证

本轮修改文件：

```text
AGENTS.md
results/problem_exploration/logprob_margin_diagnostic/summary.md
agent_md/segmentia_thinking_to_action_divergence_development.md
scripts/06_context_free_segment_cache/run_thinking_action_diagnostic.py
scripts/06_context_free_segment_cache/run_thinking_action_diagnostic.sh
scripts/06_context_free_segment_cache/analyze_thinking_action_diagnostic.py
scripts/06_context_free_segment_cache/plot_thinking_action_diagnostic.py
scripts/06_context_free_segment_cache/build_boundary_candidate_review.py
scripts/06_context_free_segment_cache/summarize_boundary_candidate_scoring.py
results/problem_exploration/thinking_to_action_divergence/source_manifest.csv
results/problem_exploration/thinking_to_action_divergence/summary.md
results/problem_exploration/thinking_to_action_divergence/figures/thinking_action_category_counts.png
results/problem_exploration/thinking_to_action_divergence/figures/embedding_vs_boundary_margin_delta.png
results/problem_exploration/thinking_to_action_divergence/figures/boundary_margin_pairs.png
results/problem_exploration/thinking_to_action_divergence/tables/bc_case_manual_review.csv
results/problem_exploration/thinking_to_action_divergence/tables/boundary_candidate_scoring_plan.csv
results/problem_exploration/thinking_to_action_divergence/tables/candidate_scoring_summary.csv
agent_md/segmentia_thinking_to_action_divergence_development.md
results/problem_exploration/summary.md
```

2026-06-19 文档边界修正：

```text
AGENTS.md
results/problem_exploration/thinking_to_action_divergence/summary.md
agent_md/segmentia_thinking_to_action_divergence_development.md
```

2026-06-19 B/C 复核与 candidate scoring 准备：

```text
scripts/06_context_free_segment_cache/build_boundary_candidate_review.py
scripts/06_context_free_segment_cache/summarize_boundary_candidate_scoring.py
results/problem_exploration/thinking_to_action_divergence/tables/bc_case_manual_review.csv
results/problem_exploration/thinking_to_action_divergence/tables/boundary_candidate_scoring_plan.csv
results/problem_exploration/thinking_to_action_divergence/tables/candidate_scoring_summary.csv
results/problem_exploration/thinking_to_action_divergence/source_manifest.csv
results/problem_exploration/thinking_to_action_divergence/summary.md
agent_md/segmentia_thinking_to_action_divergence_development.md
```

本轮运行的验证：

```text
python - <<'PY' ... headline decode_outputs.jsonl 审计
python - <<'PY' ... logprob_margin_rows.jsonl 覆盖率审计
python - <<'PY' ... margin_first_diff_summary.csv 统计
python -m py_compile scripts/06_context_free_segment_cache/run_thinking_action_diagnostic.py scripts/06_context_free_segment_cache/analyze_thinking_action_diagnostic.py
conda run -n opencode python -m py_compile scripts/06_context_free_segment_cache/run_thinking_action_diagnostic.py scripts/06_context_free_segment_cache/analyze_thinking_action_diagnostic.py
bash -n scripts/06_context_free_segment_cache/run_thinking_action_diagnostic.sh
python scripts/06_context_free_segment_cache/run_thinking_action_diagnostic.py --help
python scripts/06_context_free_segment_cache/analyze_thinking_action_diagnostic.py --help
conda run -n opencode python scripts/06_context_free_segment_cache/analyze_thinking_action_diagnostic.py
MPLCONFIGDIR=/tmp/matplotlib-cache python scripts/06_context_free_segment_cache/plot_thinking_action_diagnostic.py
python -m py_compile scripts/06_context_free_segment_cache/plot_thinking_action_diagnostic.py scripts/06_context_free_segment_cache/analyze_thinking_action_diagnostic.py
rg -n "intent parser|intent_match|误判|启发式" AGENTS.md results/problem_exploration/thinking_to_action_divergence/summary.md agent_md/segmentia_thinking_to_action_divergence_development.md
python -m py_compile scripts/06_context_free_segment_cache/build_boundary_candidate_review.py
python scripts/06_context_free_segment_cache/build_boundary_candidate_review.py --help
python scripts/06_context_free_segment_cache/build_boundary_candidate_review.py
python - <<'PY' ... 汇总 boundary_candidate_scoring_plan.csv 的 observed candidate top-k 覆盖
python -m py_compile scripts/06_context_free_segment_cache/summarize_boundary_candidate_scoring.py
python scripts/06_context_free_segment_cache/summarize_boundary_candidate_scoring.py --help
python scripts/06_context_free_segment_cache/summarize_boundary_candidate_scoring.py
python - <<'PY' ... 汇总 candidate_scoring_summary.csv 的 evidence_label、divergence_group 和 manual_status
```

验证结果：

- headline 产物完整保存 reasoning/action，可用于 thinking 语义分析。
- 当前 margin 产物 70/72 条没有到达 `</think>`，不足以支持动作边界 margin 分析。
- 用户已确认实现 `recompute` vs `rope` 版本。
- 新增 Python 脚本在系统 Python 与 `opencode` 环境中均通过 `py_compile`。
- 新增 shell wrapper 通过 `bash -n`。
- 两个 Python 脚本的 `--help` 均可正常加载。
- 用户完成 vLLM 长实验后，采集结果完整：48/48 free-generation rows，0 error，48/48 找到 `</think>`，37/48 为 token boundary，11/48 为 text boundary。
- 离线分析生成 24 个 recompute-vs-rope pair；embedding metric status 为 available。
- 首轮修复了 `thinking_similar` 分类中的布尔/字符串比较 bug，并重新生成 `thinking_pair_summary.csv`。
- 生成三张总结图和 `summary.md`。
- 2026-06-19 修正 `summary.md` 与 `AGENTS.md` 的文档边界：明确 `intent_match` 是粗粒度 parser 结果，不是人工确认；B 类结论在人工复核前应保留 parser 假阳性风险。
- 2026-06-19 新增 B/C 复核与 candidate scoring 准备脚本，生成 20 行人工复核表和 244 行候选覆盖表；observed divergence candidate rows 中 44/52 已被现有 top-k/generated token 覆盖，8/52 需要后续视情况 forced scoring。
- 2026-06-19 分析 `boundary_candidate_scoring_plan.csv`：多个 function-name 与 tool/text divergence case 已有直接 top-k 排序翻转证据；8/52 缺失多为反事实路径候选，主要影响精确 margin 而不是定性 go/no-go。
- 2026-06-20 新增 `candidate_scoring_summary.csv`：10 个 B/C action-divergence boundary 中 8 个为 `confirmed_boundary_flip`，2 个为 `partial_boundary_flip_evidence`；当前 10 行人工复核状态均为 `pending`。
- 2026-06-20 合并人工复核结果后重新生成 `candidate_scoring_summary.csv`：6 个 manual-confirmed boundary flip，2 个 confirmed-similar partial evidence，1 个 manual unclear，1 个 intent drift possible。
- 2026-06-20 完整重写 `results/problem_exploration/thinking_to_action_divergence/summary.md`：统一研究问题、数据链路、指标定义、人工复核、candidate scoring、代表 case、最终解释、限制和下一步建议；当前总体结论为 8/10 个 B/C boundary 在人工复核后仍支持 boundary-level 解释，其中 6 个是完整强证据、2 个是 partial evidence。
- 2026-06-20 复核 `/home/wsh/vllm` ContextSegmentKV 注入/复用代码：核心 scheduler/worker 数据流方向正确；发现默认旧 GPU runner 的混合 batch 0-token 注入请求过滤风险。尝试运行 pytest 时，系统 Python 缺 `tblib`、`opencode` 环境缺 `pytest`；改用 AST parse 检查 12 个相关 Python 文件，结果通过。
- 2026-06-20 更新 `results/problem_exploration/summary.md`：阶段级结论与下一步建议已同步到人工复核、candidate scoring 和 vLLM 单请求代码复核后的当前状态；确认旧的“待补 logprob/candidate scoring/人工复核”表述已移除。
