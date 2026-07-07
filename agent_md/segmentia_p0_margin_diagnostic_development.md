# Segmentia P0 Logprob Margin Diagnostic Development

## 开发阶段总览

| 阶段 | 名称 | 目标 | 当前进度 | 剩余 |
|---|---|---|---|---|
| 1 | 原理确认 | 解释为什么看 logprob margin 能验证低 margin 动作判决点假设 | 已完成，用户确认 | 无 |
| 2 | 客户端支持 | 让 vLLM OpenAI-compatible 请求可选返回 `logprobs/top_logprobs` | 已完成 | 无 |
| 3 | 诊断脚本 | 新增 P0 Python 脚本，输出 token-level 和 case-level margin 数据 | 已完成 | 无 |
| 4 | 运行 wrapper | 新增 shell wrapper，负责 per-task/per-mode 重启 vLLM | 已完成 | 无 |
| 5 | 结果目录与文档 | 新增 `logprob_margin_diagnostic/` 研究内容目录和 manifest | 已完成 | 无 |
| 6 | 静态验证 | 只做 py_compile/bash -n，不启动实验 | 已完成 | 无 |
| 7 | 结果分析 | 分析用户跑完的 P0 logprob/margin 结果并写回 `results/` | 已完成，2026-06-19 已修正结论边界 | 新诊断另起实现 |
| 8 | 结果图示化 | 将 P0 结果分析补成核心图，并更新 summary/manifest/AGENTS | 已完成 | 无 |
| 9 | 结论边界复查 | 明确旧实验主要分析 hidden reasoning early window，而非动作边界 | 已完成 | 无 |

## 目标

验证 P0 假设：

> 复用 KV 后，语义基本不变，但在低 margin 的离散动作判决点上，value drift 会把 argmax 翻到另一条行为轨迹。

如果 P0 成立，后续设计应优先考虑 low-margin action decision point 的局部校正，而不是全 span 重写 KV。

## 原理

模型每生成一个 token 时，会给所有候选 token 分配 logprob。top-1 和 top-2 的差值为 margin：

```text
margin = top1_logprob - top2_logprob
```

margin 大表示模型很确定；margin 小表示模型在两个候选之间很犹豫。对于普通文本，低 margin 通常只会改变措辞；对于 agent 动作，低 margin 可能直接翻转工具调用轨迹，例如：

```text
Write vs Edit
tool call vs text
Read vs Write
```

因此 P0 要检查发散 case 是否集中在 recompute 本来就低 margin 的动作点，并检查 direct/rope 是否在这些点改变 top-1。

## 代码逻辑

新增文件：

```text
scripts/06_context_free_segment_cache/logprob_margin_diagnostic/run_margin_diagnostic.py
scripts/06_context_free_segment_cache/logprob_margin_diagnostic/run_margin_diagnostic.sh
```

代码流程：

1. 读取 headline 阶段已有结果：

```text
results/problem_exploration/headline_semantic_action_gap/data/decode_outputs.jsonl
```

2. 按 `(task, skill, occurrence, invocation_index)` 分组，得到 recompute/direct/rope 的 action label。
3. 用和 `module/replay.py`、各研究目录本地 `decode_compare.py` 相同的 case selection、message conversion、context segment cache 配置重新请求 vLLM。
4. 请求中打开：

```text
logprobs=True
top_logprobs=10
temperature=0
```

5. 对每个生成 token 记录：

```text
token_index
generated_token
generated_logprob
top1_token / top1_logprob
top2_token / top2_logprob
margin
top_logprobs
```

6. 写出 token-level JSONL：

```text
results/problem_exploration/logprob_margin_diagnostic/data/logprob_margin_rows.jsonl
```

7. 汇总 case/mode-level 表：

```text
results/problem_exploration/logprob_margin_diagnostic/tables/margin_case_summary.csv
```

## 运行方式

由用户运行：

```bash
bash scripts/06_context_free_segment_cache/logprob_margin_diagnostic/run_margin_diagnostic.sh
```

可缩小范围试跑：

```bash
TASKS=doc_coauthoring_design_doc MODES=recompute,rope \
bash scripts/06_context_free_segment_cache/logprob_margin_diagnostic/run_margin_diagnostic.sh
```

## Go / No-Go 判据

以下是原始 P0 margin 诊断的 Go / No-Go 判据；2026-06-19 复查后确认，该判据实际只覆盖 early hidden reasoning window，不能直接判定动作边界。

GO：

- 发散 case 的 recompute early-generation window 最小 margin 系统性低于不发散 case。
- direct/rope 在 recompute 低 margin token index 附近改变 top-1 或 generated token。

NO-GO：

- 发散和不发散 case 的 margin 没有系统性差异。
- top-1 改变不发生在低 margin 或动作相关 token 附近。

## 结果分析结论

用户已完成 P0 实验运行，结果已写回：

```text
results/problem_exploration/logprob_margin_diagnostic/
```

本次分析生成：

```text
tables/margin_group_summary.csv
tables/margin_case_diagnostic_summary.csv
tables/margin_first_diff_summary.csv
```

补充生成：

```text
figures/margin_case_risk_map.png
figures/margin_group_comparison.png
figures/margin_first_diff_scatter.png
```

原始分析的核心结论曾写作 **Partial GO**。2026-06-19 复查后，结论边界修正为 **Diagnostic GO, Action-Boundary Inconclusive**：

- 24 个 case × 3 mode 的结果完整，case-level 共 72 条记录，token-level 共 18432 条记录。
- `direct` 与 `rope` 各有 10/24 个 action 分歧；按 case 合并后，14/24 个 case 有任一 reuse mode 发生 action 分歧。
- 所有发生 action 分歧的 case，recompute early-generation window 最小 margin 都为 0。
- 该 window 是 completion 前 128 个生成 token；由于启用了 thinking，主要位于 `<think>` hidden reasoning 内。
- 当前 margin run 每条只采集 256 个 token；70/72 条没有到达 `</think>`，只有 2/72 条触达 `<tool_call>`。
- 因此低 margin 是 early hidden reasoning 轨迹脆弱性的伴随风险信号，不是动作边界低 margin 的直接证据。
- margin 为 0 的 token 大多是 `the`、逗号、句号等自然语言或格式 token；当前自由生成 logprob 尚未精确定位到动作候选本身。

下一步应另起 `thinking_to_action_divergence` 诊断：保留自由生成轨迹，显式分析 `thinking 语义漂移 -> 动作边界 margin -> action divergence`。固定 prefix 的 action-candidate scoring 只作为补充机制隔离实验，不替代主诊断。

## 图示化补充

本轮新增：

```text
scripts/06_context_free_segment_cache/logprob_margin_diagnostic/plot_margin_diagnostic.py
```

该脚本只读取已完成实验的 CSV 后处理表，不启动 vLLM，不重新 decode。生成三张图：

- `margin_case_risk_map.png`：展示所有 action 分歧 case 都覆盖在 recompute zero-margin window 内，同时显示 zero-margin 并非充分条件。
- `margin_group_comparison.png`：展示分歧组与非分歧组的平均 margin 差异。
- `margin_first_diff_scatter.png`：展示第一次 token 分叉多发生在普通文本生成区域，不能直接等价动作判决点。

同时已补充 `AGENTS.md` 规范：后续实验结果分析默认应尽量图示化，至少沉淀核心图并写入 `summary.md` 与 `source_manifest.csv`。

## 当前限制

- 第一版只做 token-level 和 case-level margin 记录，且主要覆盖 hidden reasoning early window，没有完成真实动作候选的精确对齐。
- vLLM 对 chat completion 的 logprobs 返回格式可能因版本不同而变化，脚本已经兼容 OpenAI chat-style 和 completion-style 两种常见格式。
- 本次脚本复用了 headline 阶段的 action label；logprob run 本身没有额外保存完整 tool call 解析结果。后续 thinking-to-action 诊断应把本轮 reasoning、content、tool_calls、action label 和 token-level logprobs 一并写入。

## 静态验证

已完成：

```text
python -m py_compile scripts/06_context_free_segment_cache/module/vllm_client.py scripts/06_context_free_segment_cache/logprob_margin_diagnostic/run_margin_diagnostic.py
bash -n scripts/06_context_free_segment_cache/logprob_margin_diagnostic/run_margin_diagnostic.sh
```

未执行：

```text
bash scripts/06_context_free_segment_cache/logprob_margin_diagnostic/run_margin_diagnostic.sh
```

原因：按 `AGENTS.md` 约束，实验由用户手动启动。用户已在后续完成运行，本轮只做结果分析与文档整理。
