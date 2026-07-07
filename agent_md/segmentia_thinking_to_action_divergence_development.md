# Segmentia Thinking-to-Action Divergence Development

## 总开发目标

区分 context-free skill KV reuse 对 thinking 语义、完整 assistant turn 行为和局部 token 选择的影响，避免把不同 token 位置的 margin 混成统一动作边界。

## 开发阶段总览

| 阶段 | 名称 | 目标 | 当前进度 | 剩余 |
|---|---|---|---|---|
| 1 | Headline 行为差异 | 用结构化 `tool_calls` 验证完整 Agent 行为是否改变。 | 已完成。 | 无。 |
| 2 | 完整 logprob 采集 | 复现相同 case 并保存逐 token top-k。 | 已完成，48行 free generation、46971行 token 数据。 | 无。 |
| 3 | Thinking 分类 | 比较 thinking 语义并构造 A/B/C/D。 | 已完成。 | 自动 intent 判定仍有误判边界。 |
| 4 | 行为语义纠偏 | 拆分 tool-call presence、tool trajectory 和三个 token observations。 | 已完成。 | 无。 |
| 5 | Evidence 重分类 | 只将双方 tool call 的工具名变化定位到 function-name token。 | 已完成。 | Presence flip 的 sequence-level 机制待研究。 |

## 当前数据血缘

Headline：

```text
decode_outputs.jsonl
  -> evaluate_outputs.py
  -> structured turn behavior metrics
```

Thinking diagnostic：

```text
同一批 case 重新 chat completion
  + logprobs=True
  -> free_generation_rows.jsonl
  -> token_logprob_rows.jsonl
  -> corrected offline summaries
```

两次运行的48个 `recompute/rope` action labels 完全一致。

## 当前定义

Turn-level：

```text
tool_call_presence
tool_trajectory
```

Token-level：

```text
visible_start
tool_call_start
function_name
```

三个 token 位置分别保存，不再生成统一 `action_boundary_margin`。

## 当前结果

| 指标 | 结果 |
|---|---:|
| pair 数 | 24 |
| 完整行为一致 | 14 |
| tool-call presence flip | 6 |
| tool trajectory flip | 4 |
| thinking 相似且行为分歧 | 9 |

Evidence：

| label | 数量 |
|---|---:|
| `confirmed_function_name_boundary_flip` | 4 |
| `confirmed_tool_call_presence_flip_unlocalized` | 4 |
| `manual_unclear` | 1 |
| `intent_drift_possible` | 1 |

## 本轮修改文件

- `run_thinking_action_diagnostic.py`
- `analyze_thinking_action_diagnostic.py`
- `build_divergence_evidence.py`
- `plot_thinking_action_diagnostic.py`
- 对应 README、result summary、source manifest 和阶段 summary

本轮进一步将 `results/problem_exploration/thinking_to_action_divergence/summary.md` 重写为可独立阅读的当前研究报告。文档只陈述当前实验设置、指标、结果、证据边界、限制和下一步，不记录历史结论或纠错过程。

本轮删除 `build_boundary_candidate_review.py` 和 `summarize_boundary_candidate_scoring.py`，改为单一 `build_divergence_evidence.py`。新脚本直接读取 pair、free generation、token logprobs 和已完成人工复核，不生成中间 candidate plan，也不会覆盖人工标注。

## 本轮验证

- 所有修改脚本通过 `python -m py_compile`。
- 本地 BGE embedding 可用，离线重算24个 pair 成功。
- Corrected pair summary 不再包含 legacy generic boundary 字段。
- Corrected evidence summary 为10行：6行 turn-level、4行 token-level。
- `build_divergence_evidence.py` 重建的10行 evidence CSV 与替换前逐字节一致。
- 新图 `turn_divergence_type_counts.png` 已生成并人工检查。
- 用真实 mixed response 验证分离位置为 `visible_start=433 < tool_call_start=439 < function_name=445`，新采集 schema 不再生成 `action_boundary_*`。
- Headline 与 diagnostic 的48个结构化 action labels 再次核对为48/48一致。
- 未启动 vLLM，未改动原始 JSONL。

## 下一步

对6个 tool-call presence flip 设计 sequence-level 风险诊断。该诊断必须解释完整 turn 是否最终包含 tool call，不能重新使用 first-visible token 作为替代标签。
