# Segmentia Skill Prefill Attention Development

## 总开发目标

在 recompute 条件下采集正常 prefill 中 Skill token 和最终 assistant prompt
token 的局部 attention，以原始概率 heatmap 展示 Skill 如何读取前文，以及
生成入口如何读取前文和完整 Skill。

## 开发阶段总览

| 阶段 | 名称 | 目标 | 当前进度 | 剩余 |
|---|---|---|---|---|
| 1 | 语义与边界确认 | 明确 query、key、prompt 尾部和不归一化语义。 | 已完成。 | 无。 |
| 2 | Probe 扩展 | 支持每个 query 独立窗口并保证 prompt query 的 causal key 边界。 | 已完成，静态和直接单元检查通过。 | 真实 vLLM 验证。 |
| 3 | 采集入口 | recompute-only 按 `(mode, task)` 重启并按 invocation 顺序采集。 | 已完成，未运行长实验。 | 用户运行真实采集。 |
| 4 | 验证与绘图 | 校验完整行集并生成两类 heatmap。 | 已完成，合成数据验证通过。 | 用真实 dump 生成结果。 |
| 5 | 结果分析 | 解释 token/层级 attention 结构和限制。 | 未开始。 | 等待真实结果。 |

## 已确认的数据语义

### Skill token heatmap

```text
query = target_start ... target_start+29
key   = target_start-512 ... target_start-1
```

probe 对每层、每个 query 保存一个已经对 query heads 平均的 512 元素 row。
绘图再对 40 层求均值，不对 30 个 query token 求平均，最终矩阵为 `30×512`。

### Final assistant heatmap

Qwen chat template 在当前 12 个 occurrence-3 case 中均以：

```text
</context_segment>
</tool_response><|im_end|>
<|im_start|>assistant\n
```

结束。final-assistant query 是最后一个 prompt token：

```text
query_abs_position = prompt_tokens - 1
key window = [target_start-512, target_end)
```

该窗口连续包含 Skill 前 512 个 token 和 Segmentia 定义的完整 target span。
probe 对 heads 平均但保留 40 层，最终矩阵为
`40×(512+target_end-target_start)`。

### 不归一化

每个 query 的 softmax 分母覆盖其完整 causal prefix。probe 只从完整概率中截取
目标 key window，不把窗口内概率重新缩放为总和 1。因此 heatmap 同时保留窗口
内部的相对分布和模型实际分给该窗口的绝对 attention mass。

## 服务、缓存与失败边界

- 服务重启边界是 `(recompute, task)`。
- 每个 task 内由 `selected_cases()` 按 `invocation_index` 递增 replay。
- recompute 的 `context_segment_cache` 为 `None`，不显式注入当前或历史 Skill
  KV；task 内 prefix cache 只保留同 task 请求自然产生的前缀。
- 不同 task 通过重启隔离 prefix cache。
- 每个请求使用完整 trace messages 和 tools；chat template 添加 assistant
  generation prompt，随后只生成 1 个 token。
- 已有 manifest 或 dump 时默认停止；只有显式 `CLEAN_OUTPUT=1` 才替换派生
  输出。
- runner、probe 或绘图校验失败时立即退出，不跳过 case，不产出部分结论。

## 本轮修改文件

- `/home/wsh/vllm/vllm/v1/context_segment_cache/attention_probe.py`
- `scripts/06_context_free_segment_cache/cross_occurrence_controller/raw_decode_token_sequences/skill_prefill_attention/run_skill_prefill_attention.py`
- `scripts/06_context_free_segment_cache/cross_occurrence_controller/raw_decode_token_sequences/skill_prefill_attention/plot_skill_prefill_attention.py`
- `scripts/06_context_free_segment_cache/cross_occurrence_controller/raw_decode_token_sequences/skill_prefill_attention/run_skill_prefill_attention.sh`
- `scripts/06_context_free_segment_cache/cross_occurrence_controller/raw_decode_token_sequences/skill_prefill_attention/README.md`
- `results/problem_exploration/skill_prefill_attention/summary.md`
- `results/problem_exploration/skill_prefill_attention/source_manifest.csv`
- `agent_md/segmentia_skill_prefill_attention_development.md`

## 本轮验证

- 本地 tokenizer 核对 12/12 个 occurrence-3 prompt 均以
  `<|im_start|>assistant\n` 结束；`prompt_tokens=target_end+3`，最后 query 为
  `target_end+2`。
- 12/12 个 target span 长度均大于 30，且 `target_start` 均大于 512。
- marker 边界测试确认每个 case 有 30 个连续 Skill query 和 1 个 final
  assistant query；两个 key window 均符合定义。
- vLLM `begin_step()` 直接测试确认 prefill query 的诊断 key 长度被截断为
  `query_abs_position+1`，final query 使用完整 prompt。
- 人工小 tensor 测试确认 probe 只对 causal keys 做 softmax，截取窗口后的
  attention mass 保持小于 1，没有窗口内重新归一化；旧版 `query_positions`
  marker 仍可读取。
- 两个 Python 入口通过编译和 `--help`；shell 入口通过 `bash -n`。
- 使用 1 个合成 case 的 1240 条 dump rows 完成严格校验，成功生成两类 PNG、
  PDF、汇总 CSV 和逐文件 source manifest；图片已目视检查。
- 未启动 vLLM，未执行真实 Qwen3-14B prefill 实验。

## 当前问题与下一步

当前无代码阻塞。下一步由用户运行真实采集；运行后必须先确认 12 case 均各有
1200 条 Skill-query row 和 40 条 final-assistant row，再分析 heatmap。真实
结果出来前，结果 summary 保持“无实验结论”。

当前 `/home/wsh/vllm` 的 Git 索引将
`vllm/v1/context_segment_cache/attention_probe.py` 显示为未跟踪文件；本地
server 会使用工作树中的实现，但后续迁移或提交 vLLM 修改时必须显式纳入该文件。
