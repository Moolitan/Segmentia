# Skill Prefill Attention

该目录实现 recompute 条件下的正常 prefill attention 采集。它不比较 rope，也不
注入任何 context-free Skill KV。

## 采集对象

每个 occurrence-3 case 采集两组 query：

1. Segmentia `target_start` 起始的前 30 个 Skill token。每个 token 保留自己
   对固定窗口 `[target_start-512, target_start)` 的 attention row。
2. Qwen chat template 最后一个 prompt token，即
   `<|im_start|>assistant\n` 中的最后一个 token，绝对位置为
   `prompt_tokens-1`。它保留对
   `[target_start-512, target_end)` 的 attention。

attention softmax 始终在每个 query 的完整 causal prefix 上计算。导出的窗口
只是从完整概率中截取，不会再次除以窗口内概率和，因此窗口总 attention mass
可以小于 1。

FlashAttention 不物化完整 attention matrix。vLLM probe 在正常 prefill
forward 中读取同一组已应用位置编码的 Q 和 K，仅对选中的 query rows 用
float32 重算 `softmax(QK^T * scale)`。该诊断计算不覆盖 forward tensor，也不
改变生成结果。

## 服务与缓存边界

```text
for task in task_order:
  restart vLLM in recompute mode
  clear prefix cache through restart
  replay this task's occurrence-3 cases in invocation_index order
```

服务边界是 `(recompute, task)`。recompute 请求的
`context_segment_cache=None`，当前和历史 Skill span 均不显式注入。一个 task
中后续 case 可以自然命中同 task 先前请求形成的 prefix cache；不同 task 不会
共享 prefix cache。

## 输出

大体量逐层 JSONL 写入：

```text
/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/
  06_context_free_segment_cache/skill_prefill_attention/occ3/
```

轻量结果写入：

```text
results/problem_exploration/skill_prefill_attention/
```

每个 case 生成：

- `skill_first30_to_pre512_*.{png,pdf}`：heads 和 40 层平均后的 `30×512`
  heatmap。
- `final_assistant_to_context_skill_*.{png,pdf}`：仅 heads 平均、保留 40 层的
  `40×(512+Skill长度)` heatmap。

两个图都显示原始 attention probability，不做窗口内归一化。

## 运行

默认发现已有输出时停止，防止混入旧 JSONL。确认要替换派生输出时：

```bash
cd /home/wsh/openhands_code_research
conda activate opencode
CLEAN_OUTPUT=1 \
  bash scripts/06_context_free_segment_cache/cross_occurrence_controller/\
raw_decode_token_sequences/skill_prefill_attention/run_skill_prefill_attention.sh
```

该命令会启动多次 vLLM 服务并执行完整 prefill probe，应由用户手动运行。
