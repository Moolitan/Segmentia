# Segmentia Raw Decode Visualization Development

## 总开发目标

把 raw decode token sequence 审计中的逐 case embedding cosine 数据转成可复查、
可复现的描述性图，同时保持单次采样不能证明 RoPE 等价性的证据边界。

## 开发阶段总览

| 阶段 | 名称 | 目标 | 当前进度 | 剩余 |
|---|---|---|---|---|
| 1 | 数据结构检查 | 确认比较组、arm、case 数量和 cosine 数值有效。 | 已完成。 | 无。 |
| 2 | 图形实现 | 展示 thinking/action 配对关系和三个比较组的逐 case 分布。 | 已完成。 | 无。 |
| 3 | 结果沉淀 | 输出 PNG/PDF并更新 summary 与 source manifest。 | 已完成。 | 无。 |

## 实现逻辑

输入为：

```text
results/problem_exploration/raw_decode_token_sequences/tables/
  temp0.6_without_occ12_embedding_cosine.csv
```

绘图脚本按 `comparison` 分成以下三组：

```text
sampling_baseline                 -> recompute_run1 / recompute_run2
same_seed_recompute_vs_rope       -> recompute_run1 / rope
cross_seed_recompute_vs_rope      -> recompute_run2 / rope
```

每组必须有 12 个唯一 `filename`。脚本将 `thinking_cosine` 和
`action_cosine` 转成浮点数，拒绝 NaN、Inf 和 `[-1, 1]` 外的值。左面板按 case
绘制 thinking-action 散点，并用菱形标出各组算术均值；右面板按比较组绘制箱线，
同时叠加全部逐 case 点。固定 jitter 只用于避免点重叠，不改变数据。

输出为：

```text
results/problem_exploration/raw_decode_token_sequences/figures/
  temp0.6_without_occ12_embedding_cosine.png
  temp0.6_without_occ12_embedding_cosine.pdf
```

本任务是离线后处理，不涉及 vLLM server、prefix cache、并发或断点续跑。重复
执行会覆盖同名派生图，不会修改输入 CSV 或原始 decode 文件。

## 当前结论与边界

图支持的描述性结论是：三个比较组的 thinking cosine 都集中在较高区间，而
action cosine 的逐 case 离散程度更大，并存在少数低值 case。图不能证明 RoPE
与 sampling baseline 等价或更优，因为每个 arm 仍只有一次采样，箱线也不是
总体置信区间。

## 本轮修改文件

- `scripts/06_context_free_segment_cache/cross_occurrence_controller/raw_decode_token_sequences/plot_embedding_cosine.py`
- `results/problem_exploration/raw_decode_token_sequences/figures/temp0.6_without_occ12_embedding_cosine.png`
- `results/problem_exploration/raw_decode_token_sequences/figures/temp0.6_without_occ12_embedding_cosine.pdf`
- `results/problem_exploration/raw_decode_token_sequences/summary.md`
- `results/problem_exploration/raw_decode_token_sequences/source_manifest.csv`
- `agent_md/segmentia_raw_decode_visualization_development.md`

## 本轮验证

- 输入 CSV 共 36 行，三个 comparison 各 12 行且组内 filename 唯一。
- 三组 arm 映射与实验定义一致，无缺失值，全部 cosine 为有限值并位于
  `[-1, 1]`。
- 绘图脚本通过 `python -m py_compile` 和 `--help` 检查。
- 脚本成功生成非空 PNG 与 PDF；PNG 已目视检查，无图例遮挡、标签裁切或数据点
  越界。
- 按用户要求，图例和横轴仅使用 `RC1–RC2`、`RC1–RoPE`、`RC2–RoPE`
  标识比较组，不显示 seed 文字。
- 未启动 vLLM，未运行 decode 实验，未修改输入 CSV 和原始序列。

## 尚未实现

当前没有代码待办。若未来增加多 seed 数据，应重新设计统计图并加入按 case
配对的置信区间或预先定义的等价性检验，而不是沿用当前单次采样箱线作总体推断。
