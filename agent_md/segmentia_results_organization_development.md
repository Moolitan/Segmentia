# Segmentia Results Organization Development

## 总开发目标

维护 `results/problem_exploration/` 作为 Segmentia 问题探究阶段的可复查入口，使阶段 summary 负责导航和证据链，各子研究 summary 负责实验细节。

## 开发阶段总览

| 阶段 | 名称 | 目标 | 当前进度 | 剩余 |
|---|---|---|---|---|
| 1 | 阶段目录 | 使用研究阶段和研究问题命名结果目录。 | 已完成。 | 无。 |
| 2 | 子研究结构 | 每个子研究具备 summary、figures、tables、data 和 manifest。 | 已完成。 | 新研究持续遵守。 |
| 3 | 大文件迁移 | KV `.pt` 写入外存，仓库只保留轻量产物。 | 已完成。 | 新实验持续遵守。 |
| 4 | 阶段入口 | 阶段 summary 只保留导航、证据链、结论边界和下一步。 | 已完成本轮重写。 | 随新结果更新。 |

## 当前目录

```text
results/problem_exploration/
  headline_semantic_action_gap/
  stability_systematic_vs_noise/
  value_repair_key_value_diagnosis/
  logprob_margin_diagnostic/
  thinking_to_action_divergence/
  attention_matrix_visualization/
  manifests/
  summary.md
  source_manifest.csv
```

## 文档职责

阶段 `summary.md`：

```text
阶段目标
研究导航
跨实验的证据链
已验证事实 / 当前推测 / 尚未解决
下一阶段顺序
```

子研究 `summary.md`：

```text
具体研究问题
实验设置与代码逻辑
数据来源
指标定义
结果和图表
限制
```

`source_manifest.csv` 记录阶段和子研究入口、图表与数据的来源，不替代研究结论。

## 路径边界

轻量结果根目录：

```text
/home/wsh/openhands_code_research/results/problem_exploration/
```

大体量 KV：

```text
/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/06_context_free_segment_cache/
```

脚本目录 `scripts/06_context_free_segment_cache/` 是代码位置，不作为结果阶段名。

## 本轮修改

- 删除没有当前实验结果的 `cross_occurrence_controller/` 结果目录及其阶段 manifest 引用。
- 方法设计入口改为直接指向 scripts 下的当前设计文档。
- 更新阶段下一步为 FV existence、interaction mediation、block restoration 和 online policy。

## 验证

- `results/problem_exploration/cross_occurrence_controller/` 已不存在。
- 阶段 `source_manifest.csv` 中所有保留路径均存在。
- 阶段 summary 的方法设计链接指向现有 `LiangFunctionVectorDesign.md`。
- 其他问题探究子研究的原始 JSONL、CSV 和图未修改。
