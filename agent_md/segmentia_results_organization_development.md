# Segmentia Results Organization Development

## 开发阶段总览

| 阶段 | 名称 | 目标 | 当前进度 | 剩余 |
|---|---|---|---|---|
| 1 | 明确阶段命名 | 将当前结果从脚本编号转为研究阶段命名 | 已完成：当前阶段命名为 `problem_exploration` | 无 |
| 2 | 整理 Segmentia 结果 | 将旧 `results/06_context_free_segment_cache` 改为语义化阶段目录 | 已完成：迁移到 `results/problem_exploration/` | 无 |
| 3 | 删除非当前主线旧结果 | 清理 `03_*` 和 `05_context_segment_agent_kv` 旧结果目录 | 已完成 | 无 |
| 4 | 更新代码与文档 | 更新默认结果路径、README、AGENTS 和索引文档 | 已完成 | 后续新增脚本必须沿用新路径 |
| 5 | 补齐研究内容规范 | 每个研究内容补 `summary.md` 和 `source_manifest.csv`，阶段 summary 补目标/导航/总论/建议 | 已完成 | 无 |
| 6 | 验证 | 检查旧路径、目录大小、脚本语法与 Python 编译 | 已完成 | 无 |

## 目标

整理 `results/`，不再用 `06`、`05`、`03` 这类脚本编号表达研究阶段。当前 Segmentia 尚未进入设计阶段，因此归入问题探究阶段：

```text
results/problem_exploration/
```

## 实施记录

旧结果目录：

```text
results/06_context_free_segment_cache/
```

新阶段目录：

```text
results/problem_exploration/
```

子研究结构：

```text
headline_semantic_action_gap/
stability_systematic_vs_noise/
value_repair_key_value_diagnosis/
manifests/
```

删除的旧结果目录：

```text
results/03_14B_anthropic/
results/03_14B_anthropic_3/
results/03_14B_anthropic_sglang/
results/05_context_segment_agent_kv/
```

这些目录不是当前 Segmentia problem exploration 的结果，且用户已确认可以删除。

## 代码与文档更新

- `scripts/06_context_free_segment_cache/config.py`：`RESULTS_DIR` 改为 `results/problem_exploration`。
- `scripts/06_context_free_segment_cache/run_decode_compare.sh`：默认 headline decode 输出改到 `headline_semantic_action_gap/data/`。
- `scripts/06_context_free_segment_cache/run_value_repair_compare.sh`：默认 value-repair decode 输出和评估命令改到 `value_repair_key_value_diagnosis/`。
- `scripts/06_context_free_segment_cache/run_all_overnight.sh`：三类子研究的 decode、evaluate、plot 路径改到对应子目录。
- `scripts/06_context_free_segment_cache/plot_action_fidelity.py`、`plot_metrics.py`、`plot_report_figures.py`：默认读写路径改到结构化阶段目录。
- `scripts/06_context_free_segment_cache/README.md`：更新轻量结果路径示例。
- `AGENTS.md`：记录当前阶段目录为 `problem_exploration`。
- `results/problem_exploration/summary.md`：阶段级入口，已融合原 `scripts/06_context_free_segment_cache/ANALYSIS_REPORT.md` 的完整分析内容。
- `results/problem_exploration/source_manifest.csv`：记录所有轻量产物来源。
- `results/problem_exploration/headline_semantic_action_gap/summary.md` 与 `source_manifest.csv`：补齐 semantic/action gap 子研究入口。
- `results/problem_exploration/stability_systematic_vs_noise/summary.md` 与 `source_manifest.csv`：补齐 systematic-vs-noise 子研究入口。
- `results/problem_exploration/value_repair_key_value_diagnosis/summary.md` 与 `source_manifest.csv`：补齐 key/value diagnosis 子研究入口。
- `results/problem_exploration/analysis_summary.md`：已删除。该文件是早期单次实验分析，已被当前阶段 summary 和三个子研究 summary 取代。
- `results/problem_exploration/decode_texts.txt`：已移动到 `headline_semantic_action_gap/data/decode_texts.txt`。
- `scripts/06_context_free_segment_cache/ANALYSIS_REPORT.md`：已删除；不在结果包下保留单独 `analysis_report.md`，避免违反 `AGENTS.md` 中每个研究内容以 `summary.md` 作为主入口的组织方式。

## 验证结果

目录大小：

```text
results: 4.4M
results/problem_exploration: 4.4M
```

旧结果目录删除验证：

```text
results/06_context_free_segment_cache: removed
results/05_context_segment_agent_kv: removed
results/03_14B_anthropic: removed
results/03_14B_anthropic_3: removed
results/03_14B_anthropic_sglang: removed
```

文件数量验证：

```text
lightweight files in problem_exploration: 34
.pt files in repo result package: 0
.pt files in external Segmentia output: 104
```

默认路径验证：

```text
RESULTS_DIR=/home/wsh/openhands_code_research/results/problem_exploration
DEFAULT_OUTPUT_JSONL=/home/wsh/openhands_code_research/results/problem_exploration/headline_semantic_action_gap/data/decode_outputs.jsonl
DEFAULT_METRICS_CSV=/home/wsh/openhands_code_research/results/problem_exploration/headline_semantic_action_gap/tables/headline_metrics_rows.csv
DEFAULT_KV_DIR=/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/06_context_free_segment_cache/offline_skill_kv
```

语法检查：

```text
python -m py_compile ... : passed
bash -n run_decode_compare.sh run_value_repair_compare.sh run_all_overnight.sh : passed
```

## 后续注意事项

- 后续不要再把新结果写回 `results/06_context_free_segment_cache/`。
- 脚本目录名仍为 `scripts/06_context_free_segment_cache/`，这是代码位置，不再等同于结果阶段名。
- 大体量 KV `.pt` 仍保存在 `/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/06_context_free_segment_cache/`，不随轻量结果目录整理移动。
