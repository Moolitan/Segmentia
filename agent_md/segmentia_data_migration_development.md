# Segmentia 数据迁移 Development

## 开发阶段总览

| 阶段 | 名称 | 目标 | 当前进度 | 剩余 |
|---|---|---|---|---|
| 1 | 识别大体量数据 | 找到 06 实验中应迁移出仓库的 `.pt` KV 文件 | 已完成：共 104 个 `.pt`，位于 `offline_skill_kv/`、`cksim_kv/`、`repair_arms_kv/` | 无 |
| 2 | 迁移数据 | 将大体量 KV 复制到非系统盘，并从仓库结果目录删除 `.pt` | 已完成：目标路径下 104 个 `.pt`，仓库 06 结果目录内 `.pt` 数量为 0 | 无 |
| 3 | 修改默认路径 | 让后续 06 实验默认从外存读写大 KV，轻量结果继续留在仓库 | 已完成：更新 `config.py` 和 shell wrapper | 后续新脚本需要沿用 `SEGMENTIA_OUTPUT_DIR` |
| 4 | 更新文档与验证 | 更新 `AGENTS.md` 与本 development 文档，记录迁移结果 | 已完成 | 无 |

## 目标

完成 `AGENTS.md` 第 4 节要求的数据迁移：Segmentia 的大体量 KV `.pt` 文件不再保存在 `/home/wsh/openhands_code_research/results/` 下，而是迁移到非系统盘：

```text
/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/06_context_free_segment_cache/
```

迁移完成时，仓库内 `results/06_context_free_segment_cache/` 只保留轻量可复查结果，例如 JSONL、CSV、JSON、PNG、Markdown 和 manifest。后续结果目录整理已将这些轻量结果移动到：

```text
results/problem_exploration/
```

## 实施记录

迁移范围：

```text
results/06_context_free_segment_cache/offline_skill_kv/*.pt
results/06_context_free_segment_cache/cksim_kv/*.pt
results/06_context_free_segment_cache/repair_arms_kv/**/*.pt
```

迁移后默认大体量输出根目录：

```text
SEGMENTIA_OUTPUT_DIR=/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/06_context_free_segment_cache
```

代码路径调整：

- `scripts/06_context_free_segment_cache/module/config.py`：新增 `SEGMENTIA_OUTPUT_DIR`，并将 `DEFAULT_KV_DIR`、`DEFAULT_CKSIM_KV_DIR`、`DEFAULT_REPAIR_KV_DIR` 指向外存。
- `scripts/06_context_free_segment_cache/headline_semantic_action_gap/run_decode_compare.sh`：`KV_DIR` 默认指向外存 `offline_skill_kv/`，`OUTPUT` 仍写仓库轻量结果目录。
- `scripts/06_context_free_segment_cache/value_repair_key_value_diagnosis/run_value_repair_compare.sh`：`SKILL_KV_DIR`、`CKSIM_KV_DIR`、`REPAIR_KV_DIR` 默认指向外存。
- `scripts/06_context_free_segment_cache/pipeline/run_all_overnight.sh`：评估阶段默认从外存 `cksim_kv/` 读取 CKSim KV。
- `scripts/06_context_free_segment_cache/value_repair_key_value_diagnosis/build_repair_arms_kv.py`：repair arms 默认输出目录改为 `config.py` 中的外存路径。
- `scripts/06_context_free_segment_cache/README.md`：更新命令示例，明确大 KV 使用 `SEGMENTIA_OUTPUT_DIR`。
- `AGENTS.md`：更新第 4 节迁移记录。

## 验证结果

迁移前后 `.pt` 数量一致：

```text
source before cleanup: 104
target after copy:     104
source after cleanup:  0
```

迁移后大小：

```text
/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/06_context_free_segment_cache: 23G
/home/wsh/openhands_code_research/results/06_context_free_segment_cache: 4.3M at migration time; later reorganized into results/problem_exploration
```

迁移后的 06 结果目录只保留轻量结果；仓库 `results/` 中仍有其他历史目录占用空间，例如 `results/05_context_segment_agent_kv`，不属于本次 Segmentia 06 数据迁移范围。

## 风险与后续注意事项

- 旧命令如果显式传入仓库内 `--kv-dir results/06_context_free_segment_cache/...`，仍会覆盖默认外存路径；后续运行时应优先使用默认值或显式传入 `$SEGMENTIA_OUTPUT_DIR/...`。
- 后续新增脚本如果产生大 `.pt`，必须使用 `SEGMENTIA_OUTPUT_DIR`，不要重新写入仓库 `results/`。
- 当前迁移保留了仓库内 manifest 文件，便于快速复查；外存目标目录也同步保留了 manifest。
