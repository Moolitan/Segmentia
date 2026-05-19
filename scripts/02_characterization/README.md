# scripts/characterization

工作负载特征化脚本集，用于 SkillCache 研究的第一阶段（第1-2周）。

详细计划见 [`docs/week1-2-characterization-plan.md`](../../docs/week1-2-characterization-plan.md)。

---

## 前置准备

### 数据

评估结果目录已在本地：

```
eval_outputs/
└── eth-sri__SWT-bench_Verified_bm25_27k_zsp-test/
    └── litellm_proxy/
        └── jade-spark-2862_sdk_cfe52af_maxiter_500_N_SWT-litellm_proxy-jade-spark-2862/
            ├── conversations/   ← 433 个对话，每个为 .tar.gz
            └── output.jsonl     ← 实例元数据（issue 描述、通过率等）
```

以下用 `$EVAL_DIR` 代指该目录：

```bash

```

### 依赖

```
tiktoken       # simulate_skill_activation.py 需要（token 计数）
matplotlib     # visualize_skill_activation.py 需要（绘图）
numpy          # visualize_skill_activation.py 需要（数值计算）
```

其余脚本仅使用 Python 标准库。

---

## 脚本说明

### 1. `check_skills_feasibility.py` — 可行性检查

**用途**：检查现有 SWE-Bench traces 中 `activated_skills` 字段的非空率，决定是直接使用现有数据（路径A）还是需要构建新工作负载（路径B）。

**建议最先运行此脚本。**

```bash
# 快速抽查（前 10 个对话）
python scripts/characterization/check_skills_feasibility.py $EVAL_DIR

# 全量检查（433 个对话）
python scripts/characterization/check_skills_feasibility.py $EVAL_DIR --sample 0
```

**输出示例**：
```
检查 433 个对话...
  [  1] ○  django__django-1234   msg= 12  skills_activated=  0
  ...
MessageEvent 总数        : 433
activated_skills 非空数  : 0
非空率                   : 0.0%

🔧 路径B — activated_skills 基本为空，需构建 Skill-rich 工作负载。
```

**结论判断**：
- 非空率 > 20% → **路径A**，直接进行离线分析
- 非空率 ≤ 20% → **路径B**，需先构建 Skill-rich 工作负载（见下文）

---

### 2. `scan_issue_keywords.py` — Issue 关键词扫描

**用途**：扫描 SWE-Bench issue 描述中的高频词和短语，辅助设计 Skill 的 `KeywordTrigger`，确保 Skills 在真实任务中能被有效触发。

**在路径B时，构建 Skills 之前运行此脚本。**

```bash
# 基本用法
python scripts/characterization/scan_issue_keywords.py $EVAL_DIR/output.jsonl

# 显示更多词（Top 120）
python scripts/characterization/scan_issue_keywords.py $EVAL_DIR/output.jsonl --top 120

# 调整过滤阈值（只看至少出现 10 个 issue 的词）
python scripts/characterization/scan_issue_keywords.py $EVAL_DIR/output.jsonl --min-count 10
```

**输出三部分**：
1. 高频单词 Top N（按覆盖 issue 数排序）
2. 高频 2-gram 短语 Top N（识别更具体的触发场景）
3. 推荐 Skill 场景分类（预设10个软件工程场景的关键词覆盖率）

**根据输出设计 Skills**：选择覆盖率高（>30% issue）且语义明确的词作为 `KeywordTrigger`，每个 Skill 配 2-3 个 keywords，目标构建 10-20 个 Skills。

---

### 3. `simulate_skill_activation.py` — 离线 Skill 激活模拟

**用途**：对每个 SWE-Bench 实例，基于 keyword trigger 匹配模拟会触发哪些 Skills，计算 `<EXTRA_INFO>` 注入 token 开销，产出跨请求 KV Cache 低效特征化所需的全部数据。

**无需 LLM / GPU**，直接调用 SDK 的 `Skill.match_trigger()` 离线计算。

```bash
# 快速测试（前 10 个实例）
python scripts/02_characterization/simulate_skill_activation.py \
    $EVAL_DIR/output.jsonl --skills-dir skills/ --sample 10

# 全量运行（433 个实例）
python scripts/02_characterization/simulate_skill_activation.py \
    $EVAL_DIR/output.jsonl --skills-dir skills/ \
    --output results/characterization/skill_activation_stats.json
```

**参数**：

| 参数 | 说明 | 默认值 |
|---|---|---|
| `output_jsonl`（位置参数） | output.jsonl 文件路径 | — |
| `--skills-dir DIR` | Skills 目录 | `skills/` |
| `--output PATH` | 输出 JSON 路径 | `results/skill_activation_stats.json` |
| `--sample N` | 只处理前 N 个实例（0 = 全部） | `0` |

**输出** `results/skill_activation_stats.json`：

```
{
  "metadata": { total_instances, total_skills, timestamp, tokenizer, ... },
  "per_instance": [
    {
      "instance_id": "django__django-11163",
      "activated_skills": ["debugging-wizard", "django-expert"],
      "trigger_details": [{"skill": "...", "matched_keyword": "..."}],
      "num_skills_activated": 2,
      "skill_set_hash": "a1b2c3d4",       // sorted skill names 的 MD5
      "extra_info_tokens": 2847,           // 注入的 <EXTRA_INFO> 总 token
      "issue_text_tokens": 512
    }, ...
  ],
  "aggregate": {
    "skill_frequency":              每个 skill 的激活次数和百分比,
    "activation_count_distribution": 每实例激活 skill 数分布,
    "unique_skill_combinations":     不同 skill 组合总数,
    "skill_set_frequency":           各 skill 组合出现次数排序,
    "co_occurrence_matrix":          skill 两两共现次数,
    "token_stats":                   <EXTRA_INFO> token 统计（mean/median/std/min/max）,
    "diversity_metrics": {
      "jaccard_mean":        跨实例 Jaccard 距离均值,
      "jaccard_std":         标准差,
      "normalized_entropy":  skill 组合分布的归一化熵
    }
  }
}
```

**下游 Task 对接**：

| Task | 使用的字段 |
|---|---|
| Task 1 (Skill 激活模式) | `per_instance.activated_skills`, `aggregate.skill_set_frequency` |
| Task 2 (Token 分布) | `per_instance.extra_info_tokens`, `aggregate.token_stats` |
| Task 3 (EXTRA_INFO 分析) | `per_instance.trigger_details`, `per_instance.extra_info_tokens` |
| Task 5 (KV Cache 模拟) | `per_instance.skill_set_hash`（同 hash = prefix 可共享） |
| Task 7 (上界计算) | `aggregate.skill_set_frequency`（贪心排序最大化复用） |

---

### 4. `visualize_skill_activation.py` — 结果可视化

**用途**：读取 `simulate_skill_activation.py` 的输出 JSON，生成 7 张论文级图表。

```bash
# 默认读取 results/characterization/skill_activation_stats.json
python scripts/02_characterization/visualize_skill_activation.py

# 指定输入和输出目录
python scripts/02_characterization/visualize_skill_activation.py \
    --input results/characterization/skill_activation_stats.json \
    --outdir results/characterization/figures
```

**参数**：

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--input PATH` | 输入 JSON 路径 | `results/characterization/skill_activation_stats.json` |
| `--outdir DIR` | 图表输出目录 | 输入文件同目录下的 `figures/` |

**生成图表**（PDF + PNG 双格式）：

| 文件 | 内容 | 论文用途 |
|---|---|---|
| `fig1_skill_frequency` | Skill 激活频率条形图 | 展示各 Skill 覆盖率差异 |
| `fig2_activation_distribution` | 每实例激活 Skill 数分布 | 量化动态注入复杂度 |
| `fig3_token_distribution` | EXTRA_INFO vs Issue token 分布（直方图+箱线图） | 量化 Skill 注入的 token 开销 |
| `fig4_co_occurrence` | Skill 共现热力图 | 揭示 Skill 间关联，指导规范排序 |
| `fig5_top_combinations` | Top 15 最常见 Skill 组合 | 展示组合多样性 |
| `fig6_jaccard_cdf` | 跨请求 Jaccard 距离 CDF | **核心图**：量化跨请求 prefix 差异 |
| `fig7_dynamic_ratio` | Dynamic Ratio 分布 | Skill 注入 token 占比 |

---

## 推荐执行顺序

```
check_skills_feasibility.py          ✅ 已完成（结论：路径B）
        │
        └── 路径B（非空率 ≤ 20%）
                │
                ├── scan_issue_keywords.py   ✅ 已完成（设计 Skills 的依据）
                ├── 创建 Skills（skills/）    ✅ 已完成（12 个 Skills）
                ├── simulate_skill_activation.py  ✅ 已完成（离线模拟替代重新评估）
                ├── visualize_skill_activation.py ✅ 已完成（7 张论文级图表）
                ├── run_swebench_with_skills.py  ⬜ 待运行（LLM 实际运行补充验证）
                └── 后续：Task 1-8 特征化分析 & KV Cache 模拟
```

---

### 5. `run_swebench_with_skills.py` — LLM 实际运行（补充实验）

**用途**：使用 LLM 实际运行 SWE-Bench 实例（加载 Skills），捕获每 turn 的 `MessageEvent.activated_skills` 数据。补充离线模拟的不足，验证：

1. **Per-turn 激活动态**：同一会话内，后续 turn 是否触发新 skill
2. **LLM 与 `<EXTRA_INFO>` 的交互行为**：模型是否参考注入内容
3. **实际 token 开销**：真实推理中的 token 消耗

**背景概念**：

- **SWE-Bench 实例**：每个实例是一个真实开源项目的 bug（来自 GitHub issue/PR），包含 bug 描述和期望的修复。`output.jsonl` 每行对应一个实例
- **workspace**：Agent 的工作目录。脚本在此目录下执行命令、编辑文件。Skills 放在 `workspace/.agents/skills/` 下，由 SDK 自动加载（和 `scripts/01/01_quick_start.py` 的模式一致）

**前置准备**：

1. vLLM 服务已启动（默认 `localhost:8000`）
2. 创建 workspace 目录，将 Skills 拷贝到 `{workspace}/.agents/skills/` 下：

```bash
# 示例：使用 workspace/02
mkdir -p workspace/02/.agents/skills
cp -r skills/* workspace/02/.agents/skills/
```

**用法**：

```bash
# 运行单个实例
python scripts/02_characterization/run_swebench_with_skills.py \
    $EVAL_DIR/output.jsonl \
    --instance-id django__django-11163 \
    --workspace /home/wsh/openhands_code_research/workspace/02

# 运行多个实例（逗号分隔）
python scripts/02_characterization/run_swebench_with_skills.py \
    $EVAL_DIR/output.jsonl \
    --instance-id django__django-11163,sympy__sympy-18199 \
    --workspace /home/wsh/openhands_code_research/workspace/02

# 随机抽样 10 个实例
python scripts/02_characterization/run_swebench_with_skills.py \
    $EVAL_DIR/output.jsonl \
    --sample 25 \
    --max-iterations 20 \
    --workspace /home/wsh/openhands_code_research/workspace/02

# 运行全部 433 个实例（不指定 --instance-id 和 --sample）
python scripts/02_characterization/run_swebench_with_skills.py \
    $EVAL_DIR/output.jsonl \
    --workspace /home/wsh/openhands_code_research/workspace/02
```

**参数**：

| 参数 | 说明 | 默认值 |
|---|---|---|
| `output_jsonl`（位置参数） | output.jsonl 文件路径 | — |
| `--instance-id IDS` | 要运行的实例 ID（逗号分隔） | 不指定则全部运行 |
| `--sample N` | 随机抽样 N 个实例 | `0`（不抽样） |
| `--workspace DIR` | Agent 工作目录 | **必填** |
| `--skills-dir DIR` | Skills 目录 | `{workspace}/.agents/skills/` |
| `--output DIR` | 输出目录 | `results/characterization/swebench_skill_traces` |
| `--max-iterations N` | 每实例最大迭代数 | `30` |
| `--model NAME` | LLM 模型名 | `openai/Qwen2.5` |
| `--base-url URL` | LLM API base URL | `http://localhost:$VLLM_PORT/v1` |
| `--api-key KEY` | LLM API key | `$VLLM_API_KEY` 或 `EMPTY` |
| `--seed N` | 随机种子（用于 `--sample`） | `42` |

**输出**：

每个实例生成一个 `<instance_id>.json` 文件，包含完整事件 trace：

```json
{
  "instance_id": "django__django-11163",
  "total_events": 45,
  "total_turns": 8,
  "elapsed_seconds": 120.3,
  "skill_trace": [
    {"turn": 1, "source": "user", "activated_skills": ["django-expert", "debugging-wizard"], "has_extended_content": true}
  ],
  "all_activated_skills": ["debugging-wizard", "django-expert"],
  "events": [...]
}
```

另外生成 `_summary.json` 汇总所有实例的 skill 激活 trace。

**注意**：
- SWE-Bench 标准流程只有一次 `send_message()`（初始指令），因此 `activated_skills` 主要出现在第一个 MessageEvent
- SDK 会跳过已激活的 skills（`skip_skill_names`），同一会话中不会重复注入
- 建议先小规模测试（`--sample 3 --max-iterations 10`），确认流程正确后再扩大
