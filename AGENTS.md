# Segmentia Agent Guide

本文档记录本仓库当前实际推进的研究线：`Segmentia`，即 context-free skill / segment KV cache 复用实验。后续 agent 在 `/home/wsh/openhands_code_research` 中工作时，应优先遵守本文档约束。

## 1. 当前工作范围

当前主线围绕 Segmentia 实验展开，主要涉及以下路径：

- `scripts/`：实验计划表、实验脚本、评估脚本、绘图脚本，以及 vLLM 启停脚本。Segmentia 06 脚本必须按研究内容组织在 `scripts/06_context_free_segment_cache/` 的子目录中。
- `results/`：实验输出、decode 结果、评估指标、图表和分析总结。
- `src/traces/`：实验使用的 trace、system prompt 和 tools 配置。trace 来自 Claude 实际使用 skills 的过程追踪，用来避免重新运行 agent 时引入额外系统问题，从而提高实验效率。
- `scripts/vllm_start.sh`、`scripts/vllm_stop.sh`：启动和停止实验用 vLLM server。
- `/home/wsh/vllm`：当前使用的本地 vLLM 修改版，负责 `context_segment_cache` 的实际注入、保存、RoPE 修正和 prefix-cache 行为。

除非用户明确要求，不要把未讨论、未使用的旧目录写进本文档，也不要主动整理这些目录。


## 2. 环境与运行前提

默认工作目录与 Python 环境：

```bash
cd /home/wsh/openhands_code_research
conda activate opencode
```

实验服务使用本地 vLLM：

```text
/home/wsh/vllm
```

默认模型与服务配置：

```text
model path: /mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B
served name: Qwen3
port: 8000
api key: EMPTY
```

thinking 语义相似度诊断默认使用本地 embedding 模型：

```text
embedding model path: /mnt/Large_Language_Model_Lab_1/模型/rag_models/BAAI-bge-base-en-v1.5
```

该模型用于本地计算 reasoning / thinking 的 embedding cosine。默认不联网下载模型、不自动安装依赖；若当前环境缺少 `sentence_transformers`，可优先用 `opencode` 环境已有的 `transformers + torch + numpy` 直接加载本地模型并做 pooling。若依赖或模型不可用，分析脚本应记录 `embedding_metric_status=unavailable`，并继续输出 BLEU、ROUGE-L、chrF、token Jaccard 以及 task-grounding / intent 指标。

实验运行边界：

- agent 负责写好代码、shell 脚本、结果目录结构和文档。
- 默认由用户亲自启动实验运行；agent 不主动启动长时间 vLLM 实验、不主动跑 overnight / diagnostic 全量实验。
- agent 可以做轻量静态验证，例如 `python -m py_compile`、`bash -n`、路径解析检查；这些验证不得启动 vLLM decode 实验。

### 2.1 Segmentia replay 与 prefix-cache 隔离边界

Segmentia 的 decode / diagnostic / attention probe replay 不是简单把所有 selected cases 放到同一个 vLLM server 里跑。凡是实验需要复现 trace 中多轮 skill 复用状态，必须遵守以下服务生命周期：

```text
for mode in [recompute, rope, ...]:
  for task in task_order:
    restart vLLM
    clear prefix cache
    run this task's cases in invocation_index order
```

也就是说，vLLM server 生命周期和 prefix-cache 隔离边界是：

```text
(mode, task)
```

不是：

```text
mode
```

更不是把所有 task 的 selected cases 混在同一个 server 生命周期里。不同 mode 必须重启；不同 task 也必须重启。task 内部必须按 trace 的真实轮次顺序 replay，通常按 `invocation_index` 从小到大排序。这样 prefix cache 只保留同一个 task 内前面轮次自然产生的前缀 KV，不会跨 task 污染。

`rope` mode 下的 skill KV 复用语义必须特别注意：

- 当前轮次只对当前 case 的 `target_start` / `target_end` 显式注入一次。
- 如果后续轮次的 prompt 共同前缀包含前面轮次的 skill span，后续轮次应通过 vLLM prefix cache 复用前面轮次已经注入后的 KV。
- 后续轮次不能把前面轮次的历史 skill span 再作为当前 `targets` 显式注入一遍。
- 当前轮次自己的 skill span 仍然通过 `context_segment_cache.targets` 注入，`mode=rope` 时只对 key 做 RoPE position correction，value 直接复用 offline context-free skill value。

因此，正确状态推进是：

```text
task 内第 10 次 invocation:
  replay 第 10 次 trace messages
  对第 10 次当前 target span 做 rope 注入
  prefix cache 留下第 10 次注入后的前缀 KV

task 内第 20 次 invocation:
  replay 第 20 次 trace messages
  第 10 次 skill span 若属于共同前缀，则从 prefix cache 继承已注入 KV
  只对第 20 次当前 target span 做新的 rope 注入
```

实现任何 replay wrapper 前，必须先复述并检查以上三点：

- 服务重启边界是否是 `(mode, task)`。
- task 内是否按 `invocation_index` 顺序 replay。
- 后续轮次是否通过 prefix cache 继承历史注入 KV，而不是对历史 skill span 重复显式注入。

## 3. Development 跟踪文档

当用户要求围绕某个目标开展开发、实验或分析时，必须在以下目录创建或更新对应的 development 文档：

```text
/home/wsh/openhands_code_research/agent_md/
```

development 文档用于记录当前探索与开发状态。文件名应能反映目标，例如：

```text
/home/wsh/openhands_code_research/agent_md/xxx_segmentia_development.md
```

development 文档至少需要明确：

- 总开发目标是什么。
- 目标被拆分为哪些具体阶段或小目标。
- 每个阶段或小目标当前实现到什么程度。
- 还有哪些内容尚未实现。
- 当前存在什么问题、风险、阻塞或需要后续验证的事项。
- 本轮修改了哪些文件，运行了哪些验证，验证结果是什么。

每个 development 文档必须包含“开发阶段总览”表，用于快速查看完整路线图。表格至少包含：

```text
阶段 | 名称 | 目标 | 当前进度 | 剩余
```

该总览表应放在文档前部，标题统一使用：

```text
## 开发阶段总览
```

每次开发、实验、修复或结果分析后，都必须同步更新对应 development 文档。更新时要删除过期状态，把文档刷新为当前状态；不要让旧结论、旧 TODO 或已经失效的风险继续留在文档中误导后续工作。

## 4. 结果目录边界

当前统一结果根目录为：

```text
/home/wsh/openhands_code_research/results/
```

`results/` 只用于存放总结性或可复查的实验产物，例如：

- 实验结构说明。
- 总结性图表。
- 总结性结论 `.md` 文件。
- 面向论文 motivation、finding、阶段报告的汇总材料。
- 少量支撑总结材料的轻量配置或说明文件。

结果目录必须按“研究阶段 -> 研究内容”组织，不允许用脚本编号（例如 `03`、`05`、`06`）作为结果阶段名。当前阶段还没有进入 method design，统一归入问题探究阶段：

```text
/home/wsh/openhands_code_research/results/problem_exploration/
```

阶段目录必须至少包含：

- `summary.md`：说明本阶段目标、各研究内容导航、阶段总论，以及下一阶段建议。
- `source_manifest.csv`：列出阶段级 summary、各研究内容 summary、关键 manifest 等入口文件的来源。
- 若干研究内容子目录。研究内容子目录命名必须表达研究问题或诊断目标，而不是脚本编号。

每个研究内容子目录至少应包含：

- `summary.md`：本研究的问题、目的、实验方法或代码逻辑、数据来源、数据分析、核心结论和限制。
- `figures/`：规范化后的总结性图。
- `tables/`：规范化后的总结性表格。
- `data/`：支撑 `summary.md` 的轻量汇总数据。
- `source_manifest.csv`：列出每个图、表、数据来自哪些真实 run artifacts。

实验结果分析应尽量图示化。除非数据规模太小、图会误导，或用户明确只要文字/表格，否则每个完成实验分析的研究内容至少应沉淀一张核心图到 `figures/`，并在 `summary.md` 中解释这张图支持什么结论、不能支持什么结论。表格用于支撑可复查数字，图用于展示主要现象、对照关系和 go/no-go 判断。新增图必须写入对应 `source_manifest.csv`，说明图来自哪些真实 run artifacts 或后处理表。

不要把某个研究内容再套一层与阶段同义的包目录。若当前阶段只有一个大方向，也仍然应把具体子研究直接放在阶段目录下，例如：

```text
results/problem_exploration/
  summary.md
  source_manifest.csv
  headline_semantic_action_gap/
  stability_systematic_vs_noise/
  value_repair_key_value_diagnosis/
```

不要把 checkpoint、大规模中间结果或逐 run 原始日志随意放入 `results/`。需要沉淀到 `results/` 的内容，应是从真实实验输出中整理出的总结性图表、结论文档或轻量汇总文件。

所有大体量输出必须放在非系统盘 `/mnt/Large_Language_Model_Lab_1` 下，例如 KV dump 的 `.pt` 文件。Segmentia 的 06 实验大体量 KV 已完成迁移；后续新增大 KV 默认写入 `SEGMENTIA_OUTPUT_DIR` 指向的外存目录，仓库内 `results/` 只保留 JSONL、CSV、JSON、PNG、Markdown、manifest 等轻量结果。

结果整理记录：

- 完成时间：

```text
2026-06-18
```

- 当前阶段：

```text
problem_exploration
```

- 当前阶段目录：

```text
results/problem_exploration/
```

- 子研究：

```text
headline_semantic_action_gap/
stability_systematic_vs_noise/
value_repair_key_value_diagnosis/
```

迁移记录：

- 完成时间：

```text
2026-06-18
```

- Segmentia 大体量数据保存路径：

```text
/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/06_context_free_segment_cache/
```

- 已迁移内容：

```text
results/06_context_free_segment_cache/offline_skill_kv/*.pt
results/06_context_free_segment_cache/cksim_kv/*.pt
results/06_context_free_segment_cache/repair_arms_kv/**/*.pt
```

- 迁移后代码默认路径：

```text
SEGMENTIA_OUTPUT_DIR=/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/06_context_free_segment_cache
```

后续 baseline 的大体量输出默认写入独立目录。例如 baseline 为 `Cacheslide` 时：

```text
/mnt/Large_Language_Model_Lab_1/wsh/Cacheslide/output/
```

## 5. 工作流程总原则

所有 agent 必须遵守以下原则：

> 先理解，再规划，再实现；每步都要验证，不跳过步骤。遇到阻塞时，回退到上一个阶段重新分析原因。

### 5.1 Idea 开发阶段

- **识别不确定性**：实现前必须识别 idea 中的不确定性和假设。对于不确定的部分，及时追问用户，不要自行假设答案。
- **分解目标**：将大目标拆分为可验证的小步骤，每一步都要有明确的 go/no-go 判据。
- **了解已有进展**：通过搜索、阅读现有代码和文档了解已有实现与结论，避免重复工作或与已有结论矛盾。
- **不要闷头做**：如果对方向、优先级或技术路径有疑问，先问清楚，不要自己猜测后直接实现。

### 5.2 代码实现阶段

- **阅读约束文件**：实现前必须阅读相关 `agent_md`、`AGENTS.md` 等约束文件，理解当前约定和上下文。
- **先确认再实施**：实现前先复述一遍逻辑，等用户确认后再开始实现。复述必须先讲清楚原理和为什么要这么做，再讲代码逻辑、数据流、状态边界、执行顺序、写入路径和 go/no-go 判据。代码逻辑必须细到可执行层级，而不是只列函数名或高层伪代码。对任何代码任务，都要说明：输入从哪里来、输出写到哪里、核心数据结构是什么、按什么 key 查找/聚合/去重、循环或调用顺序是什么、排序依据是什么、错误/重试/跳过如何处理、哪些状态会被保留或清空、哪些行为会改变已有文件或外部状态。若任务涉及服务、缓存、队列、并发、实验运行或批处理，还必须说明服务重启边界、缓存隔离/复用边界、并发顺序、断点续跑语义和失败恢复策略。复述要像解释给后续接手的人看。例如曾经在 `repeat-task-skill` 三层循环中误写成 `skill-task-repeat`，导致反复返工；这类问题必须通过实现前复述避免。
- **最小变更原则**：一次只实现一件事情，实现后立即验证。不要堆叠多个改动后再统一验证。
- **严格禁止垃圾代码**：不允许为了“看起来更稳”而堆叠冗余 fallback、无意义 wrapper、重复判断、宽泛 `try/except` 或不会被使用的抽象。
- **代码修改小而直接**：优先沿用现有项目结构和风格。每个新增函数、参数、文件或分支都应能说明其目的，并能对应到 development 文档中的具体阶段或小目标。

### 5.3 验证阶段

- **方向一致性检查**：验证时必须确认实现符合用户给出的方向与要求。如果加入了额外改动，即使是改进，也必须在对话中明确说明，不能默默加入。
- **每步验证**：每完成一个最小变更，立即验证其正确性和效果。不要积累多个未验证改动。
- **阻塞回退**：如果验证失败或遇到阻塞，不要继续向前堆功能。先回退到上一个阶段重新分析失败原因，必要时追问用户。

### 5.4 代码质量底线

必要的错误检查、数据校验和边界处理可以保留，但必须服务于真实风险或明确的调试需求。代码实现必须逻辑清晰、目标明确，围绕当前用户要求解决具体问题。

## 6. 科研导向原则

Segmentia 当前以计算机系统结构方向为导向。代码实现、实验设计和论文叙事都应服务于发现、解释并解决一个有价值的系统问题。

- 代码实现和理想设计之间存在 gap 是可以接受的。在系统领域，这种 gap 甚至可能暴露出有价值的问题。
- 系统结构研究的目标不是单纯提供一个“看起来不错的设计”，而是深挖一个能带来 insight 的问题，并围绕该问题给出清晰证据链。
- 在讨论、探究和分析阶段，不要因为某个方向成本高、代价大就回避它；高成本现象有时正是好的系统问题入口。
- 对于“有价值的问题”或“现象问题”，首要目标是探究背后原因，并尝试提供解决方案。如果解决过程中进一步引出新的系统问题，这是可接受甚至有价值的。
- 但问题链条必须能收敛。最多遵循“现象问题 / 有价值问题 -> 探究与解决 -> 引出新的系统问题”的路径，不要在探究过程中不断引出理论问题并无限深挖。
- 如果某个实现细节对最终效果没有显著影响，不需要为了和设计文档完全一致而强行修改。
- 如果某个实现细节对最终效果有显著影响，应当修复它；修复动机是提升效果或解释现象，而不是机械对齐设计文档。
- 设计文档和 `FINAL_PROPOSAL` 是设计参考，不是实现规范。实现可以更简单，也可以更复杂，前提是能解释清楚并能跑通验证。

## 7. 文档撰写质量

文档要服务于后续复查、接续开发和论文写作，不能只服务于当下记忆。

- 遵循金字塔原理：先给结论和结构，再展开细节。
- 不要突然引入前文没有出现过的概念或名词。概念第一次出现时，需要在就近位置解释清楚。
- 当文档使用自动指标、分类标签、parser、judge 或启发式规则时，必须在首次使用附近说明它的来源、判定逻辑、可靠性边界和可能误判类型。不要把脚本判定写得像人工确认的事实。例如 `intent_match=True` 只能表示当前 intent parser 未检测到差异，除非已经人工复核，否则不能直接写成“下一步意图一致”。
- 不需要在文档开头集中堆叠所有名词定义；更重要的是在叙述中把事情讲清楚。
- 不要默认用户以后仍然记得当前实验细节。关键实验设置、路径、数据来源、判断标准和限制都要写清楚。
- 总结结论时要区分“已验证事实”“当前推测”和“尚未验证的下一步”。
