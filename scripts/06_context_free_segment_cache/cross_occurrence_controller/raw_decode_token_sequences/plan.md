# 2026-07-02: Thinking 语义一致但 Action 分叉的机制探索计划

## 背景

三组实验（复用 occ1 / 不复用 occ1 / 不复用 occ1&2）一致表明：
- Thinking 语义高度一致（cosine mean 0.93~0.94）
- Action 行为类型匹配率仅 47%~54%，且与前面有没有 recompute 无关

核心问题：分歧不是"理解错了"，而是在决策边界上被位置编码的微小数值差异推到了不同方向。需要定位这个机制。

## 分析方法（按推荐优先级排序）

### 方法 1: Token-level logit divergence trajectory（最推荐）

逐 token 比较 recompute 和 rope 的输出 logit 分布，画出 KL divergence 随 token position 的变化曲线。

**能回答的问题**：
- 分歧是在 thinking 阶段就逐渐累积的，还是到 `</think>` 边界突然跳变的？
- Action 的第一个 token（决定是输出文本还是 `<tool_call>`）的 top-k 概率差多少？是"几乎 50/50 被推了一下"还是"明确翻转"？

**实现方式**：在 decode 时同时记录 recompute 和 rope 两条路径每一步的 top-k logprobs（vLLM 的 `SamplingParams.logprobs` 已经支持），不需要改 vLLM 内部代码。

### 方法 2: Branch point analysis（最直接）

找到 recompute 和 rope 的 action 序列第一个分叉 token，分析那个位置两边的 top-k 概率分布。

**能回答的问题**：
- 分叉是发生在 `<tool_call>` vs 纯文本的第一个字符？还是更早（比如 thinking 末尾的某个推理分支）？
- 分叉点的 top-1 和 top-2 概率差距有多大？如果差距 < 5%，说明是决策边界效应（position 扰动足以翻转）；如果差距大，说明是深层表征已经被改变。

**实现方式**：对比两条路径的 token 序列，定位第一个不同 token 的位置，提取该位置的 logprobs。

### 方法 3: Attention heatmap on reused segments（机制层面最有解释力）

在 action 的前几个 token 的 decode 过程中，对比 recompute 和 rope 对被复用的 KV cache segment 的 attention 权重分布。

**能回答的问题**：
- Rope 偏移后，模型对复用 segment 的 attention 分布是否发生了系统性偏移？
- 是特定的 attention head 受影响更大，还是全局性的？
- 是浅层还是深层受影响更大？

**实现方式**：需要在 vLLM 的 attention 计算中加 hook 保存 attention weights。改动量中等，但需要注意 PagedAttention 的 block 结构。

### 方法 4: Hidden state cosine similarity per layer（定位哪一层开始分叉）

在 `</think>` token 和 action 第一个 token 的位置，逐层比较 recompute 和 rope 的 hidden state。

**能回答的问题**：
- 是从哪一层开始，两条路径的表征出现显著分歧？
- 浅层（position-sensitive）还是深层（semantic）受影响更大？

**实现方式**：在 vLLM 的 model forward 中逐层加 hook 提取 hidden state，改动量中等。

### 方法 5: Activation patching（最重，但最有因果性）

把 recompute 的某些层的 hidden state 替换进 rope 的 forward pass，看 action 是否"修复"。

**能回答的问题**：因果性地定位到底是哪些 layer × position 导致了 action 分叉。

**实现方式**：需要自定义 forward pass，改动量最大。

## 执行策略

先做方法 1 + 2，不需要改 vLLM 代码，只需要在 decode 时多记录 logprobs。能先回答最核心的问题：分歧是渐进累积的还是突变的？分叉点的决策边界有多窄？

如果 1+2 的结论是"分叉点的 top-1 和 top-2 概率差距非常小（< 5%）"，那基本可以下结论：这是位置编码扰动在决策边界附近的放大效应，属于模型本身对 tool-use 决策的不确定性。

如果差距大，才需要做方法 3（attention heatmap）来定位具体是哪些 attention head 被 rope 偏移破坏了。

---

# 2026-07-03：因果关系的重新校正 + 温度采样重跑设计

## 背景：一个可能被搞反了的因果关系

在此之前我们一直纠结于一个看似矛盾的现象：**thinking 语义高度相似（cosine 0.93），但 action 行为类型只有 47%~54% 匹配**。当时的隐含假设是：既然 thinking 都相似了，action 却不同，那一定有某个**单独的机制**在 `</think>` 边界附近专门把 action 决策搞坏了——于是去找"位置编码专门破坏 action 那一步"。

方法 1+2（temp=0 贪心，见上文与 SUMMARY.md）的结果让我们重新校正这个因果：

- recompute 与 rope 的输出**在 thinking 的头几十个 token 内（first_div_idx = 4~36）就已经表层分岔**，全部落在 `</think>` 之前。
- action 只是这条早已分岔的轨迹的自然延续。**action 不一样是"果"，不是被单独搞坏的"因"**；它继承自 thinking 早期的表层分岔，不需要一个专门作用于 action 的机制来解释。

关键澄清（避免把结论走偏）：**"thinking 语义相似"和"thinking 表层早分岔"两件事同时成立、不矛盾**。token 级分岔量的是"这一步选哪个词"（表层，欠定、脆弱）；语义 cosine 量的是"整体在说什么事"（有强吸引子、鲁棒）。所以真正的含义是：**驱动 action 分歧的不是语义（语义好好的），而是表层选择本身的脆弱性**。

## 由此逼出的真正问题：rope 到底有没有让结果"变差"

如果 action 差异只是表层脆弱性的下游，那有个尖锐推论：**即使两次纯 recompute（在有随机性的解码下），action 也会不一致**。那么 47%~54% 这个"不一致率"可能根本不是 rope 复用的锅，而是**模型自身解码方差的基线**。

因此，问题应从"为什么相似的 thinking 会给出不同的 action"（伪悖论）改为：

> **rope 复用带来的输出方差，有没有超过模型自身"运行间方差"的基线？**

- 若**没超过** → rope 复用不劣于模型固有方差，复用是"安全"的，之前追的那个"因"并不存在。这对 SegKV 是利好。
- 若**显著超过** → rope 确实额外损害了输出。结合方法 1+2，元凶应是那批 rope 实质性改变了分布的 case（SUMMARY.md 里 5/12 的 distribution-shift），而非近似平局翻转的那批。这时才需要 repair，且 repair 目标明确：只救那批深层表征。

## 实验设计：temp=0.6 采样档 + recompute 自基线

### 为什么换采样参数（不再贪心）
两个原因：
1. Qwen3 官方建议 thinking 模式**不要用贪心**（易性能退化 / 无限重复）。采用其 `generation_config.json` 默认：**Temperature=0.6, TopP=0.95, TopK=20, MinP=0**。
2. 自基线**必须有随机性**：temp=0 的 recompute 是确定性的（已被 6/18 前那次全 0 结果实证——两条 recompute-等价轨迹逐 token 完全一致），两次恒等 → 基线恒为 0，无意义。

### seed 策略（两条腿都要）
- **同 seed 的 rope vs recompute** → 隔离 KV 纯效应（RNG 相同，差异只来自 KV）。
- **不同 seed 的 recompute vs recompute** → 模型固有方差基线。
- 两者夹出结论：rope 的分歧是否真的高于固有方差。

### 首批运行范围（occ 只跑 (3,)）
temp=0.6 下三次采集，**换 mode 必须重启 vLLM 并清 prefix cache**：

| run 名 | mode | seed | 用途 |
|---|---|---|---|
| `recompute_run1` | recompute | 1111 | 基准 A（与 rope 同 seed） |
| `recompute_run2` | recompute | 2222 | 基准 B（与 A 不同 seed，测固有方差） |
| `rope` | rope | 1111 | 复用臂（与 run1 同 seed，测 KV 纯效应） |

对照对：`rope vs recompute_run1`（KV 效应）、`recompute_run1 vs recompute_run2`（固有方差基线）。

### 结果目录：脚本自动分层，不覆盖 temp=0 旧结果
两个 `.sh` 现在**自动**按 `温度档 / 复用条件 / run` 拼输出路径（无需手写死标签）。三个旋钮：
- `TEMP_TAG`（默认 `temp${TEMPERATURE}`，如 `temp0.6` / `temp0.0`）——温度档。
- `RUN_LABEL`（默认 `without_occ12`）——复用条件：`full_reuse` / `without_occ1` / `without_occ12`（occ=(3,) 即历史的 `without_occ12`）。
- `RUN_NAME`——同 mode 多次采样运行分隔：`recompute_run1` / `recompute_run2` / `rope`。

旧的贪心默认（temp=0）现在自动落到 `temp0.0/` 子目录，历史结果原地保留、不被覆盖。自动布局：
```
.../logit_divergence/temp0.6/without_occ12/
    logprobs/{recompute_run1, recompute_run2, rope}/
    per_token_<pair>/  divergence_summary_<pair>.csv
.../raw_decode_token_sequences/temp0.6/without_occ12/{recompute_run1,recompute_run2,rope}/
    sequences/  sequence_manifest.jsonl
```

## 运行方式（最小改动、不新增脚本，手动逐个跑）

已对现有文件做最小改动（默认值全部保持 temp=0 旧行为，不影响历史实验）：
- `module/vllm_client.py`：`chat_completion` 新增 `top_k`/`min_p`（`seed` 本已支持）。
- `run_logit_divergence.py`：collect 解开写死的温度，新增 `--temperature/--top-p/--top-k/--min-p/--seed/--run-name`；analyze 新增 `--rc-name/--rp-name/--tag`（可比较任意两个 run 子目录）。
- `run_raw_decode_token_sequences.py`：新增 `--top-k/--min-p/--seed`。
- 两个 `.sh`：转发 `TEMPERATURE/TOP_P/TOP_K/MIN_P/SEED`；输出路径自动按 `TEMP_TAG/RUN_LABEL/RUN_NAME` 分层；logit 版另有 `RUN_ANALYZE`（=0 只采集不分析，单 mode 手动跑时不会因缺另一条腿而报错）。

### logit divergence（方法 1+2 重跑）
```bash
COMMON="TEMPERATURE=0.6 TOP_P=0.95 TOP_K=20 MIN_P=0 RUN_LABEL=without_occ12 RUN_ANALYZE=0"

# 三次采集（换 mode 会自动重启 vLLM；输出自动进 .../logit_divergence/temp0.6/without_occ12/）
env $COMMON MODES=recompute SEED=1111 RUN_NAME=recompute_run1 bash run_logit_divergence.sh
env $COMMON MODES=recompute SEED=2222 RUN_NAME=recompute_run2 bash run_logit_divergence.sh
env $COMMON MODES=rope      SEED=1111 RUN_NAME=rope           bash run_logit_divergence.sh

# 两次分析（不需要 vLLM）；OUT 就是上面自动生成的目录
OUT=/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/06_context_free_segment_cache/cross_occurrence_function_vector/logit_divergence/temp0.6/without_occ12
python run_logit_divergence.py analyze --output-dir "$OUT" --rc-name recompute_run1 --rp-name rope            # KV 效应
python run_logit_divergence.py analyze --output-dir "$OUT" --rc-name recompute_run1 --rp-name recompute_run2  # 固有方差基线
```

### raw decode token sequences（原始序列重跑）
```bash
COMMON="TEMPERATURE=0.6 TOP_P=0.95 TOP_K=20 MIN_P=0 RUN_LABEL=without_occ12"
env $COMMON MODES=recompute SEED=1111 RUN_NAME=recompute_run1 bash run_raw_decode_token_sequences.sh
env $COMMON MODES=recompute SEED=2222 RUN_NAME=recompute_run2 bash run_raw_decode_token_sequences.sh
env $COMMON MODES=rope      SEED=1111 RUN_NAME=rope           bash run_raw_decode_token_sequences.sh
```

> 注意：`analyze`/后续对比脚本目前只做 recompute↔rope 与 run1↔run2 的两两比较；跨 run 的最终"是否超过基线"判定图表待数据齐了再补（可复用已有 plot 脚本思路）。
