# 通用判官（general judge）

给 agent 的一次产出打分：**读「任务要求 + agent 产物」→ 输出成功概率 + PASS/FAIL/PARTIAL
+ 问题标签**。核心特点是**通用**——判官自己从任务要求里抽出成功标准，对任意任务都能用，
不需要一个任务写一张手写检查表。

> 这个目录原来放的是 OpenHands / verify agent 的**学习副本**，现在已替换为这个**可用的
> 通用判官**。

## 为什么这样设计

| 借鉴来源 | 借来的东西 |
|---|---|
| Claude Code verification agent（`/home/wsh/claude-code/.../verificationAgent.ts`） | **判的方法**：从任务本身抽成功标准、对抗心态（找毛病而非盖章）、证据纪律（判"满足"要能引用产物原文）、PASS/FAIL/PARTIAL 裁决 |
| OpenHands critic taxonomy（`software-agent-sdk/.../critic/impl/api/taxonomy.py`） | **问题标签**：`did_not_follow_instruction` / `improper_tool_use_or_setup` / `incomplete_implementation` 等 |

不照搬 verify agent 的"把代码跑起来取证据"——我们的产物多是静态内容（doc / spec / HTML），
把"运行它"换成"逐条核对产物内容并引用原文"。可运行的任务（如 mcp server 代码）以后可再加执行式检查。

## 文件

| 文件 | 作用 |
|---|---|
| `prompt.py` | 判官的评判规则（system prompt）+ 允许的问题标签 |
| `schema.py` | `JudgeResult` 数据结构 + 从 LLM 输出稳健解析 JSON |
| `judge.py` | `GeneralJudge`：拼输入 → 调 LLM → 解析结果。判官模型走 OpenAI 兼容端点 |
| `extract.py` | 从 trace 抽「当前步任务要求」、从 decode TXT 抽「产物（工具调用）」 |
| `run_judge.py` | 批量给现有 decode 结果打分，多 arm 并排对比，输出 CSV + 明细 JSONL |

## 判官用哪个模型

`GeneralJudge` 默认指向**本地 vLLM（Qwen3-14B，`http://127.0.0.1:8000`）**，
用环境变量或参数即可切换，无需改代码：

```bash
export JUDGE_BASE_URL=http://127.0.0.1:8000   # 或任意 OpenAI 兼容端点
export JUDGE_MODEL=Qwen3
export JUDGE_API_KEY=EMPTY
```

- **本地 Qwen3-14B**：快、离线、可大批量；但与被判 agent 同族，有轻微"自己夸自己"偏差。
- **跨模型（GPT-5.4 / MiniMax）**：更权威、无同族偏差，外部有成本；论文阶段建议用它对
  一部分 case 交叉验证，证明结论不依赖判官选择。

## 用法

单次（代码里）：

```python
from judge import GeneralJudge
from extract import task_request_text, agent_output_text
from trace_utils import load_invocations

request = task_request_text(load_invocations("mcp_server_and_spec")[10])
output = agent_output_text(open("....txt").read())
result = GeneralJudge().evaluate(request, output)
print(result.success_probability, result.verdict, [i["label"] for i in result.issues])
```

批量对比两个 arm（需要判官 LLM 端点在跑）：

```bash
cd scripts/06_context_free_segment_cache/module/critic_reference
python run_judge.py \
  --results-dir ../../../../results/problem_exploration/cross_task_contextual_occ1_reuse \
  --arms recompute cross_contextual_rope \
  --out ../../../../results/problem_exploration/cross_task_contextual_occ1_reuse/judge
```

输出 `judge_summary.csv`（每 case 各 arm 的 score/verdict/issues 并排）和
`judge_details.jsonl`（每次评判的抽出标准、逐条 check、证据、问题、理由，供复核）。

## 现状

- 离线部分已验证：JSON 解析、从 trace 抽当前步要求、从 decode TXT 抽产物、拼判官输入。
- 要真正打分需要一个判官 LLM 端点（本地 vLLM 或外部 API）在跑。
- 方法上是"非劣性"用途：比 recompute 与复用两 arm 的成功率分布，而不是要求逐 token 一致。
