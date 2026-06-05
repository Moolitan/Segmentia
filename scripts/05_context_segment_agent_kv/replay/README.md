# Trace 回放 ContextSegmentKV harness（不依赖 agent 框架）

本目录替代了由 OpenHands 驱动的 `real_system/` runner，改成纯 **trace 回放**。
我们已经在 `src/traces/<task>/turn_<N>_inv_<M>.json` 里捕获了 agent 在每一次
LLM 调用时*本该*发送的完整请求（见 `src/traces/README.md`）。这里只是把这些
JSON 按顺序回放给改过的 vLLM，并加上 **skill KV 复用**。没有 agent、没有
OpenHands——只有 vLLM。

## 它做了什么

对每个任务，按顺序把 invocation JSON 当作 `/v1/chat/completions` 请求发出：

1. **格式转换**：Anthropic typed-block 消息 → OpenAI chat 格式
   （`tool_use` → `assistant.tool_calls`，`tool_result` → `role:tool`）。
   `_system_prompt.txt` 作为 system 消息；`_tools.json` 作为 `tools` 参数传入，
   这样它会落进模板化后的前缀里。
2. **结构化识别 skill 段**：某个 `tool_result` 对应的 `Read` 工具读的是
   `.../skills/<name>/SKILL.md`，则它就是 skill `<name>`。其内容会被包进
   `<context_segment id="<name>">…</context_segment>`，保证 source 和 target
   两侧的 token 边界完全一致。
3. **分配 source/target**（按首中尾 reload pattern）：
   - 某 skill 文本第一次出现 → **source**（vLLM 正常 prefill 并把这段 span 的
     KV 收集进 in-memory registry）；
   - 之后同一 skill 文本在*不同 token 位置*的每次首现 → **target**（从已保存的
     source 注入 KV，经 RoPE 位置修正，而不是重算）。
4. **测量**：每个请求的延迟（TTFT 代理，`max_tokens=1`）和 token 用量；对比
   recompute 与 reuse。

vLLM 机制本身没有改动——source 收集 / target 注入的代码链路见
`../real_system/VLLM_CONTEXT_SEGMENT_KV_CODE_WALKTHROUGH.md`。请求线格式也一致：
`vllm_xargs.context_segment_cache = {"sources":[…],"targets":[…]}`。

## 运行

```bash
# recompute vs reuse，全部 6 个任务，per-task 复用范围
bash run_replay.sh

# cross-task 复用：在某个任务里收集的 skill，在后续任务里被复用
REUSE_SCOPE=cross-task bash run_replay.sh

# 子集 / 单一模式
TASKS=internal_comms_incident_update,slack_launch_pack MODE=reuse bash run_replay.sh
TASKS=internal_comms_incident_update MODE=recompute bash run_replay.sh

# 不重启 vLLM（复用正在跑的 server；cache 可能是热的）
RESTART_VLLM=0 bash run_replay.sh
```

结果输出到 `results/05_context_segment_agent_kv/replay/`。

## Trace CKSim

真实 trace 上的 recompute-vs-reuse CKSim 也放在本目录。它跑两条独立轨迹：

1. 启动 vLLM，跑 `--phase recompute`，只 dump 完整上下文 recompute KV：
   `cksim-recompute-<task>-<skill>-occ<N>.pt`
2. 重启 vLLM，跑 `--phase reuse`，在 reuse pass 内 occurrence 1 收集 source，
   occurrence 2/3 只注入这个 pass 自己收集的 source，并 dump RoPE 位置重旋转结果：
   `cksim-reuse-<task>-<skill>-occ<N>.pt`
3. 再次重启 vLLM，跑 `--phase reuse_no_rope`，重新收集 source，只 dump
   direct/no-rope 复用结果：
   `cksim-reuse-no-rope-<task>-<skill>-occ<N>.pt`
4. 跑 `--phase summarize`，读取三边 `.pt` 并计算 CKSim。

运行：

```bash
bash run_cksim.sh
TASKS=launch_poster_page_pack bash run_cksim.sh
```

输出：

```text
results/05_context_segment_agent_kv/CKSim/trace_reuse_cksim.csv
results/05_context_segment_agent_kv/CKSim/trace_reuse_cksim_summary.json
```

CSV 的 `comparison` 列会区分 `recompute_vs_reuse` 和
`recompute_vs_reuse_no_rope`。

CKSim replay 只给 vLLM 设置 `VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR`。不要在这些
phase 里设置 `VLLM_CONTEXT_SEGMENT_KV_DIR=$KV_SAVE_DIR`，否则每次重启都会把前面
dump 的 `.pt` 全部预加载进 GPU，随着 recompute/reuse/no-rope 文件累积很容易 OOM；
每个 phase 的 source 都应在该 phase 内重新收集。

## 复用范围（reuse scope）

- `per-task`（默认）：source = 某 skill 在该任务内的第一次读取；target = 该任务
  内的中、尾两次重读。单独验证首中尾 pattern。
- `cross-task`：某 skill 的 source 在*任意任务*第一次读到它时收集；之后每个用到
  同一 skill 的任务都注入复用。`internal-comms`（internal_comms + slack_launch）、
  `doc-coauthoring`（doc_coauthoring + mcp_server）、以及
  `canvas-design`/`web-artifacts-builder`/`theme-factory`（launch_poster +
  web_artifact）都跨任务复现。

## 测量须知 / 注意点（看数前必读）

- **静态 token 位置**：replay 的 skill span 来自
  `core.config.SKILL_TOKEN_LOCATIONS`，这张表是用原始 `_system_prompt.txt` 生成的。
  因此 replay 不再在 system prompt 前拼 pass nonce；如果 system prompt、tools、
  模型 tokenizer 或 chat template 变化，需要先重新生成并更新配置里的位置表。
- **单任务内 prefix cache 是很强的 recompute**：invocation JSON 是累积超集，所以
  普通 prefix cache 本来就只重算每步的增量。带 target 的请求会设置
  `skip_reading_prefix_cache`（scheduler 必须精确停在 `target_start`），所以单任务
  内 SegKV 是用"整个请求的 prefix-cache 复用"换"一段拼接进来的 skill span"——对
  小 skill 通常**持平甚至更差**。这是预期行为，不是 bug。
- **复用真正划算的地方：cross-task**。当同一 skill 在一个没有共享前缀的任务里
  再次出现时，prefix cache 帮不上忙，但 segment KV 可以被拼接进去。这正是
  `REUSE_SCOPE=cross-task` 针对的场景。
- **`prompt_tokens` 在 recompute/reuse 之间是相同的**（它统计完整 prompt，与是否
  缓存/注入无关）。复用信号请看 `segkv_target_tokens`（被注入而非重算的 token 数）
  和延迟。
- **`cached_tokens` 通常是 `None`**——vLLM 的 OpenAI usage 这里不填
  `prompt_tokens_details.cached_tokens`。要做 prefix-cache 命中统计，请用 vLLM
  自己的日志 / `/metrics`。
- **上下文长度溢出**：Qwen3 以 `max-model-len 32768` 提供服务。最大的那些累积
  请求（`mcp_server_and_spec` ≈ 36K 估计，`doc_coauthoring`/`launch_poster` ≈
  29–30K）可能超过上限；这类请求会被捕获、记为 `[SKIP]`、计入 `num_errors`，而不会
  让整轮崩溃。把 `VLLM_MAX_MODEL_LEN` 调大并重启 vLLM 即可纳入它们。
  （注：实测当前 6 个任务在 32K 内，`num_errors=0`，估算偏保守。）
