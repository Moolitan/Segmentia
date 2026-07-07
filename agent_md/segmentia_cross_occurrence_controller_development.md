# Segmentia Cross-Occurrence Controller Development

## 总开发目标

利用同一 task 内同一 skill 的首次上下文化校准和跨 occurrence 历史，为后续 occurrence 选择在线重算预算与逻辑 blocks，在保留 KV 复用收益的同时改善完整工具行为一致性。

本轮验证的机制候选：

```text
缺失 skill-to-prefix interaction
    ↓
generation marker KV 偏移
    ↓
动作边界远端重新读取
    ↓
完整工具行为改变
```

12-pair batch已表明generation marker不具备稳定位置特异性，该候选当前
No-Go。后续controller不能再以前述链条为既定机制。

## 开发阶段总览

| 阶段 | 名称 | 目标 | 当前进度 | 剩余 |
|---|---|---|---|---|
| 1 | 行为对象确认 | 使用完整 `tool_calls` 定义 presence 和 trajectory。 | 已完成。 | 无。 |
| 2 | 机制设计 | 定义 interaction -> FV drift -> behavior change 中介链。 | 文档完成。 | 需要实验验证。 |
| 3 | Marker relay causality | 验证 generation marker 是否把 rope 误差中继到动作边界。 | 已完成pilot和12-pair batch；marker仅1/10 eligible case独占恢复，不优于random且会破坏正确行为，结论No-Go。 | 无；若继续需作为新的“多位置敏感性扫描”重新立项。 |
| 4 | Raw decode token audit | 保存未经chat parser改写的生成序列，并比较RoPE差异与sampling baseline。 | 已完成旧版72条greedy结果和temp=0.6 occurrence-3三路各12条采集审计；服务隔离与当前span注入正确。单样本两种比较均为9/12；当前批BGE-large thinking cosine为0.9470 baseline、0.9535 RC1–RoPE，仍不足以证明等价；当前目录不存在文档声称的validator。 | 修复采集元数据、occurrence配置、无损bytes/token-ID保存与validator；再做多seed等价性实验。 |
| 5 | Interaction mediation | 验证 masking 是否同时引起 FV drift 和 rope 方向行为变化。 | 未开始，当前停止。 | 先完成raw token审计并形成新机制假设。 |
| 6 | Block restoration | 验证高 interaction blocks 是否恢复 FV 和行为。 | 未开始，当前停止。 | 依赖新的机制假设及独立gate。 |
| 7 | Online controller | 用 FV drift 定义 `fun_d`，用历史 drift 和 interaction coverage 定义 `fun_r`。 | 未开始，当前阻塞。 | 阶段3 No-Go；需要先选择新的可验证机制。 |
| 8 | 系统评估 | 测量行为、重算比例、TTFT 和端到端开销。 | 未开始。 | 尚无通过机制gate的controller可评估。 |

## 当前保留文件

```text
scripts/06_context_free_segment_cache/cross_occurrence_controller/
  README.md
  Plan.md
  LiangFunctionVectorDesign.md
  function_vector_capture/
    FunctionVectorDefinition.md
    README.md
    analyze_repair_screening.py
    prepare_function_vector_capture.py
    run_function_vector_capture.py
    run_function_vector_capture.sh
    summarize_function_vector_capture.py
    validate_function_vector_capture.py
  generation_marker_relay/
    README.md
    prepare_marker_relay.py
    run_marker_relay.py
    run_marker_relay.sh
    run_marker_relay_batch.sh
    summarize_marker_relay.py
    summarize_marker_relay_batch.py
    validate_marker_relay_kv.py
  raw_decode_token_sequences/
    README.md
    run_send_request_message.py
    run_raw_decode_token_sequences.py
    run_raw_decode_token_sequences.sh
  gated_selective_recompute/
    VLLM_MODIFICATION_MAP.md
```

Function Vector capture 只作为历史工程数据采集和描述性 screening。
Generation marker relay 已完成pilot和batch，并判定为No-Go；它不是在线控制器。
当前工作切换到raw decode token审计；在看清全部真实生成结构前，不继续假设
固定action boundary或Function Vector。
工程验证记录位于：

```text
results/problem_exploration/function_vector_capture/
```

## 当前方法定义

当前已成立的内部对象只有：

```text
head activation:
  a[c, layer, head, mode]

paired recompute-rope difference:
  delta_a[c, layer, head] = a_recompute - a_rope
```

以下对象已经完成数学定义，但尚未通过实验验证，当前不再作为优先机制：

```text
layer-local Contextual Repair Vector:
  rho[c, layer]

layer-local Skill Function Vector:
  v[skill, layer]
```

在线接口仍保留为远期系统骨架：

```text
fun_r(H_<t, c_t) -> r_t
fun_d(H_<t, c_t, y_t, o_t) -> e_t
```

在 Function Vector existence、matched control 和无 oracle drift estimation
通过前，不定义 `function_vector_ref`，也不把 occurrence 1 activation 或
recompute–rope cosine 写入 `fun_d`。

## 已验证事实

- Context-free skill KV 复用可以在 thinking 整体语义相近时改变完整 `tool_calls` 行为。
- 完整行为需要分别表示 `tool_call_presence` 和有序 `tool_trajectory`。
- `visible_start`、`tool_call_start` 和 `function_name` 是不同 token observation，不能合并为统一动作边界。
- Function-name 候选分数只能提供局部候选排序证据，不能单独证明动作正确性。
- 当前 FV capture 的 `target_end` readout 实际是 `<|im_start|>` 模板 token。
- Attention probe 的 `prompt_after_skill_context` 实际是
  `<|im_start|>assistant\n` 3个 generation marker token。
- `slack-gif-creator` occurrence 2/3 的动作边界 query 对 marker region 的
  rope-recompute attention mass 差异都为正，且主要从 Layer 20后增大。
- Generation marker relay batch覆盖12个task-skill pair、24个case，其中
  10个case存在RoPE工具名行为差异。
- Marker在eligible case中至少部分恢复6/10、完整工具名行为恢复4/10；
  random分别为6/10和5/10，marker没有位置优势。
- Marker只在1/10 eligible case中独占恢复；该case的完整结构化
  `tool_calls`与recompute一致，是局部阳性，但不足以支持跨case机制。
- 在14个RoPE工具名行为原本正确的case中，marker patch破坏4个，说明该操作
  不是无害修复。

## 尚未验证的假设

- Agent skill 会形成 Todd-style function vector。
- 该向量在同一 task-skill 的不同 occurrence 中保持稳定。
- Function-vector patch/addition 能够因果改变完整工具行为。
- Context-free reuse 通过缺失 skill-to-prefix interaction 破坏该向量。
- 高 interaction blocks 的在线重算能够恢复向量和完整行为。
- 历史 FV drift 能够指导后续预算。
- 多个随机位置的恢复分布是否呈现稀疏位置敏感性。
- 局部KV patch引起的行为恢复究竟是机制修复还是一般decode轨迹扰动。

## 当前风险

- 现有 readout 是固定模板 token，不保证携带 skill function。
- 当前保存的是原始逐 head activation，不是已经定义好的 Function Vector。
- 直接比较不同 occurrence 的原始 activation 会混入上下文、位置和历史长度差异。
- Function vector 可能编码具体工具，而不是稳定 skill function。
- Per-head capture 已经过真实 Qwen3 replay，tensor 数值完整性通过。
- Activation patching 和定向 masking 尚未实现。
- Occurrence 1 全层全头 attention capture 的开销尚未测量。
- PagedAttention 下 selected recompute 不能覆盖共享 prefix-cache blocks。
- Marker机制阶段已经No-Go，后续interaction、block restoration和controller
  当前没有依据。
- Temp=0.6 raw decode 每个条件只有一次采样；9/12 对 9/12 是相同点估计，
  不是 RoPE 与固有方差等价的证据。
- 当前temp=0.6批次的BGE-large CLS cosine已独立重算：RC1–RC2与RC1–RoPE
  的thinking均值为0.9470/0.9535，action均值为0.9078/0.9386。高语义cosine
  不等于工具行为一致，也不能在单样本下证明等价。
- Raw manifest 未固化采样参数、模型/服务版本、KV路径和finish reason。
- Raw TXT直接拼接API token字符串而非按bytes/token IDs无损保存；同名raw与
  logit run只有26/36逐字符一致，虽未改变本轮工具名序列分类，不能视为同一
  条精确轨迹。
- 分叉后的index-wise JSD条件在不同生成前缀上，不能作为纯KV损害量。
- Marker patch 使用同 case recompute KV，是 oracle机制诊断，不是可部署方法。
- Post-prefill patch 只改 KV，不改当前 hidden state和首 token logits；即使行为
  未恢复，也只能否定“marker作为后续 memory relay”这一受限版本。

## 系统与 Replay 边界

在线 KV：

```text
selected recompute 只覆盖当前 request 私有 KV
历史记录使用逻辑 token/block 标识
不修改共享 prefix-cache blocks
```

实验 replay：

```text
服务重启边界 = (mode, task)
task 内按 invocation_index 顺序
后续 occurrence 通过 prefix cache 继承历史注入
当前轮只显式处理当前 skill span
不重复显式注入历史 skill spans
```

## 本轮修改与验证

修改：

- 新建 `function_vector_capture/` 子研究目录。
- 增加固定 pilot manifest、task 内 replay、marker 协议和离线 tensor 校验。
- 在本地 vLLM 增加默认关闭的 Function Vector capture probe。
- 在 Qwen3 `o_proj` 后采集 readout 的逐头 raw output 和 residual-stream contribution。
- 为 Pilot replay 代码补充 warmup、occurrence 1 calibration、marker 生命周期
  和失败终止语义的就近注释。
- 轻量 `capture_manifest.json`、`decode_rows.jsonl` 和运行时
  `current_capture.json` 默认路径改为
  `results/problem_exploration/function_vector_capture/data/`。
- 大体量 `.pt` 继续保存在外存
  `$SEGMENTIA_OUTPUT_DIR/cross_occurrence_function_vector/pilot1/head_outputs/`。
- 增加离线汇总脚本、tensor 验证表格和逐层重构误差图。
- 将 FV Capture 从独立研究结果降级为工程验证记录，并从问题探究阶段的
  研究结论、研究导航和阶段 source manifest 中移除。
- 保留轻量数据和外存 tensor，不重新运行 capture。
- 删除 capture 工程链路中的历史行为依赖：manifest 不再读取 pair summary，
  validator 只检查 tensor，summarizer 不再生成行为表。
- 新增 `FunctionVectorDefinition.md`，严格区分 head activation、layer-local
  Contextual Repair Vector 和经过对照、跨 case 平均及因果验证的 Skill
  Function Vector。
- Function Vector 首先采用 layer-local 定义，禁止在未验证 layer alignment
  时直接求和远距离 layer 的 residual vectors。
- 在旧 `LiangFunctionVectorDesign.md` 顶部加入定义优先级说明；旧 occurrence 1
  reference 和跨层求和公式不再作为当前正式定义。
- 新增 `analyze_repair_screening.py`，按 `(task, skill, occurrence,
  invocation_index)` 配对 recompute/rope tensor，输出 head/layer repair、
  跨 occurrence alignment、Top-K concentration 和 overlap。
- 新建 `results/problem_exploration/function_vector_repair_screening/`，保存
  6张可复查表、3张核心图、metadata、source manifest 和 summary。
- 重写 repair screening summary：补充实验对象、逐步伪代码、全部指标的通俗
  定义、三张图的读图方法，以及“可支持/不可支持”结论边界；原始数值和
  Go/No-Go 判断保持不变。
- 核对真实 tokenizer 后确认 `target_end` 是 `<|im_start|>`，并修正 capture
  README 与 screening summary 中“post-skill正常位置”的误导性表述。
- 新增 `generation_marker_relay/`：固定 occurrence 2/3 manifest、same-case
  recompute control KV dump、`recompute/rope/marker/tail/random` 五个 arms、
  patch acknowledgment、KV validator和行为 gate summarizer。
- 本地 vLLM 新增默认关闭的 `post_prefill_patch.py`。它在正常 forward 后、
  logits/sampling 前，使用 marker registration 提供的当前 request block
  mapping覆盖指定3-token span的 K/V；source/target位置、token identity、
  layer和component均严格校验。
- Patch arm 按 occurrence隔离为独立 run mode；occ3 arm中的occ2只做普通
  rope warmup，避免历史 marker patch污染当前因果判断。
- 为 `prepare_marker_relay.py` 补充中文设计注释，解释公共模块导入、三类
  3-token span、确定性随机 control、manifest完整性检查、arm映射和CLI边界；
  未修改任何执行逻辑。
- 为 `run_marker_relay.py` 补充中文设计注释，解释五种 arm、occurrence 1
  calibration、`invocation_index` replay顺序、registration anchor与patch
  target分离、marker/status原子协议、失败终止和JSONL字段；未修改执行逻辑。
- 修复首次真实 patch replay 暴露的 request ID边界：OpenAI Chat serving会将
  client ID `cf-marker-...` 转为 EngineCore ID `chatcmpl-cf-marker-...`；
  marker/status现在同时记录两者，并使用 engine ID匹配 registration。
- Shell新增 `RUN_RECOMPUTE/RUN_ROPE/RUN_PATCH`阶段开关；在已有 baseline和
  control KV完整时可 `CLEAN_OUTPUT=0` 只重跑 patch arms，避免重复长实验。
- 第二次真实 patch replay仍未写 acknowledgment。宿主进程检查确认 EngineCore
  已收到 marker环境变量，日志再次确认唯一 registration成功收集。因此移除
  serving request ID的硬匹配，改用唯一 `registration_cache_id + anchor span`
  匹配；expected/actual/client IDs全部保留在 status和日志中用于审计，0个匹配
  会打印 available registrations，多个匹配直接失败。
- 分析已完成的12-pair marker relay batch，并新增
  `batch/batch_arm_outcomes.csv`、恢复/副作用核心图和batch `source_manifest.csv`。
- 重写marker relay根summary和batch summary，区分脚本原始
  `conditional_go_partial_cases`与研究层面的No-Go；原始machine gate未修改。
- 同步更新问题探究阶段summary、阶段source manifest和本development文档，
  停止以marker relay为前提的后续controller路线。
- 新增`raw_decode_token_sequences/`，覆盖12个task-skill pair的occurrence
  1/2/3和recompute/rope，共72条原始生成序列。
- 将`run_send_request_message.py`从单case message打印工具改为全case Qwen3
  chat-template文本导出工具：读取36个唯一task-skill-occurrence case，通过
  本地tokenizer执行`apply_chat_template(tokenize=False)`，按
  recompute/rope分别写出72个TXT；不连接vLLM、不decode。
- 采集脚本只读取`choices[0].logprobs.content[*].token`并直接拼接TXT；
  不读取或保存API parser生成的`reasoning_content/content/tool_calls`。
- Shell按`mode -> task`循环，在每个`(mode, task)`边界重启vLLM；Python在
  task内按`invocation_index`运行，occurrence 1/2/3全部保存。
- 断点续跑仍重放已有case以重建prefix状态；新旧raw sequence不一致时立即
  失败，不覆盖原文件。
- 新增离线validator，检查预期72个唯一key、文件存在性、SHA256、字符数和
  token数；结果目录与旧attention/marker实验隔离。

验证：

- 本轮未启动 vLLM 或 decode 实验；分析使用2026-06-28已有真实 run。
- Python 文件通过 `py_compile`。
- Shell 入口通过 `bash -n`。
- 真实 manifest 包含 `slack-gif-creator` occurrence 1/2/3，readout
  分别为7286、11287和15580。
- 真实 run 包含16个 replay rows：5个 selected capture 和11个 warmup。
- 5/5 capture tensor 的 shape 均为预期值；最大逐层相对 L2 误差范围为
  0.002624–0.002862，全部低于0.02阈值。
- 工程 summary 已明确区分“采集数据完整”与“Function Vector 尚未定义”。
- 数学定义文档明确列出 sufficiency、necessity、held-out generalization、
  skill specificity 和 compactness 判据；当前5份 tensor 只能做描述性 screening。
- Markdown 数学结构检查通过：30个 display-math 块成对闭合，3个 `aligned`
  环境及所有定界符、花括号均配对。
- 数学定义文档补充五步直观流程、核心对象对照表和每个关键公式的通俗解释；
  保留原公式与严格判据不变。
- 将容易误解为模型训练的 `train/test cases` 全部改为
  `discovery/validation cases`；这里只做候选选择与独立复核，不更新模型权重。
- Repair screening 读取2个 paired occurrences，生成3200行 head metrics、
  80行 layer metrics、1600行 cross-occurrence head alignment、40行
  layer alignment、20行 concentration 和10行 overlap。
- 所有 tensor metadata、shape、layer index 和有限值检查通过；所有 cosine
  与相对 norm 均可用，无零范数、NaN 或 Inf。
- Head repair cosine 中位数为0.801，40层 layer repair cosine 全为正；
  Top-20 energy share 为0.476/0.407，Top-20 重合13个 heads。
- Log recompute norm 与 log repair norm 的 Pearson 相关为0.884/0.890，
  说明绝对 repair energy 存在强 activation-scale 混杂。
- 输出行数、Top-K share 边界、metadata row counts 和一个独立重算的 head
  repair norm 均通过断言；脚本通过 `py_compile`。
- 重写后的 summary 已核对本地图片链接、表格关键数值、tensor shape、
  occurrence/invocation/readout metadata 与分析脚本计算逻辑；本轮未修改或
  重新生成脚本、表格、图片和实验数据。
- Generation marker manifest 静态生成通过：occ2 marker
  `[11287,11290)`，occ3 marker `[15580,15583)`，三个 arm span均为3 tokens。
- 新增 Python 文件通过 `py_compile`，shell入口通过 `bash -n`。
- vLLM post-prefill patch通过 `py_compile`；手工 tensor test验证了
  registration anchor与patch target位置不同时仍能按当前 request block table
  正确覆盖 K/V并写出 acknowledgment。
- pytest测试文件已加入；当前 `opencode` 缺少 pytest，base pytest又缺少
  `tblib`，因此未安装依赖、未运行完整 pytest。
- Batch包含120/120条唯一selected rows；72/72条patch row均确认
  `status=applied`、覆盖40层和3个token；12/12份control KV validation均为
  `valid`。
- 10/24个case存在RoPE工具名行为差异。Marker/tail/random至少部分恢复分别为
  6/5/6个case，完整工具名行为恢复分别为4/4/5个case。
- Marker/tail/random在14个RoPE原本正确的case中分别破坏4/4/3个case。
- 对arguments JSON规范化后，eligible case中完整结构化`tool_calls`恢复数为
  marker 2/10、tail 1/10、random 1/10；其中一个三arm共同恢复为空
  `tool_calls`，唯一非空marker独占恢复是`mcp_server_and_spec/mcp-builder`
  occurrence 3。
- 静态核对batch shell确认每个pair调用独立run，run内部按
  `(mode, task)`重启；唯一阳性中的历史skill路径原本就存在于source trace，
  不是跨task prefix-cache污染。
- 静态case plan确认36个唯一task-skill-occurrence case，双mode共72条；
  每个task内均按`invocation_index`排序。
- 新增Python文件通过`py_compile`，shell入口通过`bash -n`，两个CLI的
  `--help`均可加载。
- raw-stream模拟确认采集函数忽略`message`中的解析字段，并能从
  `logprobs.content`保留`<think>`、`</think>`和`<tool_call>`。
- `run_send_request_message.py`通过`py_compile`；先向`/tmp`离线导出72个
  TXT，确认recompute/rope各36个、所有文件均以
  `<|im_start|>assistant\n`结束，两个mode目录逐文件完全一致。
- 正式离线导出已写入
  `results/problem_exploration/raw_decode_token_sequences/rendered_prompts/`，
  共72个TXT；未启动vLLM、未发送HTTP请求、未执行模型推理。
- 审计temp=0.6三路真实结果：3份manifest各12个唯一case，36/36 SHA256和
  字符数一致；全部包含think边界和结束token，工具JSON均可解析，最大1767
  token，未触及4096上限。
- 复算有序工具名序列：recompute_run1–run2为9/12，
  recompute_run1–rope为9/12，run2–rope为8/12；前两者paired outcome为
  both-match 7、baseline-only 2、rope-only 2、both-fail 1，
  McNemar exact p=1.0。
- 9/12的Wilson 95%区间为46.8%–91.1%，当前样本不能做等价性结论。
- 对照raw TXT与独立logit token stream：26/36逐字符一致；其余10条为
  emoji/Unicode表层差异，action signature未变化，但证明两次采集不是同一
  条精确trajectory。
- 新增参数化BGE embedding cosine分析器，按旧口径使用BGE-large、CLS
  pooling、L2 normalization和512-token truncation。当前三路各12条均通过
  think/action非空与文件集合一致性检查；生成36行逐case CSV和汇总JSON。
- Embedding分析器通过`py_compile`和CLI检查；使用本地GPU完成真实计算，
  cosine点积裁剪到`[-1,1]`以去除float32的`1.000000119`舍入噪声。
- Python文件通过`py_compile`，shell入口通过`bash -n`；本轮未启动vLLM。
- 新增raw decode根summary、逐case审计表和source manifest，并加入问题探究
  阶段导航。
- 使用10行合成 selected outputs验证 behavior summarizer：
  presence、first function、ordered trajectory和
  `go_all_eligible_cases` gate计算符合预期。
- 本轮未启动 vLLM server或任何 decode实验。
- 注释修改后重新执行 AST解析和临时 manifest生成，输出边界保持不变。
- `run_marker_relay.py` 注释修改后重新执行 AST解析和 replay-plan断言，各
  mode的请求数、selected occurrence及顺序保持不变。
- 首次真实运行已完成 recompute、rope和6份 recompute control KV验证；第一个
  `marker_patch_occ2` 因 client/engine request ID不一致未执行 patch。日志确认
  marker registration `[11287,11290)` 已成功收集，故问题不在 KV数据或block
  mapping。
- 第一版 `chatcmpl-` request ID映射只通过了静态 tensor测试，真实 replay仍
  未 acknowledgment，已被后续 registration identity方案替代，不再作为当前
  正确性结论。
- Registration identity修复增加了“expected/actual request ID不同但唯一
  cache/span一致”的单元与手工 tensor case；runner同时校验 source、anchor
  span、target span和双 ID审计字段。修复后尚未再次重跑长实验。
- 手工 mismatch case已通过：`guessed-engine-id` 与
  `actual-engine-id-with-unknown-decoration` 不同时，唯一 cache/span仍正确
  patch并回写 actual/expected/client IDs；Python通过 `py_compile`，shell通过
  `bash -n`。
- 新增
  `scripts/06_context_free_segment_cache/cross_occurrence_controller/gated_selective_recompute/VLLM_MODIFICATION_MAP.md`，
  静态整理当前本地 `/home/wsh/vllm` 修改：请求解析、scheduler 与 prefix-cache
  边界、GPU worker KV 注入/采集、RoPE 修正、post-prefill patch、attention probe、
  function-vector probe、工作树状态和相关测试入口。本轮只写文档，未修改
  `/home/wsh/vllm`，未启动 vLLM。

## 下一步

当前先不要用现有单样本图宣称“RoPE不劣于模型固有方差”。下一轮应先修复
manifest配置快照、occurrence参数化、bytes/token-ID无损保存、finish reason
与validator；随后按同一组多个seed采集recompute/rope，并预先定义完整行为和
任务质量的等价界值。Logit比较只在共享生成前缀或teacher-forced同一前缀上做
因果解释。
