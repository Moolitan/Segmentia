# Hardware I/O characterization

This directory measures the real CSKCache data path with offline Skill KV.
Tests 01--06, 08, and 11--14 do not start vLLM. Tests 07, 09, and 10 are
long-running Agent experiments that start one independent vLLM server per task.
None of these tests modifies the original offline KV tensors.

## 测试内容

五个独立测试分别为：

1. `01_hardware_topology.py`：采集 CPU、NUMA、SSD、GPU 和 PCIe 拓扑。
2. `02_ssd_to_cpu.py`：读取真实 40 层 Skill KV，测量 cold/warm、1/4 线程的 SSD/page cache→CPU 性能。
3. `03_cpu_memory.py`：使用 `libnuma` 固定并验证物理页归属，测量本地 NUMA、跨 NUMA 和 pageable→pinned 内存拷贝。
4. `04_pcie_h2d.py`：测量 pageable/pinned、同步/异步 PCIe H2D 传输。
5. `05_skill_kv_path.py`：测量真实 SSD/page cache→pinned CPU→GPU 的串行完整路径。

另有结果分析和 Agent 调度窗口测试入口：

- `06_analyze_results.py`：汇总前五项硬件测试结果并生成论文图表、CSV 和总结文档。
- `07_run_agent_skill_schedule_batch.py`：依次运行 `src/task_prompt/` 的 13 个任务，测量从 vLLM 解析出完整 SkillAction 到下一次请求进入 scheduler 前的完整预取窗口。
- `07_agent_skill_schedule_window.py`：旧的单 case、Observation 之后尾部窗口诊断脚本；不作为正式批量实验入口。
- `prepare_990pro_skill_pool.py`：把除`docx`和`writing-systems-papers`外的11个离线Skill KV复制到挂载在`/mnt/990_pro`的990 PRO；逐文件校验SHA-256并重写目标副本manifest中的绝对路径。它不会格式化或挂载设备，且在目录不是`/dev/nvme1n1`独立挂载点时拒绝写入。
- `08_measure_agent_skill_kv_loading.py`：不运行Agent/vLLM，启动时一次性分配5 GiB pinned arena，逐一实测11个Skill的完整40层990 PRO→pinned CPU、page cache→pinned CPU和pinned CPU→GPU延迟。该实验只测数据加载，不涉及提前预取或Agent窗口覆盖。
- `09_run_agent_skill_locator_batch.py`：使用direct reuse重跑相同13个任务，只使用vLLM内部的四个单调时钟边界测量Skill预取窗口；任务读取、缓存解析、trace关联、fingerprint和子进程管理均由09自身实现，不导入07。批量测量显式关闭OpenHands Visualizer并取消子进程终端回显，但仍完整保存`launcher.log`。除完整13任务图外，后处理还生成剔除`docx`和`writing-systems-papers`的11任务图；生成该附加图不需要重新运行Agent。
- `10_run_mermaid_skill_timeline.py`：对同一个Mermaid任务执行3次相互隔离的direct-reuse运行，将粗粒度T0--T1进一步拆分到HTTP响应、SDK转换、SkillTool、Observation、请求组装和发送阶段；该脚本独立解析manifest和trace并管理运行，不导入07或09。
- `11_prepare_raw_skill_kv.py`：把相同11个Skill的440个层对象离线写入990 PRO上的一个24 GiB LMCache raw-block文件。它保留原始逐层文件，按Skill写检查点，支持中断后重跑。
- `12_measure_raw_skill_kv_to_pinned.py`：预先分配并注册40个pinned CPU buffer，然后对每个Skill用一次LMCache/Rust `batched_read()`提交40层O_DIRECT I/O；只测SSD→pinned CPU，不启动Agent、vLLM、H2D或纠错。
- `13_verify_cskcache_metadata.py`：按CSKCache Catalog中的真实物理extent读取11个Skill、440层，并逐层验证payload SHA-256。
- `14_verify_async_cskcache_host_load.py`：使用一次性预分配和注册的pinned buffer组，验证异步T0提交、`HOST_READY`、整组内容和逻辑lease回收。

Edit `config.yaml` when changing the run name, cache object, repetitions, or GPU.
No command-line arguments are required.

```bash
conda activate opencode
cd /home/wsh/openhands_code_research/scripts/08_lmcache_mp/paper_experiment/hardware_io_characterization

python 01_hardware_topology.py
python 02_ssd_to_cpu.py
python 03_cpu_memory.py
python 04_pcie_h2d.py
python 05_skill_kv_path.py
python 06_analyze_results.py
```

## 旧Agent Skill窗口实验（已废弃）

`07_run_agent_skill_schedule_batch.py`的T1/T2依赖OpenHands callback，且跨进程
直接相减`time.time_ns()`墙上时钟。该脚本和它产生的T0--T5数据只保留
用于追溯，不再作为正式延迟结果。下文旧边界说明不得用于新图表。

旧入口是：

```bash
conda activate opencode
cd /home/wsh/openhands_code_research
python scripts/08_lmcache_mp/paper_experiment/hardware_io_characterization/07_run_agent_skill_schedule_batch.py
```

脚本读取 `config.yaml` 中的 13 个任务。每个任务将完整 `.txt` 作为一条 user
message，使用 `SEGMENTIA_MODE=no_reuse` 和 `max_iterations=2`，并启动独立的
vLLM 与独立 OpenHands conversation。不同任务不会共享 prefix cache、对话历史或
server 状态。工作流任务只把入口 Skill 作为该任务的样本；下游 Skill 不会额外计数。

脚本记录六个边界：

```text
请求 A 的模型输出经 vLLM tool-call parser 形成完整结构化调用
tool="skill", arguments={"name": "..."}
    ↓
T0：目标 Skill 已知，此时可以立即发起 KV 预取
    ↓
T1：OpenHands 客户端收到请求 A 的结构化响应
    ↓
OpenHands 解析并执行目标 SkillTool
    ↓
T2：目标 Skill Observation callback 完成
    ↓
OpenHands 处理同轮其他工具、更新历史并构造请求 B
    ↓
T3：请求 B 进入 LiteLLM transport wrapper
    ↓
T4：请求 B 交给原始 LiteLLM transport
    ↓
请求 B 经 HTTP 进入 vLLM，完成 chat template、tokenization 和内部 Request 构造
    ↓
T5：EngineCore 即将调用 scheduler.add_request(request)
```

该旧脚本原定义`prefetch_window_ms = (T5 - T0) / 1e6`。T5之前的所有真实
处理都会自然计入窗口，包括 SkillTool、同轮其他工具、OpenHands 事件处理、
prompt/history 构造、HTTP 传输、vLLM tokenization 和 EngineCore 请求预处理。
T5 之后的 scheduler 排队、GPU embedding、Transformer Prefill 和 decode 不计入。

请求 A 和请求 B 均只增加唯一 `X-Request-Id` 用于跨进程关联，不改变 prompt、
token 序列或 KV 路径。T0 记录同时保存请求 A ID 与 `tool_call_id`；T1 按请求 A
ID 匹配客户端响应；OpenHands 用同一 `tool_call_id` 关联 T2 与请求 B；T3/T4
来自请求 B 的 transport 探针；T5 再按请求 B ID 精确匹配。任一时间戳缺失、
重复、无法关联、不能满足 `T0 <= T1 <= T2 <= T3 <= T4 <= T5`，或五段之和
与 `T5-T0` 不一致，该 case 都会失败并保留日志。

每个 case 的原始日志和结果保存在外存 batch 目录。已完成且实现 fingerprint
不变的 case 会在重跑时跳过，失败 case 可用同一命令继续。当前批次名是
`all_task_prompts_v2`，因此不会覆盖已完成的 v1 原始结果。全部 13 个任务完成后，
脚本为每个任务发布三类图表：完整 `T5-T0`、五段堆叠分解，以及预取窗口与
Skill KV 加载估计的对比。阶段分解图包含 `T0-T1`、`T1-T2`、`T2-T3`、
`T3-T4` 和 `T4-T5`，每个任务一行。

KV 加载对比同样覆盖全部 13 个任务。脚本从每个离线 manifest 的 `cache_dir`
和 `data_files` 读取40层文件的实际总字节数，再使用 Test 05 的8,021-token真实
Skill KV路径中位数按容量线性缩放，分别估计：pinned CPU→GPU H2D、page-cache
warm→GPU串行路径和当前Gen2×4 SSD cold→GPU串行路径。三个面板都同时画出
`T5-T0`可用窗口与对应加载估计。它们是基于真实字节数和实测路径的机会估计，
不是本批次实际执行了KV预取，也不包含并发争用、KV page安装或纠错开销。

## 逐 Skill 实际 KV 加载时间

本实验固定从Samsung 990 PRO上的文件系统读取11个Skill，明确排除
`docx`和`writing-systems-papers`。它不启动OpenHands、vLLM、LMCache或模型推理，
也不分析提前预取；目标只是测量真实数据移动延迟。

当前配置要求990 PRO整盘文件系统挂载到`/mnt/990_pro`，设备为
`/dev/nvme1n1`。仅创建目录不等于挂载：脚本会使用`findmnt`验证目标、设备和
读写属性，防止误把数GiB KV复制到系统根分区。挂载及创建用户可写目录：

```bash
sudo mount /dev/nvme1n1 /mnt/990_pro
sudo install -d -o wsh -g wsh /mnt/990_pro/skill_save_pool
findmnt -T /mnt/990_pro
```

先复制和验证测试对象：

```bash
conda activate opencode
cd /home/wsh/openhands_code_research
python scripts/08_lmcache_mp/paper_experiment/hardware_io_characterization/prepare_990pro_skill_pool.py
```

复制程序保留原离线pool，只创建990 PRO副本。它按与实验相同的manifest解析优先级
选择11个Skill，复制每层`.pt`、统一capture manifest和完成标记，完整校验SHA-256；
历史pool若仍有逐层sidecar则一并复制，但新CSKCache对象不再需要sidecar。程序只重写
目标副本的`cache_dir`和`staging_dir`。已存在且SHA-256一致的文件会跳过，因此
中断后可以直接重跑。复制过程本身会污染page cache；应等复制结束后再启动正式测试，
正式cold样本还会逐文件执行`POSIX_FADV_DONTNEED`。

运行加载测试：

```bash
conda activate opencode
cd /home/wsh/openhands_code_research
python scripts/08_lmcache_mp/paper_experiment/hardware_io_characterization/08_measure_agent_skill_kv_loading.py
```

脚本首先一次性分配5 GiB pinned CPU arena；该分配时间记录在原始JSON中，但排除
在所有加载计时之外。每个Skill的40层使用互不重叠、4 KiB对齐的arena切片，因而
整个Skill在H2D开始前已经同时驻留于pinned CPU内存。GPU端只分配一个最大层大小的
buffer，40层H2D按同一个CUDA stream依次提交，传输总字节数与真实Skill一致。

每个Skill分别测量：

1. 990 PRO文件系统cold读取→完整Skill的pinned CPU切片；
2. Linux page cache warm读取→相同pinned CPU切片；
3. 完整Skill已驻留pinned pool后的pinned CPU→GPU H2D。

三条路径分别预热5轮并记录10轮，图中使用p50。cold状态定义为每次完整Skill读取前
对40层文件执行`POSIX_FADV_DONTNEED`；这是buffered I/O下的best-effort文件级
冷缓存状态，不等于Direct I/O。所有Skill串行执行并复用同一个5 GiB pinned arena；
每个Skill结束后其切片可被下一个Skill覆盖，但arena本身直到进程结束才释放。

只有11个Skill全部完成、每条路径样本数正确且文件checksum稳定后，才会生成：

```text
/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/08_lmcache_mp/
  paper_experiment_hardware_io/a6000_qwen3_14b_io_v1/
  08_skill_kv_loading_990pro.json

results/problem_exploration/hardware_io_characterization/
  figures/skill_kv_loading_990pro.pdf
  figures/skill_kv_loading_990pro.png
  tables/skill_kv_loading_990pro.csv
```

这些结果只描述加载路径本身，不能解释为Agent请求中已经发生提前预取，也不能
等价为端到端TTFT收益。虽然内存布局模拟`LMCACHE_LOCAL_CPU=True`时完整Skill
进入持久pinned hot cache的状态，但脚本没有启动LMCache，因此不会声称测量了
LMCache元数据查找、cache policy、KV page安装或调度开销。

## LMCache raw-block批量读取优化

逐层文件实验包含40次文件open/read/close和40个独立同步点，不能利用990 PRO的
高queue depth。本组实验改用LMCache已有的Rust raw-block后端：离线阶段将440个
层对象写入一个固定slot文件，在线microbenchmark阶段为一个Skill的40层准备40个
长期pinned buffer，并将40个offset和buffer放进同一次`io_uring` batch。这里的
“一次batch”不是把40层拼成一个连续read；它是一次提交40个独立、4 KiB对齐的
O_DIRECT请求，使NVMe能够并行处理它们，随后调用方只等待一次batch完成。

前提条件：

- `/mnt/990_pro`必须是`/dev/nvme1n1`的独立读写挂载点；
- `opencode`环境必须能够`import lmcache_rust_raw_block_io`；
- `config.yaml`中的raw文件、manifest、容量、alignment和queue depth在转换与测量
  之间不得变化；
- 24 GiB raw文件只属于本实验。两个脚本不会格式化、挂载或删除磁盘内容。

先执行一次离线转换：

```bash
conda activate opencode
cd /home/wsh/openhands_code_research
PYTHONPATH=/home/wsh/openhands_code_research/LMCache \
python scripts/08_lmcache_mp/paper_experiment/hardware_io_characterization/11_prepare_raw_skill_kv.py
```

脚本会创建并预分配：

```text
/mnt/990_pro/skill_save_pool/cskcache_raw/qwen3_14b_skill_kv_v1.bin
/mnt/990_pro/skill_save_pool/cskcache_raw/qwen3_14b_skill_kv_build.json
/mnt/990_pro/skill_save_pool/cskcache_raw/qwen3_14b_skill_kv_v1.bin.cskcache-generation.json
/mnt/990_pro/skill_save_pool/cskcache_raw/qwen3_14b_skill_kv_catalog.json
```

每完成一个Skill便执行raw-block checkpoint并原子更新manifest。若进程中断，使用
同一命令重跑；已完成且仍能从raw-block索引恢复的Skill会跳过。任何layout变化、
层数/shape/dtype/position metadata不一致或raw容量不足都会停止，不会静默生成
另一套布局。

前两个文件是LMCache raw-block数据和可恢复的转换checkpoint。后两个文件由
CSKCache管理：generation sidecar用于拒绝旧offset，Catalog保存
container身份及11个Skill的440个最终物理extent。11号脚本只在离线收尾阶段调用
LMCache `entry_offset()`；在线`StorageManager`直接使用已发布extent，不访问
LMCache key索引。已有24 GiB raw文件再次运行11号脚本时不会重写payload，只会
跳过已打包Skill并补齐或验证CSKCache metadata。

发布完成后先运行不含Agent/vLLM的真实性检查：

```bash
conda activate opencode
cd /home/wsh/openhands_code_research
PYTHONPATH=/home/wsh/openhands_code_research/LMCache \
python scripts/08_lmcache_mp/paper_experiment/hardware_io_characterization/13_verify_cskcache_metadata.py
```

13号脚本用`StorageManager`按Catalog extent读取全部11个Skill，每个Skill恰好
提交40层，并逐层对比持久payload SHA-256。任何generation、文件大小、offset、
层完整性或payload错误都会终止；成功结果应为11个Skill、440个extent全部通过。

完成同步真实性检查后，运行异步T0 host-load检查：

```bash
conda activate opencode
cd /home/wsh/openhands_code_research
PYTHONPATH=/home/wsh/openhands_code_research/LMCache \
python scripts/08_lmcache_mp/paper_experiment/hardware_io_characterization/14_verify_async_cskcache_host_load.py
```

`14`同样不启动Agent、vLLM或模型。它在计时前只分配一次40层pinned CPU buffer
并向io_uring注册一次，随后为每个Skill创建真实CSKCache ticket，调用
`submit_host_load()`后立即返回，由后台线程执行一次40层extent batch。脚本等待
`HOST_READY`、验证全部payload SHA，再释放逻辑lease并把同一buffer组交给下一个
Skill。通过条件为：11个Skill、440层全部正确；每个Skill恰好一次物理batch；
pool acquire/release计数相等；结束时没有buffer仍在使用。结果写入：

```text
/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/08_lmcache_mp/
  paper_experiment_hardware_io/a6000_qwen3_14b_io_v1/
  14_cskcache_async_host_load.json
```

其中`submit_ms`只包含T0同步校验、状态登记和后台任务提交；`ready_wait_ms`包含后台
读取等待。脚本不清空Linux page cache，因此这些时间用于验证异步控制流和资源
生命周期，不能作为cold-read带宽。

转换完成后运行SSD→pinned CPU测试：

```bash
conda activate opencode
cd /home/wsh/openhands_code_research
PYTHONPATH=/home/wsh/openhands_code_research/LMCache \
python scripts/08_lmcache_mp/paper_experiment/hardware_io_characterization/12_measure_raw_skill_kv_to_pinned.py
```

`12`在所有计时开始前一次性分配约1.86 GiB的pinned pool，并把40个等长buffer
注册为io_uring fixed buffers；分配和注册时间只写入JSON，不计入每次加载。每个
Skill先warmup 1次，再测10次；一次样本调用一次`load_many_into()`并提交40层I/O。
最后逐层比较SHA-256，任一buffer不一致则整次实验失败。原始结果与轻量产物分别为：

```text
/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/08_lmcache_mp/
  paper_experiment_hardware_io/a6000_qwen3_14b_io_v1/
  12_raw_skill_kv_to_pinned.json

results/problem_exploration/hardware_io_characterization/
  figures/raw_skill_kv_to_pinned.pdf
  figures/raw_skill_kv_to_pinned.png
  tables/raw_skill_kv_to_pinned.csv
```

该结果只回答“一个完整Skill能否通过并发raw I/O更快进入预注册pinned内存”。它
不包含Agent时间窗口、请求定位、GPU H2D、KV page table安装、RoPE校正或残差
纠错，因此不能直接报告端到端TTFT收益。

## direct reuse下的vLLM内部预取窗口

离线Skill manifest schema v4保存开始标签的token序列、token数和SHA-256；完整
Skill span仍由已有`token_count`和`token_ids_sha256`认证。OpenHands只判断本轮
新加载的Skill并把这些manifest元数据随请求B发送给vLLM，不再额外调用
`/tokenize`，也不计算`segment_start/segment_end`。

运行全部13个任务：

```bash
conda activate opencode
cd /home/wsh/openhands_code_research
python scripts/08_lmcache_mp/paper_experiment/hardware_io_characterization/09_run_agent_skill_locator_batch.py
```

每个任务使用独立vLLM和Conversation，模式固定为`direct_reuse`，任务内只运行
两个Agent iteration。实验不再对OpenHands内部阶段分解，只记录：

09是自包含实验入口：它只复用同目录`common.py`中的原子写入、配置路径和离线
manifest底层解析，不加载或执行07的实验模块。任务清单校验、cache对象解析、
fingerprint、JSONL关联和子进程生命周期均由09自身负责，因此07的代码变化不会改变
09的实验语义或触发无关重跑。

```text
T0：vLLM已将请求A的模型输出解析为完整SkillAction JSON
  ↓ 返回OpenHands、执行SkillTool、构造并发送请求B
T1：vLLM chat handler收到已解析的请求B
  ↓ chat template与tokenization
T2：vLLM已生成最终prompt_token_ids
  ↓ Skill span定位与认证、请求构造、EngineCore IPC
T3：EngineCore即将调用scheduler.add_request
```

T0、T1和T2由vLLM API进程记录，T3由vLLM EngineCore进程记录。四个边界全部
使用`time.monotonic_ns()`，并要求Linux boot ID一致。请求B在
`lmcache_segmentia_lookup.source_tool_call_id`中带回请求A的tool-call ID，vLLM以此
唯一关联T0与T1--T3，不根据Skill名称或请求顺序猜测。

正式输出只包含`T0--T1`、`T1--T2`、`T2--T3`和完整`T0--T3`。任何关联
缺失、boot ID不一致、时间逆序、Skill span认证失败或分段和与总窗口不一致，
都会使该case失败。新批次写入
`09_agent_skill_locator_batch/all_task_prompts_v3/`，不复用v1/v2的旧墙上时钟数据。

如果Agent采集已经结束、只需修复或重跑分析关联，不要再次启动vLLM。使用：

```bash
python scripts/08_lmcache_mp/paper_experiment/hardware_io_characterization/09_run_agent_skill_locator_batch.py --analysis-only
```

该模式读取`agent_locator.batch_id`对应的现有13个workspace，按vLLM trace中的
`source_tool_call_id`关联SkillAction，并重新生成`case_result.json`、批量summary、CSV和图。
请求B可能再次生成同名SkillAction；只要该调用没有对应已执行事件，就不会被误当成
本实验的请求A，也不会触发新的vLLM或OpenHands进程。

09同时保留完整13任务图，并生成以下11任务版本：

```text
figures/agent_skill_prefetch_window_vllm_boundaries_typical.pdf
tables/agent_skill_prefetch_window_vllm_boundaries_typical.csv
```

该版本只在绘图阶段剔除`docx`和`writing-systems-papers`；原始case、完整表和完整图
均不删除。若只需要从已发布CSV重画图，不需要重新运行Agent。

## Mermaid Skill细粒度诊断

`09_run_agent_skill_locator_batch.py`能够证明高延迟发生在T0--T1，但不能进一步
确定它属于响应返回、SkillTool还是下一请求组装。运行以下独立诊断：

```bash
conda activate opencode
cd /home/wsh/openhands_code_research
python scripts/08_lmcache_mp/paper_experiment/hardware_io_characterization/10_run_mermaid_skill_timeline.py
```

该脚本固定使用`mermaid-diagram-skill-reuse-pipeline.txt`和
`mermaid-diagram` Skill，独立运行3次。每次都重启vLLM、创建新的OpenHands
workspace，并只执行两个Agent iteration。因此3次运行之间不共享prefix cache、
Conversation、scheduler队列或GPU KV状态。已完成且实现fingerprint未变化的运行会
跳过；实现或配置变化后会自动重跑对应目录。

10不导入07或09，也不通过其他实验脚本间接获得任务、缓存或trace处理逻辑。它只
复用`common.py`中的底层文件/配置工具，并在自身代码中完成CSKCache源对象校验、
JSONL读取、唯一记录关联、fingerprint和launcher子进程管理。因而10的三次诊断不会
随07/09内部实现变化而失效。

细粒度时间线为：

```text
T0  vLLM解析出请求A的完整SkillAction
O1  请求A响应返回OpenHands的LiteLLM transport
O2  OpenHands转换响应并发出ActionEvent
O3  OpenHands开始执行SkillTool
O4  SkillTool读取SKILL.md及资源清单后返回
O5  OpenHands构造完ObservationEvent
O6  ObservationEvent经过Visualizer与实验callback
O7  请求B进入OpenHands的LLM transport wrapper
O8  OpenHands附加Segmentia lookup元数据并调用原transport
T1  请求B到达vLLM chat handler
T2  vLLM完成chat template和tokenization
T3  EngineCore即将调用scheduler.add_request
```

所有时间点使用同一台Linux主机的`time.monotonic_ns()`，并验证boot ID一致；
请求A、SkillTool和请求B通过同一个`tool_call_id`关联。实验明确不把T3之后的
LMCache external lookup、KV retrieve、H2D或GPU page安装计入这些区间。
脚本仅在这3次运行中设置`SEGMENTIA_FINE_TIMELINE=1`；普通09批次不启用SDK
Skill执行探针，因此不会引入额外的JSONL写入。

原始结果写入：

```text
/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/08_lmcache_mp/
  paper_experiment_hardware_io/a6000_qwen3_14b_io_v1/
  10_mermaid_skill_timeline/mermaid_skill_fine_timeline_v1/
```

三个运行全部完成后发布：

```text
results/problem_exploration/hardware_io_characterization/
  figures/mermaid_skill_t0_t3_fine_timeline.pdf
  figures/mermaid_skill_t0_t3_fine_timeline.png
  tables/mermaid_skill_t0_t3_fine_timeline.csv
  data/mermaid_skill_timeline_run_pointer.txt
```

The five hardware measurements are independent. Re-running one script atomically
replaces only that script's JSON result in the configured run directory. Run
`06_analyze_results.py` after the desired hardware measurements finish. The
legacy `07_agent_skill_schedule_window.py` can still read one old-style Agent
event file and scheduler trace for tail-latency diagnosis, but its start point
is later than the formal T0 and it is not used by the batch experiment.

`04_pcie_h2d.py` executes pageable-sync, pinned-sync, and pinned-async in a
deterministic cyclic order. Each warm-up or measured round runs all three modes,
but rotates which mode is first, so GPU/PCIe power-state warm-up is not always
credited to the mode that happens to run last. The modes remain serial rather
than concurrent, and each JSON row still contains 20 samples for one mode.

Raw output:

```text
/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/08_lmcache_mp/
  paper_experiment_hardware_io/<run.id>/
```

Lightweight paper artifacts:

```text
results/problem_exploration/hardware_io_characterization/
```

`02_ssd_to_cpu.py` uses per-file `POSIX_FADV_DONTNEED` for the cold condition;
it does not write to `/proc/sys/vm/drop_caches`. The complete-path script uses
one pinned layer buffer and one GPU layer buffer, so it measures the serial
SSD→pinned→GPU baseline without allocating a 1.3-GiB staging object.

`03_cpu_memory.py` requires the system `libnuma` library. It allocates pageable
buffers with an explicit node policy and queries every page with `move_pages`
before and after timing. A placement mismatch aborts the run before the result
JSON is replaced; CPU affinity is restored and NUMA allocations are freed on
all exit paths.
