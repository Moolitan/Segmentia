# CSKCache real-Agent latency experiment

本目录只比较当前系统的两个模式：

- `recompute`：真实 OpenHands Agent 正常计算请求 B；
- `cskcache`：同一 Agent 通过当前 CSKCache raw-block、pinned CPU、逐层 H2D 和 K 纠错路径处理请求 B。

实验使用 `doc-coauthoring-retry-design-doc.txt` 与 3314-token
`doc-coauthoring` Skill。请求 A 由模型真实生成 `SkillAction`，OpenHands
真实执行 `SkillTool`，请求 B 携带真实 Skill Observation。每个模式启动独立
vLLM；每个 case 之前清空 vLLM prefix cache，但请求 A 与请求 B 之间不清空，
从而保留真实 Agent 两请求执行语义。

直接运行：

```bash
conda activate opencode
bash scripts/08_lmcache_mp/paper/paper_experiment/latency/run.sh
```

所有参数集中在 `config.py`，命令行不接收参数。默认执行 3 个独立 server
replica；每个 `(replica, mode)` 包含 1 个 warmup 和 5 个测量 case。模式顺序
在 replica 间轮换。

主指标是 request B 从 vLLM API 收到请求到第一个输出 token 的 TTFT。辅助
时间戳覆盖：客户端发送/收包、vLLM render+tokenize、scheduler admission、首
token、API 响应，以及 CSKCache 的 SSD→pinned、H2D、K 纠错与 commit。
renderer 当前把 chat template 和 tokenizer 封装在一次调用里，因此该区间被
诚实记录为合并指标，不人为拆分。

GPU侧的逐层事件共享一个在CSKCache executor开始时建立的CUDA Event时间
原点。LMCache的load stream记录每层H2D区间，默认compute stream记录每层K
纠错与KV commit区间。分析器要求三组事件均恰好覆盖Layer 0–39且无重复，随后
直接计算`H2D(i+1)`与`Correction/Commit(i)`的时间交集，不依据
`non_blocking=True`推测重叠。第一层H2D和最后一层计算没有相邻流水伙伴，不计
入可重叠容量。

原始数据写入：

```text
/mnt/Large_Language_Model_Lab_1/wsh/CSKCache/output/08_lmcache_mp/
  end_to_end_latency/<run_id>/
```

轻量汇总写入：

```text
results/problem_exploration/cskcache_end_to_end_latency/
```

其中新增：

```text
cskcache_layerwise_gpu_pipeline.pdf/png
cskcache_layerwise_gpu_overlap.csv
```

流水线图选择GPU流水总跨度最接近样本中位数的一个真实请求，三条lane分别展示
H2D、K correction和KV commit；矩形内数字为模型层号。CSV保留每个请求的实际
相邻层重叠时长、可重叠容量、重叠比例、串行组件总时间、区间并集时间和并发
节省时间。

分析器会拒绝以下无效样本：请求 B 缺失、公共时间点缺失、跨模式 prompt token
数不同，或 CSKCache 未完成 T0、host read、认证、reuse plan、scheduler 激活、
H2D、纠错和 commit 中的任一阶段。
旧run没有共享CUDA时间原点，分析器会明确要求重跑，不会把旧的独立耗时相加或
拼接成流水线图。
