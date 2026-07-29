# Fixed Prefix-256 tri-level latency benchmark

本实验只测固定前缀256重计算路径的系统时延和Skill长度break-even，不评估Agent action一致性。reuse arm使用真实三级缓存：服务启动从SSD恢复metadata，请求准入且通过短Skill gate后，在独立probe RPC路径把完整Skill KV提前提升到LMCache CPU hot cache；请求计算`A`到达`P`后转移CPU pin lease，由worker逐层异步执行CPU→GPU。若提升尚未完成，请求在`P`保留已分配blocks等待；失败则从`P`本地计算。所有长时间vLLM实验由用户在`opencode`环境运行。

## 先运行sanity

```bash
conda activate opencode
RUN_ID=20260729-fixed-prefix-cpu-prefetch-sanity-v1 \
MPLCONFIGDIR=/tmp/segmentia-mpl \
bash scripts/08_lmcache_mp/fixed_prefix_latency/run_sanity.sh
```

Sanity包含一个service replica、`512/768/1536/3301`四个长度、每长度1次warmup和5次measurement。脚本先构建combined SSD，再依次运行四个arm；每个完整leaf发布后可断点续跑。同一`RUN_ID`重跑会校验并跳过有效leaf，只补未完成部分。

## 正式矩阵

Sanity控制面通过后，使用新的RUN_ID运行：

```bash
RUN_ID=20260729-fixed-prefix-latency-v1 \
MPLCONFIGDIR=/tmp/segmentia-mpl \
bash scripts/08_lmcache_mp/fixed_prefix_latency/run_benchmark.sh
```

默认正式矩阵为3个service replicas、10个Skill长度（`512/640/768/1024/1280/1536/1792/2048/2560/3301`）、每长度2次warmup和10次measurement。原始输出写入外存`fixed_prefix_latency_runs/<RUN_ID>/`；轻量表格、图和summary写入`results/problem_exploration/fixed_prefix_latency_break_even/`。`cpu_pipeline_by_request.csv`另外记录首请求SSD→CPU提升、到P等待、CPU读取和纯H2D时间。

若GPU运行已经完成，只需重新分析：

```bash
RUN_ID=<原RUN_ID> RUN_SOURCE=0 RUN_MEASURE=0 RUN_ANALYSIS=1 \
MPLCONFIGDIR=/tmp/segmentia-mpl \
bash scripts/08_lmcache_mp/fixed_prefix_latency/run_benchmark.sh
```

`dry_run.sh`只构造合成请求并验证输入矩阵，不启动服务。
