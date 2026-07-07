#!/usr/bin/env bash
set -euo pipefail

# 测试 Qwen3-Coder-30B-A3B-Instruct 的离线吞吐量，
# 将 30GB 权重放到 CPU 内存（vLLM --cpu-offload-gb / cpu_offload_gb=30）。
#
# 用法:
#   bash scripts/throughput_cpu_offload.sh
# 可用环境变量覆盖:
#   MODEL_PATH / CPU_OFFLOAD_GB / NUM_PROMPTS / INPUT_LEN / OUTPUT_LEN
#   GPU_UTIL / MAX_MODEL_LEN / DTYPE

MODEL_PATH="${MODEL_PATH:-/mnt/Large_Language_Model_Lab_1/llm_models/Qwen-Qwen3-Coder-30B-A3B-Instruct}"
CPU_OFFLOAD_GB="${CPU_OFFLOAD_GB:-30}"
NUM_PROMPTS="${NUM_PROMPTS:-64}"
INPUT_LEN="${INPUT_LEN:-512}"
OUTPUT_LEN="${OUTPUT_LEN:-256}"
GPU_UTIL="${GPU_UTIL:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
DTYPE="${DTYPE:-auto}"
LOG_DIR="${LOG_DIR:-/home/wsh/openhands_code_research/log}"

# conda 环境的 libstdc++ 包含 vLLM 所需的 CXXABI（与 vllm_start.sh 一致）
export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "$LOG_DIR"

echo "===== vLLM CPU-offload 吞吐量测试 ====="
echo "model          : $MODEL_PATH"
echo "cpu_offload_gb : $CPU_OFFLOAD_GB"
echo "num_prompts    : $NUM_PROMPTS"
echo "input/output   : $INPUT_LEN / $OUTPUT_LEN tokens"
echo "gpu_util       : $GPU_UTIL"
echo "max_model_len  : $MAX_MODEL_LEN"
echo "======================================"

export MODEL_PATH CPU_OFFLOAD_GB NUM_PROMPTS INPUT_LEN OUTPUT_LEN \
       GPU_UTIL MAX_MODEL_LEN DTYPE

python - <<'PY' 2>&1 | tee "$LOG_DIR/throughput_cpu_offload.log"
import os, time, random
from vllm import LLM, SamplingParams

model      = os.environ["MODEL_PATH"]
offload_gb = float(os.environ["CPU_OFFLOAD_GB"])
n          = int(os.environ["NUM_PROMPTS"])
in_len     = int(os.environ["INPUT_LEN"])
out_len    = int(os.environ["OUTPUT_LEN"])
gpu_util   = float(os.environ["GPU_UTIL"])
max_len    = int(os.environ["MAX_MODEL_LEN"])
dtype      = os.environ["DTYPE"]

print(f"[load] 加载模型，{offload_gb} GB 权重放到 CPU 内存 ...", flush=True)
t_load = time.perf_counter()
llm = LLM(
    model=model,
    cpu_offload_gb=offload_gb,
    gpu_memory_utilization=gpu_util,
    max_model_len=max_len,
    dtype=dtype,
    trust_remote_code=True,
    enforce_eager=False,
)
print(f"[load] 完成，用时 {time.perf_counter() - t_load:.1f}s", flush=True)

# 用模型自身 tokenizer 构造固定长度的 token 提示，保证输入长度可控
tok = llm.get_tokenizer()
vocab = tok.vocab_size
rng = random.Random(0)
prompts = [
    {"prompt_token_ids": [rng.randint(0, vocab - 1) for _ in range(in_len)]}
    for _ in range(n)
]

sp = SamplingParams(temperature=0.0, max_tokens=out_len, ignore_eos=True)

# 预热（一条），排除编译/首次内核开销
print("[warmup] 预热 ...", flush=True)
llm.generate([prompts[0]], sp, use_tqdm=False)

print("[run] 开始计时 ...", flush=True)
t0 = time.perf_counter()
outs = llm.generate(prompts, sp, use_tqdm=True)
elapsed = time.perf_counter() - t0

n_in  = sum(len(o.prompt_token_ids) for o in outs)
n_out = sum(len(o.outputs[0].token_ids) for o in outs)
n_tot = n_in + n_out

print("\n===== 结果 =====")
print(f"请求数           : {len(outs)}")
print(f"耗时             : {elapsed:.2f} s")
print(f"输入 tokens      : {n_in}")
print(f"生成 tokens      : {n_out}")
print(f"请求吞吐         : {len(outs)/elapsed:.2f} req/s")
print(f"生成吞吐         : {n_out/elapsed:.1f} tok/s (output)")
print(f"总吞吐           : {n_tot/elapsed:.1f} tok/s (in+out)")
PY

echo "日志已保存: $LOG_DIR/throughput_cpu_offload.log"
