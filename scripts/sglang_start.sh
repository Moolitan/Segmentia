#!/usr/bin/env bash
set -euo pipefail

SGLANG_MODEL_PATH="${SGLANG_MODEL_PATH:-/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B}"
# VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-/mnt/Large_Language_Model_Lab_1/llm_models/Qwen-Qwen2.5-7B-Instruct}"
SGLANG_SERVED_NAME="${SGLANG_SERVED_NAME:-Qwen3}"
SGLANG_PORT="${SGLANG_PORT:-8000}"
SGLANG_API_KEY="${SGLANG_API_KEY:-EMPTY}"
SGLANG_MAX_MODEL_LEN="${SGLANG_MAX_MODEL_LEN:-32768}"
SGLANG_GPU_UTIL="${SGLANG_GPU_UTIL:-0.75}"
SGLANG_LOG_DIR="${SGLANG_LOG_DIR:-/home/wsh/openhands_code_research/log}"

# # # conda 环境的 libstdc++ 包含 vLLM 所需的 CXXABI_1.3.15


mkdir -p "$SGLANG_LOG_DIR"

cd /home/wsh/sglang

export NVCC_PREPEND_FLAGS=""

source /home/wsh/miniconda3/etc/profile.d/conda.sh
conda activate sglang
# export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

export CUDA_HOME="/usr/local/cuda-12.8"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
# export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LIBRARY_PATH="/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}"


# python -m vllm.entrypoints.openai.api_server \
#   --model "$VLLM_MODEL_PATH" \
#   --served-model-name "$VLLM_SERVED_NAME" \
#   --enable-prefix-caching \
#   --cpu-offload-gb 20 \
#   --no-enable-log-requests \
#   --dtype auto \
#   --api-key "$VLLM_API_KEY" \
#   --port "$VLLM_PORT" \
#   --tool-call-parser qwen3_xml \
#   --max-model-len "$VLLM_MAX_MODEL_LEN" \
#   --gpu-memory-utilization "$VLLM_GPU_UTIL" \
#   --enable-auto-tool-choice \
#   > "$VLLM_LOG_DIR/vllm.log" 2>&1 &
python -m sglang.launch_server \
  --model-path "$SGLANG_MODEL_PATH" \
  --served-model-name "$SGLANG_SERVED_NAME" \
  --dtype float16 \
  --api-key "$SGLANG_API_KEY" \
  --port "$SGLANG_PORT" \
  --context-length "$SGLANG_MAX_MODEL_LEN" \
  --mem-fraction-static "$SGLANG_GPU_UTIL" \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_coder \
  --max-running-requests 64 \
  --log-requests-level 0 \
  --cuda-graph-max-bs 16 \
  > "$SGLANG_LOG_DIR/sglang.log" 2>&1 &

# python -m vllm.entrypoints.openai.api_server \
#   --model "$VLLM_MODEL_PATH" \
#   --served-model-name "$VLLM_SERVED_NAME" \
#   --enable-prefix-caching \
#   --no-enable-log-requests \
#   --dtype auto \
#   --api-key "$VLLM_API_KEY" \
#   --port "$VLLM_PORT" \
#   --enable-auto-tool-choice \
#   --tool-call-parser hermes \
#   --reasoning-parser qwen3 \
#   --max-model-len "$VLLM_MAX_MODEL_LEN" \
#   --gpu-memory-utilization "$VLLM_GPU_UTIL" \
#   > "$VLLM_LOG_DIR/vllm.log" 2>&1 &

echo "SGLANG started (PID $!, port $SGLANG_PORT). Log: $SGLANG_LOG_DIR/sglang.log"

