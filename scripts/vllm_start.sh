#!/usr/bin/env bash
set -euo pipefail
#  wtz
VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B}"

# VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-/mnt/Large_Language_Model_Lab_1/llm_models/Qwen-Qwen3-Coder-30B-A3B-Instruct}"
# VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-/mnt/Large_Language_Model_Lab_1/llm_models/Qwen-Qwen2.5-7B-Instruct}"
VLLM_SERVED_NAME="${VLLM_SERVED_NAME:-Qwen3}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.9}"
VLLM_LOG_DIR="${VLLM_LOG_DIR:-/home/wsh/openhands_code_research/log}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
# Prefix caching: 1 = on (default). Set PREFIX_CACHING=0 for ContextSegmentKV
# reuse experiments so a cached prefix can't silently cover the skill span and
# skip the injection (which would make rope behave like recompute).
PREFIX_CACHING="${PREFIX_CACHING:-1}"

# # # conda 环境的 libstdc++ 包含 vLLM 所需的 CXXABI_1.3.15
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "$VLLM_LOG_DIR"

cd /home/wsh/vllm

EXTRA_ARGS=()
if [[ "$VLLM_ENFORCE_EAGER" == "1" ]]; then
  EXTRA_ARGS+=(--enforce-eager)
fi
if [[ "$PREFIX_CACHING" == "1" ]]; then
  EXTRA_ARGS+=(--enable-prefix-caching)
else
  EXTRA_ARGS+=(--no-enable-prefix-caching)
fi

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

# wtz
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

# wsh
python -m vllm.entrypoints.openai.api_server \
  --model "$VLLM_MODEL_PATH" \
  --served-model-name "$VLLM_SERVED_NAME" \
  --no-enable-log-requests \
  --dtype auto \
  --api-key "$VLLM_API_KEY" \
  --port "$VLLM_PORT" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  --max-model-len "$VLLM_MAX_MODEL_LEN" \
  --gpu-memory-utilization "$VLLM_GPU_UTIL" \
  "${EXTRA_ARGS[@]}" \
  > "$VLLM_LOG_DIR/vllm.log" 2>&1 &

echo "vLLM started (PID $!, port $VLLM_PORT). Log: $VLLM_LOG_DIR/vllm.log (prefix_caching=$PREFIX_CACHING)"
