#!/usr/bin/env bash
set -euo pipefail

VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B}"
VLLM_SERVED_NAME="${VLLM_SERVED_NAME:-Qwen3}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.9}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"
VLLM_START_GATE="${VLLM_START_GATE:-}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
    echo "[ERROR] CONDA_PREFIX 未设置，请先激活 vLLM 环境" >&2
    exit 1
fi

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cd /home/wsh/vllm

EXTRA_ARGS=()

if [[ "$VLLM_ENFORCE_EAGER" == "1" ]]; then
    EXTRA_ARGS+=(--enforce-eager)
fi

if [[ -n "$VLLM_START_GATE" ]]; then
    echo "[vLLM] 等待启动门: $VLLM_START_GATE"

    while [[ ! -e "$VLLM_START_GATE" ]]; do
        sleep 0.1
    done

    echo "[vLLM] 启动门已打开"
fi

echo "[vLLM] model: $VLLM_MODEL_PATH"
echo "[vLLM] served name: $VLLM_SERVED_NAME"
echo "[vLLM] port: $VLLM_PORT"
echo "[vLLM] max model len: $VLLM_MAX_MODEL_LEN"
echo "[vLLM] GPU utilization: $VLLM_GPU_UTIL"

exec python -m vllm.entrypoints.openai.api_server \
    --model "$VLLM_MODEL_PATH" \
    --served-model-name "$VLLM_SERVED_NAME" \
    --enable-prefix-caching \
    --no-enable-log-requests \
    --dtype auto \
    --api-key "$VLLM_API_KEY" \
    --port "$VLLM_PORT" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --reasoning-parser qwen3 \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --gpu-memory-utilization "$VLLM_GPU_UTIL" \
    "${EXTRA_ARGS[@]}"
