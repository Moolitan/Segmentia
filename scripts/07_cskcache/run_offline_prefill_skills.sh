#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CSKCACHE_ROOT="$ROOT/CSKCache"
VLLM_ROOT="${VLLM_ROOT:-/home/wsh/vllm}"
MODEL_PATH="${VLLM_MODEL_PATH:-/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B}"
SERVED_MODEL="${VLLM_SERVED_NAME:-Qwen3}"
PORT="${VLLM_PORT:-8013}"
API_KEY="${VLLM_API_KEY:-EMPTY}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.9}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
OUTPUT_DIR="${CSKCACHE_OUTPUT_DIR:-/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/07_cskcache/offline_skill_kv}"
LOG_DIR="${VLLM_LOG_DIR:-$ROOT/log}"
LOG_FILE="$LOG_DIR/07_cskcache_offline_prefill_vllm.log"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
export PYTHONPATH="$CSKCACHE_ROOT:$VLLM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CSKCACHE_KV_DIR="$OUTPUT_DIR"
export CSKCACHE_DISK_DIR="$OUTPUT_DIR"
export CSKCACHE_CPU_MAX_BYTES=0
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

KV_TRANSFER_CONFIG='{"kv_connector":"CSKCacheConnectorV1","kv_connector_module_path":"cskcache.integration.vllm.v1_connector","kv_role":"kv_both"}'

cd "$VLLM_ROOT"
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL" \
  --api-key "$API_KEY" \
  --port "$PORT" \
  --dtype auto \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --no-enable-prefix-caching \
  --no-enable-log-requests \
  --kv-transfer-config "$KV_TRANSFER_CONFIG" \
  >"$LOG_FILE" 2>&1 &
SERVER_PID=$!

cleanup() {
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "$ROOT"
python scripts/07_cskcache/offline_prefill_skills.py \
  --base-url "http://127.0.0.1:$PORT" \
  --api-key "$API_KEY" \
  --model-path "$MODEL_PATH" \
  --served-model "$SERVED_MODEL" \
  --output-dir "$OUTPUT_DIR" \
  "$@"
