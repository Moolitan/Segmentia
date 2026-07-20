#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$ROOT/scripts/08_lmcache_mp/configurable_secondary_lookup"
CONFIG="${1:-$SCRIPT_DIR/cases.json}"
MODEL_PATH="${VLLM_MODEL_PATH:-/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B}"
SERVED_MODEL="${VLLM_SERVED_NAME:-Qwen3}"
PORT="${VLLM_PORT:-8100}"
API_KEY="${VLLM_API_KEY:-EMPTY}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
SHUTDOWN_TIMEOUT="${VLLM_SHUTDOWN_TIMEOUT:-30}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${SEGMENTIA_CONFIGURED_RUN_DIR:-/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/08_lmcache_mp/configurable_secondary_lookup/$RUN_ID}"
PYTHONPATH_VALUE="$ROOT/vllm:$ROOT/LMCache${PYTHONPATH:+:$PYTHONPATH}"
SERVER_PID=""

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

mkdir -p "$RUN_DIR/lmcache_disk"
export LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-256}"
export LMCACHE_ENABLE_BLENDING="${LMCACHE_ENABLE_BLENDING:-True}"
export LMCACHE_BLEND_SPECIAL_STR="${LMCACHE_BLEND_SPECIAL_STR:-<|fim_pad|><|repo_name|>}"
export LMCACHE_USE_LAYERWISE="${LMCACHE_USE_LAYERWISE:-True}"
export LMCACHE_BLEND_CHECK_LAYERS="${LMCACHE_BLEND_CHECK_LAYERS:-1}"
export LMCACHE_BLEND_RECOMPUTE_RATIOS="${LMCACHE_BLEND_RECOMPUTE_RATIOS:-0.15}"
export LMCACHE_LOCAL_CPU="${LMCACHE_LOCAL_CPU:-True}"
export LMCACHE_MAX_LOCAL_CPU_SIZE="${LMCACHE_MAX_LOCAL_CPU_SIZE:-5}"
export LMCACHE_LOCAL_DISK="file://$RUN_DIR/lmcache_disk/"
export LMCACHE_MAX_LOCAL_DISK_SIZE="${LMCACHE_MAX_LOCAL_DISK_SIZE:-50}"

PYTHONPATH="$PYTHONPATH_VALUE" vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL" \
  --api-key "$API_KEY" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --shutdown-timeout "$SHUTDOWN_TIMEOUT" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --enable-prefix-caching \
  --no-async-scheduling \
  --enforce-eager \
  --no-enable-log-requests \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
  >"$RUN_DIR/vllm.log" 2>&1 &
SERVER_PID=$!

READY=0
for _ in $(seq 1 450); do
  if curl -fsS -H "Authorization: Bearer $API_KEY" \
    "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    READY=1
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "vLLM exited before readiness; inspect $RUN_DIR/vllm.log" >&2
    exit 1
  fi
  sleep 2
done
if [[ "$READY" != "1" ]]; then
  echo "vLLM readiness timeout; inspect $RUN_DIR/vllm.log" >&2
  exit 1
fi

env -u LD_LIBRARY_PATH PYTHONPATH="$PYTHONPATH_VALUE" python "$SCRIPT_DIR/run_configured.py" \
  --config "$CONFIG" \
  --output-dir "$RUN_DIR" \
  --base-url "http://127.0.0.1:$PORT" \
  --api-key "$API_KEY" \
  --model "$SERVED_MODEL" \
  --separator-token-id "${SEGMENTIA_SEPARATOR_TOKEN_ID:-151663}"

echo "[done] run_dir=$RUN_DIR"
