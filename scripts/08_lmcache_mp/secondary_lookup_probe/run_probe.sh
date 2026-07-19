#!/usr/bin/env bash
# Real single-request validation for vLLM APC requeue + LMCache segment lookup.
# Probe-only is the default; --apply-external-kv explicitly enables phase 2B.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${VLLM_MODEL_PATH:-/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B}"
SERVED_MODEL="${VLLM_SERVED_NAME:-Qwen3}"
PORT="${VLLM_PORT:-8100}"
API_KEY="${VLLM_API_KEY:-EMPTY}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
BLOCK_SIZE="${VLLM_BLOCK_SIZE:-16}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
READY_ATTEMPTS="${VLLM_READY_MAX_ATTEMPTS:-450}"
READY_INTERVAL="${VLLM_READY_INTERVAL:-2}"
SERVER_PID=""

PREPARE_ONLY=0
APPLY_EXTERNAL_KV=0
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --prepare-only) PREPARE_ONLY=1 ;;
    --apply-external-kv) APPLY_EXTERNAL_KV=1 ;;
    *) echo "usage: $0 [--prepare-only] [--apply-external-kv]" >&2; exit 2 ;;
  esac
  shift
done

if [[ "$APPLY_EXTERNAL_KV" == "1" ]]; then
  RUN_KIND="secondary_lookup_apply_runs"
else
  RUN_KIND="secondary_lookup_runs"
fi
RUN_DIR="${LMCACHE_SECONDARY_RUN_DIR:-/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/08_lmcache_mp/$RUN_KIND/$RUN_ID}"
VLLM_LOG="$RUN_DIR/vllm.log"

mkdir -p "$RUN_DIR"

export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-256}"
export LMCACHE_ENABLE_BLENDING="${LMCACHE_ENABLE_BLENDING:-True}"
export LMCACHE_BLEND_SPECIAL_STR="${LMCACHE_BLEND_SPECIAL_STR:-<|fim_pad|><|repo_name|>}"
export LMCACHE_USE_LAYERWISE="${LMCACHE_USE_LAYERWISE:-True}"
export LMCACHE_BLEND_CHECK_LAYERS="${LMCACHE_BLEND_CHECK_LAYERS:-1}"
export LMCACHE_BLEND_RECOMPUTE_RATIOS="${LMCACHE_BLEND_RECOMPUTE_RATIOS:-0.15}"
export LMCACHE_LOCAL_CPU="${LMCACHE_LOCAL_CPU:-True}"
export LMCACHE_MAX_LOCAL_CPU_SIZE="${LMCACHE_MAX_LOCAL_CPU_SIZE:-5}"
export LMCACHE_LOCAL_DISK="file://${RUN_DIR}/lmcache_disk/"
export LMCACHE_MAX_LOCAL_DISK_SIZE="${LMCACHE_MAX_LOCAL_DISK_SIZE:-10}"
export LMCACHE_LOG_LEVEL="${LMCACHE_LOG_LEVEL:-INFO}"
mkdir -p "$RUN_DIR/lmcache_disk"

PYTHONPATH_VALUE="$ROOT/vllm:$ROOT/LMCache${PYTHONPATH:+:$PYTHONPATH}"
DRIVER_ARGS=(
  --model-path "$MODEL_PATH"
  --served-model "$SERVED_MODEL"
  --port "$PORT"
  --api-key "$API_KEY"
  --run-id "$RUN_ID"
  --run-dir "$RUN_DIR"
  --vllm-log "$VLLM_LOG"
  --blend-special-str "$LMCACHE_BLEND_SPECIAL_STR"
)
if [[ "$APPLY_EXTERNAL_KV" == "1" ]]; then
  DRIVER_ARGS+=(--apply-external-kv)
fi

if [[ "$PREPARE_ONLY" == "1" ]]; then
  PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$SCRIPT_DIR/run_probe.py" \
    "${DRIVER_ARGS[@]}" --prepare-only
  echo "[prepared] run_dir=$RUN_DIR"
  exit 0
fi

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap cleanup EXIT INT TERM

wait_vllm_ready() {
  local attempt=0
  local code="000"
  while ((attempt < READY_ATTEMPTS)); do
    if [[ -n "$SERVER_PID" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "[error] vLLM exited before readiness" >&2
      return 1
    fi
    code="$(curl -sS -o /dev/null -w '%{http_code}' \
      --connect-timeout 3 --max-time 10 \
      -H "Authorization: Bearer $API_KEY" \
      "http://127.0.0.1:$PORT/v1/models" 2>/dev/null || true)"
    if [[ "$code" == "200" ]]; then
      return 0
    fi
    attempt=$((attempt + 1))
    sleep "$READY_INTERVAL"
  done
  echo "[error] vLLM readiness timeout port=$PORT" >&2
  return 1
}

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
echo "[vLLM] fresh boundary run_id=$RUN_ID mode=$RUN_KIND cache=$RUN_DIR/lmcache_disk"
PYTHONPATH="$PYTHONPATH_VALUE" vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL" \
  --api-key "$API_KEY" \
  --port "$PORT" \
  --dtype auto \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --block-size "$BLOCK_SIZE" \
  --enable-prefix-caching \
  --no-async-scheduling \
  --enforce-eager \
  --no-enable-log-requests \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
  >"$VLLM_LOG" 2>&1 &
SERVER_PID=$!

wait_vllm_ready
set +e
PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$SCRIPT_DIR/run_probe.py" \
  "${DRIVER_ARGS[@]}"
STATUS=$?
set -e
cleanup

if [[ "$STATUS" -ne 0 ]]; then
  echo "[no-go] inspect $RUN_DIR/summary.json and $VLLM_LOG" >&2
  exit "$STATUS"
fi
echo "[go] run_dir=$RUN_DIR"
