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

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
export PYTHONPATH="$CSKCACHE_ROOT:$VLLM_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export CSKCACHE_DISK_DIR="$OUTPUT_DIR"
export CSKCACHE_CPU_MAX_BYTES=0
export CSKCACHE_CAPTURE_ONLY=1
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

KV_TRANSFER_CONFIG='{"kv_connector":"CSKCacheConnectorV1","kv_connector_module_path":"cskcache.integration.vllm.v1_connector","kv_role":"kv_both"}'

cd "$VLLM_ROOT"
SERVER_PID=""

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
  VLLM_PORT="$PORT" bash "$ROOT/scripts/vllm_stop.sh" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

OVERWRITE=0
for arg in "$@"; do
  if [[ "$arg" == "--overwrite" ]]; then
    OVERWRITE=1
  fi
done

failures=0
for skill_file in "$ROOT"/skills/*/SKILL.md; do
  skill="$(basename "$(dirname "$skill_file")")"
  digest="$(printf '%s' "$skill" | sha256sum | cut -c1-32)"
  if [[ "$OVERWRITE" == "0" && -f "$OUTPUT_DIR/$digest.pt" && -f "$OUTPUT_DIR/$digest.json" ]]; then
    echo "[skipped_existing] $skill"
    continue
  fi

  LOG_FILE="$LOG_DIR/07_cskcache_offline_prefill_${skill}.log"
  echo "[start] skill=$skill port=$PORT log=$LOG_FILE"
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

  cd "$ROOT"
  set +e
  python scripts/07_cskcache/offline_prefill_skills.py \
    --skill "$skill" \
    --base-url "http://127.0.0.1:$PORT" \
    --api-key "$API_KEY" \
    --model-path "$MODEL_PATH" \
    --served-model "$SERVED_MODEL" \
    --output-dir "$OUTPUT_DIR" \
    "$@"
  status=$?
  set -e

  cleanup
  if [[ "$status" != "0" ]]; then
    failures=$((failures + 1))
    echo "[failed] skill=$skill status=$status"
  else
    echo "[stopped] skill=$skill"
  fi
done

if [[ "$failures" != "0" ]]; then
  echo "$failures skill(s) failed"
  exit 1
fi
