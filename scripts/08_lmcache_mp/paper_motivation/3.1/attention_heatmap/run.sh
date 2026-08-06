#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
SCRIPT_DIR="$ROOT/scripts/08_lmcache_mp/paper_motivation/3.1/attention_heatmap"
VLLM_ROOT="$ROOT/vllm"
LMCACHE_ROOT="$ROOT/LMCache"
PYTHON_BIN="${PYTHON_BIN:-python}"
VLLM_BIN="${VLLM_BIN:-vllm}"
MODEL_PATH="${VLLM_MODEL_PATH:-/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B}"
SERVED_MODEL="${VLLM_SERVED_NAME:-Qwen3}"
SKILL_PATH="${SKILL_PATH:-$ROOT/skills/internal-comms/SKILL.md}"
POOL_DIR="${SKILL_SAVE_POOL_DIR:-/mnt/Large_Language_Model_Lab_1/wsh/skill_save_pool/Qwen3-14B}"
OUTPUT_ROOT="${SEGMENTIA_ATTENTION_OUTPUT_ROOT:-/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/08_lmcache_mp/paper_motivation_3_1_attention_heatmap}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
SPEC_PATH="$RUN_DIR/current_spec.json"
PORT="${VLLM_PORT:-8014}"
API_KEY="${VLLM_API_KEY:-EMPTY}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.9}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
PYTHONPATH_VALUE="$SCRIPT_DIR:$VLLM_ROOT:$LMCACHE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "opencode" ]]; then
  echo "[error] activate conda environment opencode first" >&2
  exit 2
fi
if [[ ! -f "$SKILL_PATH" ]]; then
  echo "[error] missing Skill: $SKILL_PATH" >&2
  exit 2
fi
if [[ ! -d "$POOL_DIR" ]]; then
  echo "[error] missing offline Skill pool: $POOL_DIR" >&2
  exit 2
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-256}"
export LMCACHE_USE_LAYERWISE=True
export LMCACHE_LOCAL_CPU=False
export LMCACHE_MAX_LOCAL_CPU_SIZE="${LMCACHE_MAX_LOCAL_CPU_SIZE:-5}"
export LMCACHE_MAX_LOCAL_DISK_SIZE="${LMCACHE_MAX_LOCAL_DISK_SIZE:-1000}"
export LMCACHE_FORCE_SKIP_SAVE=1
export SEGMENTIA_ATTENTION_HEATMAP_SPEC="$SPEC_PATH"

mkdir -p "$RUN_DIR"
export MPLCONFIGDIR="$RUN_DIR/.matplotlib"
SERVER_PID=""

stop_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap stop_server EXIT INT TERM

wait_ready() {
  local ready=0
  for _poll_i in $(seq 1 450); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "[error] vLLM exited before readiness" >&2
      return 1
    fi
    local code
    code="$(curl -sS -o /dev/null -w '%{http_code}' \
      --connect-timeout 3 --max-time 10 \
      -H "Authorization: Bearer $API_KEY" \
      "http://127.0.0.1:$PORT/v1/models" 2>/dev/null || true)"
    if [[ "$code" == "200" ]]; then
      ready=1
      break
    fi
    sleep 2
  done
  if [[ "$ready" != "1" ]]; then
    echo "[error] vLLM readiness timeout" >&2
    return 1
  fi
}

run_mode() {
  local mode="$1"
  local mode_dir="$RUN_DIR/$mode"
  local server_log="$mode_dir/vllm.log"
  mkdir -p "$mode_dir"
  export SEGMENTIA_ATTENTION_HEATMAP_OUT_DIR="$mode_dir"

  if [[ "$mode" == "recompute" ]]; then
    export LMCACHE_ENABLE_SEGMENTIA=False
    export LMCACHE_EXTRA_CONFIG='{}'
    export LMCACHE_LOCAL_DISK_REHYDRATE=False
    export LMCACHE_LOCAL_DISK="file://$mode_dir/lmcache"
  else
    export LMCACHE_ENABLE_SEGMENTIA=True
    export LMCACHE_EXTRA_CONFIG='{"local_disk_rehydrate_recursive":true,"segmentia_direct_reuse":true}'
    export LMCACHE_LOCAL_DISK_REHYDRATE=True
    export LMCACHE_LOCAL_DISK="file://$POOL_DIR/"
  fi

  echo "[server] start mode=$mode log=$server_log"
  PYTHONPATH="$PYTHONPATH_VALUE" "$VLLM_BIN" serve "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL" \
    --api-key "$API_KEY" \
    --port "$PORT" \
    --dtype auto \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
    --enforce-eager \
    --no-enable-log-requests \
    --enable-prefix-caching \
    --no-async-scheduling \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --reasoning-parser qwen3 \
    >"$server_log" 2>&1 &
  SERVER_PID=$!
  wait_ready

  PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$SCRIPT_DIR/run_attention_heatmap.py" capture \
    --mode "$mode" \
    --base-url "http://127.0.0.1:$PORT" \
    --api-key "$API_KEY" \
    --model "$SERVED_MODEL" \
    --model-path "$MODEL_PATH" \
    --skill-path "$SKILL_PATH" \
    --output-dir "$RUN_DIR" \
    --spec-path "$SPEC_PATH"

  stop_server
}

run_mode recompute
run_mode direct

PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$SCRIPT_DIR/run_attention_heatmap.py" plot \
  --output-dir "$RUN_DIR"

echo "[done] $RUN_DIR"
