#!/usr/bin/env bash
# Controlled shared-dominant geometry: Full vs materialized vs Shared Bank.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_ROOT="$ROOT/scripts/08_lmcache_mp"
CAPTURE_DIR="$SCRIPT_ROOT/cross_request_kv_capture"
GPU_DIR="$SCRIPT_ROOT/shared_bank_gpu_closure"
SCALING_DIR="$SCRIPT_ROOT/shared_bank_scaling"
PYTHON_BIN="${PYTHON_BIN:-python}"
VLLM_BIN="${VLLM_BIN:-vllm}"
MODEL_PATH="${VLLM_MODEL_PATH:-/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B}"
SERVED_MODEL="${VLLM_SERVED_NAME:-Qwen3}"
PORT="${VLLM_PORT:-8100}"
API_KEY="${VLLM_API_KEY:-EMPTY}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
READY_ATTEMPTS="${VLLM_READY_MAX_ATTEMPTS:-450}"
READY_INTERVAL="${VLLM_READY_INTERVAL:-2}"
NUM_LAYERS="${SEGMENTIA_MODEL_NUM_LAYERS:-40}"
KV_HEADS="${SEGMENTIA_MODEL_KV_HEADS:-8}"
HEAD_DIM="${SEGMENTIA_MODEL_HEAD_DIM:-128}"
DTYPE_BYTES="${SEGMENTIA_KV_DTYPE_BYTES:-2}"
RUN_ID="${RUN_ID:?Set a fresh RUN_ID}"
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:?Set SOURCE_RUN_DIR to a completed real-Skill run}"
SEED_SPEC="${SEGMENTIA_DOMINANT_SEED_SPEC:-$SOURCE_RUN_DIR/requests/source.json}"
OUTPUT_ROOT="${SEGMENTIA_DOMINANT_OUTPUT_ROOT:-/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/08_lmcache_mp/shared_dominant_scaling_runs}"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
RESULT_DIR="${SEGMENTIA_DOMINANT_RESULT_DIR:-$ROOT/results/problem_exploration/shared_dominant_skill_scaling}"
PYTHONPATH_VALUE="$SCRIPT_ROOT:$CAPTURE_DIR:$ROOT/vllm:$ROOT/LMCache${PYTHONPATH:+:$PYTHONPATH}"
SEPARATOR="${LMCACHE_BLEND_SPECIAL_STR:-<|fim_pad|><|repo_name|>}"
SHAPES=(long-6k long-8k)
MODES=(full materialized shared)
FOLLOWER_POINTS=(1 4)
PRE_P_CAP="${VLLM_SEGMENTIA_SHARED_KV_PRE_P_CAP:-2}"
RECOVER_INCOMPLETE="${RECOVER_INCOMPLETE:-0}"

if [[ "$RECOVER_INCOMPLETE" != "0" && "$RECOVER_INCOMPLETE" != "1" ]]; then
  echo "[error] RECOVER_INCOMPLETE must be 0 or 1" >&2
  exit 2
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "opencode" ]]; then
  echo "[error] activate the opencode conda environment before running" >&2
  exit 2
fi
for command in "$PYTHON_BIN" "$VLLM_BIN" curl cp; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "[error] required command is unavailable: $command" >&2
    exit 2
  }
done
[[ -f "$SEED_SPEC" ]] || { echo "[error] missing seed spec: $SEED_SPEC" >&2; exit 2; }

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export VLLM_API_KEY="$API_KEY"
export VLLM_SERVED_NAME="$SERVED_MODEL"
export VLLM_USE_V2_MODEL_RUNNER=0
export LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-256}"
export LMCACHE_BLEND_SPECIAL_STR="$SEPARATOR"
export LMCACHE_USE_LAYERWISE=True
export LMCACHE_ENABLE_SEGMENTIA=True
export LMCACHE_ENABLE_BLENDING=False
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE="${LMCACHE_MAX_LOCAL_CPU_SIZE:-8}"
export LMCACHE_MAX_LOCAL_DISK_SIZE="${LMCACHE_MAX_LOCAL_DISK_SIZE:-80}"
export LMCACHE_LOCAL_DISK_REHYDRATE=True
export LMCACHE_LOG_LEVEL="${LMCACHE_LOG_LEVEL:-INFO}"
export LMCACHE_EXTRA_CONFIG='{"segmentia_prefix_correction":true,"segmentia_prefix_apply_correction":true,"segmentia_cpu_prefetch":true}'
export SEGMENTIA_PROFILE=1

SERVER_PID=""
cleanup_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap cleanup_server EXIT INT TERM

wait_vllm_ready() {
  local log_path="$1" attempt=0 code="000"
  while ((attempt < READY_ATTEMPTS)); do
    if [[ -f "$log_path" ]] && grep -Fq "EngineCore failed to start" "$log_path"; then
      echo "[error] EngineCore failed during startup; inspect $log_path" >&2
      return 1
    fi
    if [[ -n "$SERVER_PID" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "[error] vLLM exited before readiness" >&2
      return 1
    fi
    code="$(curl -sS -o /dev/null -w '%{http_code}' --connect-timeout 3 \
      --max-time 10 -H "Authorization: Bearer $API_KEY" \
      "http://127.0.0.1:$PORT/v1/models" 2>/dev/null || true)"
    [[ "$code" == "200" ]] && return 0
    attempt=$((attempt + 1))
    sleep "$READY_INTERVAL"
  done
  echo "[error] vLLM readiness timeout" >&2
  return 1
}

start_server() {
  local shared_enabled="$1" cache_dir="$2" log_path="$3"
  mkdir -p "$cache_dir" "$(dirname "$log_path")"
  export LMCACHE_LOCAL_DISK="file://${cache_dir}/"
  export VLLM_SEGMENTIA_SHARED_KV="$shared_enabled"
  if [[ "$shared_enabled" == "1" ]]; then
    export VLLM_SEGMENTIA_SHARED_KV_PRE_P_CAP="$PRE_P_CAP"
  else
    unset VLLM_SEGMENTIA_SHARED_KV_PRE_P_CAP || true
  fi
  PYTHONPATH="$PYTHONPATH_VALUE" "$VLLM_BIN" serve "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL" --api-key "$API_KEY" --port "$PORT" \
    --dtype auto --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
    --enforce-eager --no-enable-log-requests --enable-prefix-caching \
    --no-async-scheduling >"$log_path" 2>&1 &
  SERVER_PID=$!
  wait_vllm_ready "$log_path"
  if grep -Fq "Using V2 Model Runner" "$log_path"; then
    echo "[error] unsupported V2 Model Runner selected" >&2
    return 1
  fi
}

send_one() {
  PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$GPU_DIR/send_request.py" \
    --spec "$1" --output "$2" --base-url "http://127.0.0.1:$PORT" \
    --api-key "$API_KEY"
}

point_complete() {
  local manifest="$1"
  [[ -f "$manifest" ]] && "$PYTHON_BIN" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); raise SystemExit(0 if p.get("failed")==0 and p.get("completed")==p.get("followers") else 1)' \
    "$manifest"
}

archive_incomplete() {
  local label="$1"
  shift
  if [[ "$RECOVER_INCOMPLETE" != "1" ]]; then
    echo "[error] incomplete artifact: $label" >&2
    echo "[hint] rerun with RECOVER_INCOMPLETE=1 to archive it and retry" >&2
    return 1
  fi
  local archive_dir="$RUN_DIR/failed_attempts/${label}-$(date -u +%Y%m%dT%H%M%S)-$$"
  mkdir -p "$archive_dir"
  local path
  for path in "$@"; do
    if [[ -e "$path" ]]; then
      mv "$path" "$archive_dir/"
    fi
  done
  echo "[recovered] archived incomplete artifact to $archive_dir"
}

mkdir -p "$RUN_DIR"
requests_current=0
if [[ -f "$RUN_DIR/requests/manifest.json" ]] && \
   "$PYTHON_BIN" -c \
     'import json,sys; p=json.load(open(sys.argv[1])); raise SystemExit(0 if p.get("generator_version")==3 else 1)' \
     "$RUN_DIR/requests/manifest.json"; then
  requests_current=1
fi
if [[ "$requests_current" == "0" ]]; then
  if [[ -e "$RUN_DIR/requests" ]]; then
    archive_incomplete "requests-old-generator" "$RUN_DIR/requests"
  fi
  PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$SCRIPT_DIR/prepare_shapes.py" \
    --seed-spec "$SEED_SPEC" --output-dir "$RUN_DIR/requests"
fi

echo "[phase 1/2] create one immutable SSD snapshot per controlled shape"
for shape in "${SHAPES[@]}"; do
  source_dir="$RUN_DIR/$shape/source"
  source_ssd="$RUN_DIR/$shape/source_ssd"
  if [[ -f "$source_dir/request.json" ]] && \
     "$PYTHON_BIN" -c 'import json,sys; raise SystemExit(0 if json.load(open(sys.argv[1])).get("status")=="completed" else 1)' "$source_dir/request.json"; then
    if PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$SCRIPT_DIR/wait_for_shape_cache.py" \
      --request "$source_dir/request.json" --cache-dir "$source_ssd" \
      --layers "$NUM_LAYERS" --kv-heads "$KV_HEADS" \
      --head-dim "$HEAD_DIM" --dtype-bytes "$DTYPE_BYTES" --timeout-s 1 \
      >/dev/null 2>&1; then
      echo "[skip-complete] shape=$shape source"
      continue
    fi
    archive_incomplete "$shape-source" "$source_dir" "$source_ssd"
  elif [[ -e "$source_dir" || -e "$source_ssd" ]]; then
    archive_incomplete "$shape-source" "$source_dir" "$source_ssd"
  fi
  mkdir -p "$source_dir" "$source_ssd"
  start_server 0 "$source_ssd" "$source_dir/vllm.log"
  send_one "$RUN_DIR/requests/$shape/source.json" "$source_dir/request.json"
  PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$SCRIPT_DIR/wait_for_shape_cache.py" \
    --request "$source_dir/request.json" --cache-dir "$source_ssd" \
    --layers "$NUM_LAYERS" --kv-heads "$KV_HEADS" \
    --head-dim "$HEAD_DIM" --dtype-bytes "$DTYPE_BYTES"
  cleanup_server
done

echo "[phase 2/2] isolated (shape, mode, N) points"
for shape in "${SHAPES[@]}"; do
  for mode in "${MODES[@]}"; do
    for followers in "${FOLLOWER_POINTS[@]}"; do
      point_dir="$RUN_DIR/$shape/$mode/n$followers"
      if point_complete "$point_dir/followers/manifest.json"; then
        echo "[skip-complete] shape=$shape mode=$mode followers=$followers"
        continue
      fi
      if [[ -e "$point_dir" ]]; then
        archive_incomplete "$shape-$mode-n$followers" "$point_dir"
      fi
      mkdir -p "$point_dir/lmcache_disk"
      shared_enabled=0
      spec_arm="reuse"
      if [[ "$mode" == "full" ]]; then
        spec_arm="full"
      else
        cp -a "$RUN_DIR/$shape/source_ssd/." "$point_dir/lmcache_disk/"
      fi
      [[ "$mode" == "shared" ]] && shared_enabled=1
      echo "[point-start] shape=$shape mode=$mode followers=$followers"
      start_server "$shared_enabled" "$point_dir/lmcache_disk" "$point_dir/vllm.log"
      send_one "$RUN_DIR/requests/$shape/$spec_arm/owner.json" "$point_dir/owner.json"
      if [[ "$mode" == "shared" ]]; then
        PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$GPU_DIR/wait_for_event.py" \
          --log "$point_dir/vllm.log" --event segmentia_shared_bank_publish \
          --response-record "$point_dir/owner.json"
      fi
      PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$SCALING_DIR/run_point.py" \
        --spec-dir "$RUN_DIR/requests/$shape/$spec_arm" \
        --output-dir "$point_dir/followers" --followers "$followers" \
        --mode "$mode" --base-url "http://127.0.0.1:$PORT" --api-key "$API_KEY"
      cleanup_server
    done
  done
done

PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$SCRIPT_DIR/validate_sanity.py" \
  --run-dir "$RUN_DIR" --output-dir "$RESULT_DIR"
echo "[completed] run=$RUN_DIR analysis=$RESULT_DIR"
