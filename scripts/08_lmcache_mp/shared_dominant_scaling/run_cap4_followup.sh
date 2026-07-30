#!/usr/bin/env bash
# Isolate the fixed pre-P admission bottleneck with Shared N=4, cap=4.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_ROOT="$ROOT/scripts/08_lmcache_mp"
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
BASELINE_RUN_DIR="${BASELINE_RUN_DIR:?Set BASELINE_RUN_DIR to the completed shared-dominant sanity run}"
OUTPUT_ROOT="${SEGMENTIA_DOMINANT_ADMISSION_OUTPUT_ROOT:-/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/08_lmcache_mp/shared_dominant_admission_runs}"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
RESULT_DIR="${SEGMENTIA_DOMINANT_ADMISSION_RESULT_DIR:-$ROOT/results/problem_exploration/shared_dominant_admission}"
PYTHONPATH_VALUE="$SCRIPT_ROOT:$SCRIPT_ROOT/cross_request_kv_capture:$ROOT/vllm:$ROOT/LMCache${PYTHONPATH:+:$PYTHONPATH}"
SHAPES=(long-6k long-8k)
FOLLOWERS=4
PRE_P_CAP=4
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
for shape in "${SHAPES[@]}"; do
  for required in \
    "$BASELINE_RUN_DIR/$shape/source/request.json" \
    "$BASELINE_RUN_DIR/$shape/source_ssd" \
    "$BASELINE_RUN_DIR/requests/$shape/reuse/owner.json" \
    "$BASELINE_RUN_DIR/requests/$shape/reuse/follower-003.json" \
    "$BASELINE_RUN_DIR/$shape/materialized/n4/followers/manifest.json" \
    "$BASELINE_RUN_DIR/$shape/shared/n4/followers/manifest.json"; do
    [[ -e "$required" ]] || {
      echo "[error] baseline artifact is missing: $required" >&2
      exit 2
    }
  done
done

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export VLLM_API_KEY="$API_KEY"
export VLLM_SERVED_NAME="$SERVED_MODEL"
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_SEGMENTIA_SHARED_KV=1
export VLLM_SEGMENTIA_SHARED_KV_PRE_P_CAP="$PRE_P_CAP"
export LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-256}"
export LMCACHE_BLEND_SPECIAL_STR="${LMCACHE_BLEND_SPECIAL_STR:-<|fim_pad|><|repo_name|>}"
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
      echo "[error] vLLM exited before readiness; inspect $log_path" >&2
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

verify_local_sources() {
  PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" -c '
from pathlib import Path
import lmcache
import vllm
from vllm import envs
root = Path("'"$ROOT"'").resolve()
actual = {"vllm": Path(vllm.__file__).resolve(), "lmcache": Path(lmcache.__file__).resolve()}
for name, path in actual.items():
    if not path.is_relative_to(root / ("vllm" if name == "vllm" else "LMCache")):
        raise RuntimeError(f"{name} resolved to unexpected source: {path}")
if envs.VLLM_USE_V2_MODEL_RUNNER is not False:
    raise RuntimeError("cap=4 follow-up requires the V1 Model Runner")
print(f"[sources] vllm={actual['"'"'vllm'"'"']} lmcache={actual['"'"'lmcache'"'"']}")
print("[runner] v1 (VLLM_USE_V2_MODEL_RUNNER=0)")
'
}

start_server() {
  local cache_dir="$1" log_path="$2"
  export LMCACHE_LOCAL_DISK="file://${cache_dir}/"
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

point_complete() {
  local manifest="$1"
  [[ -f "$manifest" ]] && "$PYTHON_BIN" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); raise SystemExit(0 if p.get("failed")==0 and p.get("completed")==4 else 1)' \
    "$manifest"
}

archive_incomplete() {
  local label="$1" path="$2"
  if [[ "$RECOVER_INCOMPLETE" != "1" ]]; then
    echo "[error] incomplete artifact: $label" >&2
    echo "[hint] rerun with RECOVER_INCOMPLETE=1 to archive it and retry" >&2
    return 1
  fi
  local archive_dir="$RUN_DIR/failed_attempts/${label}-$(date -u +%Y%m%dT%H%M%S)-$$"
  mkdir -p "$archive_dir"
  mv "$path" "$archive_dir/"
  echo "[recovered] archived incomplete artifact to $archive_dir"
}

mkdir -p "$RUN_DIR"
verify_local_sources
for shape in "${SHAPES[@]}"; do
  PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$SCRIPT_DIR/wait_for_shape_cache.py" \
    --request "$BASELINE_RUN_DIR/$shape/source/request.json" \
    --cache-dir "$BASELINE_RUN_DIR/$shape/source_ssd" --layers "$NUM_LAYERS" \
    --kv-heads "$KV_HEADS" --head-dim "$HEAD_DIM" \
    --dtype-bytes "$DTYPE_BYTES" --timeout-s 1 >/dev/null

  point_dir="$RUN_DIR/$shape/cap4"
  if point_complete "$point_dir/followers/manifest.json"; then
    echo "[skip-complete] shape=$shape cap=4 followers=4"
    continue
  fi
  [[ ! -e "$point_dir" ]] || archive_incomplete "$shape-cap4" "$point_dir"
  mkdir -p "$point_dir/lmcache_disk"
  cp -a "$BASELINE_RUN_DIR/$shape/source_ssd/." "$point_dir/lmcache_disk/"
  echo "[point-start] shape=$shape mode=shared followers=4 pre_p_cap=4"
  start_server "$point_dir/lmcache_disk" "$point_dir/vllm.log"
  PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$GPU_DIR/send_request.py" \
    --spec "$BASELINE_RUN_DIR/requests/$shape/reuse/owner.json" \
    --output "$point_dir/owner.json" --base-url "http://127.0.0.1:$PORT" \
    --api-key "$API_KEY"
  PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$GPU_DIR/wait_for_event.py" \
    --log "$point_dir/vllm.log" --event segmentia_shared_bank_publish \
    --response-record "$point_dir/owner.json"
  PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$SCALING_DIR/run_point.py" \
    --spec-dir "$BASELINE_RUN_DIR/requests/$shape/reuse" \
    --output-dir "$point_dir/followers" --followers "$FOLLOWERS" \
    --mode shared --base-url "http://127.0.0.1:$PORT" --api-key "$API_KEY"
  cleanup_server
done

PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$SCRIPT_DIR/validate_cap4_followup.py" \
  --baseline-run-dir "$BASELINE_RUN_DIR" --run-dir "$RUN_DIR" \
  --output-dir "$RESULT_DIR"
echo "[completed] run=$RUN_DIR analysis=$RESULT_DIR"
