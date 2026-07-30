#!/usr/bin/env bash
# N=4 diagnostic: Shared Bank pre-P admission cap 1/2/4.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_ROOT="$ROOT/scripts/08_lmcache_mp"
GPU_CLOSURE_DIR="$SCRIPT_ROOT/shared_bank_gpu_closure"
PYTHON_BIN="${PYTHON_BIN:-python}"
VLLM_BIN="${VLLM_BIN:-vllm}"
MODEL_PATH="${VLLM_MODEL_PATH:-/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B}"
SERVED_MODEL="${VLLM_SERVED_NAME:-Qwen3}"
PORT="${VLLM_PORT:-8100}"
API_KEY="${VLLM_API_KEY:-EMPTY}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
READY_ATTEMPTS="${VLLM_READY_MAX_ATTEMPTS:-450}"
READY_INTERVAL="${VLLM_READY_INTERVAL:-2}"
RUN_ID="${RUN_ID:?Set a fresh RUN_ID}"
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:?Set SOURCE_RUN_DIR to the completed scaling sanity run}"
OUTPUT_ROOT="${SEGMENTIA_PRE_P_CAP_OUTPUT_ROOT:-/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/08_lmcache_mp/shared_bank_pre_p_cap_runs}"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
RESULT_DIR="${SEGMENTIA_PRE_P_CAP_RESULT_DIR:-$ROOT/results/problem_exploration/shared_skill_pre_p_admission}"
PYTHONPATH_VALUE="$SCRIPT_ROOT:$SCRIPT_ROOT/cross_request_kv_capture:$ROOT/vllm:$ROOT/LMCache${PYTHONPATH:+:$PYTHONPATH}"
CAPS=(1 2 4)
FOLLOWERS=4

if [[ "${CONDA_DEFAULT_ENV:-}" != "opencode" ]]; then
  echo "[error] activate the opencode conda environment before running" >&2
  exit 2
fi
for command in "$PYTHON_BIN" "$VLLM_BIN" curl cp; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "[error] required command is unavailable: $command" >&2
    exit 2
  fi
done
for required in \
  "$SOURCE_RUN_DIR/source_ssd" \
  "$SOURCE_RUN_DIR/requests/owner.json" \
  "$SOURCE_RUN_DIR/requests/follower-003.json"; do
  if [[ ! -e "$required" ]]; then
    echo "[error] source artifact is missing: $required" >&2
    exit 2
  fi
done
if [[ -e "$RUN_DIR" ]]; then
  echo "[error] immutable run directory already exists: $RUN_DIR" >&2
  exit 2
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export VLLM_API_KEY="$API_KEY"
export VLLM_SERVED_NAME="$SERVED_MODEL"
export VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_SEGMENTIA_SHARED_KV=1
export LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-256}"
export LMCACHE_BLEND_SPECIAL_STR="${LMCACHE_BLEND_SPECIAL_STR:-<|fim_pad|><|repo_name|>}"
export LMCACHE_USE_LAYERWISE=True
export LMCACHE_ENABLE_SEGMENTIA=True
export LMCACHE_ENABLE_BLENDING=False
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE="${LMCACHE_MAX_LOCAL_CPU_SIZE:-5}"
export LMCACHE_MAX_LOCAL_DISK_SIZE="${LMCACHE_MAX_LOCAL_DISK_SIZE:-50}"
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

verify_local_sources() {
  PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" -c '
from pathlib import Path
import lmcache
import vllm
from vllm import envs
root = Path("'"$ROOT"'").resolve()
actual = {"vllm": Path(vllm.__file__).resolve(), "lmcache": Path(lmcache.__file__).resolve()}
expected = {"vllm": root / "vllm", "lmcache": root / "LMCache"}
for name, path in actual.items():
    if not path.is_relative_to(expected[name]):
        raise RuntimeError(f"{name} resolved to {path}, expected under {expected[name]}")
if envs.VLLM_USE_V2_MODEL_RUNNER is not False:
    raise RuntimeError("pre-P cap diagnostic requires the V1 Model Runner")
print(f"[sources] vllm={actual['"'"'vllm'"'"']} lmcache={actual['"'"'lmcache'"'"']}")
print("[runner] v1 (VLLM_USE_V2_MODEL_RUNNER=0)")
'
}

start_server() {
  local cap="$1"
  local cache_dir="$2"
  local log_path="$3"
  export VLLM_SEGMENTIA_SHARED_KV_PRE_P_CAP="$cap"
  export LMCACHE_LOCAL_DISK="file://${cache_dir}/"
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
    >"$log_path" 2>&1 &
  SERVER_PID=$!
  wait_vllm_ready
  if grep -Fq "Using V2 Model Runner" "$log_path"; then
    echo "[error] service selected the unsupported V2 Model Runner" >&2
    return 1
  fi
}

send_owner() {
  local output="$1"
  PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$GPU_CLOSURE_DIR/send_request.py" \
    --spec "$SOURCE_RUN_DIR/requests/owner.json" \
    --output "$output" \
    --base-url "http://127.0.0.1:$PORT" \
    --api-key "$API_KEY"
}

mkdir -p "$RUN_DIR"
verify_local_sources
for cap in "${CAPS[@]}"; do
  point_dir="$RUN_DIR/cap$cap"
  mkdir -p "$point_dir/lmcache_disk"
  cp -a "$SOURCE_RUN_DIR/source_ssd/." "$point_dir/lmcache_disk/"
  echo "[point-start] followers=$FOLLOWERS pre_p_cap=$cap"
  start_server "$cap" "$point_dir/lmcache_disk" "$point_dir/vllm.log"
  send_owner "$point_dir/owner.json"
  PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$GPU_CLOSURE_DIR/wait_for_event.py" \
    --log "$point_dir/vllm.log" \
    --event segmentia_shared_bank_publish \
    --response-record "$point_dir/owner.json"
  PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$SCRIPT_DIR/run_point.py" \
    --spec-dir "$SOURCE_RUN_DIR/requests" \
    --output-dir "$point_dir/followers" \
    --followers "$FOLLOWERS" \
    --mode shared \
    --base-url "http://127.0.0.1:$PORT" \
    --api-key "$API_KEY"
  cleanup_server
done

PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$SCRIPT_DIR/validate_pre_p_cap.py" \
  --run-dir "$RUN_DIR" \
  --output-dir "$RESULT_DIR"
echo "[completed] run=$RUN_DIR analysis=$RESULT_DIR"
