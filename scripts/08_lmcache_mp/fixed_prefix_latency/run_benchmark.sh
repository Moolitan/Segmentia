#!/usr/bin/env bash
# Fixed Prefix-256 clean latency and Skill-length break-even benchmark.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
OUTPUT_ROOT="${SEGMENTIA_FIXED_PREFIX_OUTPUT_ROOT:-/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/08_lmcache_mp/fixed_prefix_latency_runs}"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
RESULT_DIR="${SEGMENTIA_FIXED_PREFIX_RESULT_DIR:-$ROOT/results/problem_exploration/fixed_prefix_latency_break_even}"
PYTHONPATH_VALUE="$SCRIPT_DIR:$ROOT/vllm:$ROOT/LMCache${PYTHONPATH:+:$PYTHONPATH}"
LENGTHS="${LENGTHS:-512,640,768,1024,1280,1536,1792,2048,2560,3301}"
REPLICAS="${REPLICAS:-3}"
WARMUPS="${WARMUPS:-2}"
MEASUREMENTS="${MEASUREMENTS:-10}"
SEED="${SEED:-20260729}"
ARMS_CSV="${ARMS:-full,direct,prefix_no_correction,prefix_256}"
RESUME="${RESUME:-1}"
RUN_SOURCE="${RUN_SOURCE:-1}"
RUN_MEASURE="${RUN_MEASURE:-1}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"
NUM_LAYERS="${NUM_LAYERS:-40}"

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
if ! [[ "$REPLICAS" =~ ^[1-9][0-9]*$ && "$WARMUPS" =~ ^[0-9]+$ && "$MEASUREMENTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] invalid REPLICAS/WARMUPS/MEASUREMENTS" >&2
  exit 2
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export VLLM_API_KEY="$API_KEY"
export VLLM_SERVED_NAME="$SERVED_MODEL"
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
export SEGMENTIA_PROFILE="${SEGMENTIA_PROFILE:-1}"

mkdir -p "$RUN_DIR/.work"

PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" -c '
from pathlib import Path
import lmcache
import vllm
root = Path("'"$ROOT"'").resolve()
actual = {"vllm": Path(vllm.__file__).resolve(), "lmcache": Path(lmcache.__file__).resolve()}
expected = {"vllm": root / "vllm", "lmcache": root / "LMCache"}
for name, path in actual.items():
    if not path.is_relative_to(expected[name]):
        raise RuntimeError(f"{name} resolved to {path}, expected under {expected[name]}")
print(f"[sources] vllm={actual['"'"'vllm'"'"']} lmcache={actual['"'"'lmcache'"'"']}")
'

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
    code="$(curl -sS -o /dev/null -w '%{http_code}' --connect-timeout 3 --max-time 10 \
      -H "Authorization: Bearer $API_KEY" "http://127.0.0.1:$PORT/v1/models" 2>/dev/null || true)"
    if [[ "$code" == "200" ]]; then
      return 0
    fi
    attempt=$((attempt + 1))
    sleep "$READY_INTERVAL"
  done
  echo "[error] vLLM readiness timeout port=$PORT" >&2
  return 1
}

start_server() {
  local role="$1"
  local arm="$2"
  local work_dir="$3"
  local -a connector_args=()
  unset LMCACHE_LOCAL_DISK LMCACHE_EXTRA_CONFIG || true
  if [[ "$role" == "source" || "$arm" != "full" ]]; then
    export LMCACHE_LOCAL_DISK="file://$work_dir/ssd/"
    connector_args+=(--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}')
    if [[ "$role" == "source" || "$arm" == "direct" ]]; then
      if [[ "$role" == "source" ]]; then
        export LMCACHE_EXTRA_CONFIG='{"segmentia_direct_reuse":true}'
      else
        export LMCACHE_EXTRA_CONFIG='{"segmentia_direct_reuse":true,"segmentia_cpu_prefetch":true}'
      fi
    elif [[ "$arm" == "prefix_no_correction" ]]; then
      export LMCACHE_EXTRA_CONFIG='{"segmentia_prefix_correction":true,"segmentia_prefix_apply_correction":false,"segmentia_cpu_prefetch":true}'
    elif [[ "$arm" == "prefix_256" ]]; then
      export LMCACHE_EXTRA_CONFIG='{"segmentia_prefix_correction":true,"segmentia_prefix_apply_correction":true,"segmentia_cpu_prefetch":true}'
    else
      echo "[error] invalid reuse arm: $arm" >&2
      return 2
    fi
  fi
  PYTHONPATH="$PYTHONPATH_VALUE" "$VLLM_BIN" serve "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL" \
    --api-key "$API_KEY" \
    --port "$PORT" \
    --dtype auto \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --enforce-eager \
    --no-enable-log-requests \
    --enable-prefix-caching \
    --no-async-scheduling \
    "${connector_args[@]}" \
    >"$work_dir/vllm.log" 2>&1 &
  SERVER_PID=$!
  wait_vllm_ready
}

source_leaf="$RUN_DIR/source"
if [[ "$RUN_SOURCE" == "1" ]]; then
  if [[ -d "$source_leaf" ]]; then
    if [[ "$RESUME" != "1" ]]; then
      echo "[error] source leaf already exists: $source_leaf" >&2
      exit 2
    fi
    PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$SCRIPT_DIR/validate_run.py" ssd \
      --cache-dir "$source_leaf/ssd" --lengths "$LENGTHS" --layers "$NUM_LAYERS" --timeout-s 0
    echo "[skip-valid] source=$source_leaf"
  else
    source_work="$RUN_DIR/.work/source-$$-$(date +%s%N)"
    mkdir -p "$source_work/ssd"
    echo "[phase source] cold misses -> combined shared SSD"
    start_server source direct "$source_work"
    PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$SCRIPT_DIR/run_requests.py" \
      --phase source --lengths "$LENGTHS" --output-dir "$source_work" \
      --base-url "http://127.0.0.1:$PORT" --api-key "$API_KEY" --model "$SERVED_MODEL" --seed "$SEED"
    PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$SCRIPT_DIR/validate_run.py" ssd \
      --cache-dir "$source_work/ssd" --lengths "$LENGTHS" --layers "$NUM_LAYERS"
    cleanup_server
    mv "$source_work" "$source_leaf"
    echo "[published] source=$source_leaf"
  fi
fi
if [[ ! -d "$source_leaf/ssd" ]]; then
  echo "[error] shared source SSD is unavailable: $source_leaf/ssd" >&2
  exit 2
fi

IFS=',' read -ra selected_arms <<< "$ARMS_CSV"
if [[ "$RUN_MEASURE" == "1" ]]; then
  for ((replica=0; replica<REPLICAS; replica++)); do
    arm_count="${#selected_arms[@]}"
    for ((arm_index=0; arm_index<arm_count; arm_index++)); do
      rotated_index=$(((arm_index + replica) % arm_count))
      raw_arm="${selected_arms[$rotated_index]}"
      arm="${raw_arm// /}"
      case "$arm" in
        full|direct|prefix_no_correction|prefix_256) ;;
        *) echo "[error] invalid arm: $arm" >&2; exit 2 ;;
      esac
      leaf="$RUN_DIR/replica_$replica/$arm"
      if [[ -d "$leaf" ]]; then
        if [[ "$RESUME" != "1" ]]; then
          echo "[error] completed leaf already exists: $leaf" >&2
          exit 2
        fi
        PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$SCRIPT_DIR/validate_run.py" leaf \
          --leaf "$leaf" --log "$leaf/vllm.log" --arm "$arm"
        echo "[skip-valid] replica=$replica arm=$arm"
        continue
      fi
      work="$RUN_DIR/.work/replica_${replica}-${arm}-$$-$(date +%s%N)"
      mkdir -p "$work"
      if [[ "$arm" != "full" ]]; then
        mkdir -p "$work/ssd"
        cp -al "$source_leaf/ssd/." "$work/ssd/"
      fi
      echo "[phase measure] restart boundary=(replica=$replica, arm=$arm)"
      start_server target "$arm" "$work"
      request_args=(
        --phase measure --arm "$arm" --replica "$replica" --lengths "$LENGTHS"
        --warmups "$WARMUPS" --measurements "$MEASUREMENTS" --seed "$SEED"
        --output-dir "$work" --base-url "http://127.0.0.1:$PORT"
        --api-key "$API_KEY" --model "$SERVED_MODEL"
      )
      if [[ "$arm" != "full" ]]; then
        request_args+=(--cpu-prefetch)
      fi
      PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$SCRIPT_DIR/run_requests.py" \
        "${request_args[@]}"
      cleanup_server
      PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$SCRIPT_DIR/validate_run.py" leaf \
        --leaf "$work" --log "$work/vllm.log" --arm "$arm"
      mkdir -p "$(dirname "$leaf")"
      mv "$work" "$leaf"
      echo "[published] replica=$replica arm=$arm leaf=$leaf"
    done
  done
fi

if [[ "$RUN_ANALYSIS" == "1" ]]; then
  if [[ "$ARMS_CSV" != "full,direct,prefix_no_correction,prefix_256" ]]; then
    echo "[error] analysis requires all four arms in canonical order" >&2
    exit 2
  fi
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/segmentia-mpl}" PYTHONPATH="$PYTHONPATH_VALUE" \
    "$PYTHON_BIN" "$SCRIPT_DIR/analyze_latency.py" \
      --run-dir "$RUN_DIR" --output-dir "$RESULT_DIR" --replicas "$REPLICAS"
fi

echo "[completed] run=$RUN_DIR analysis=$RESULT_DIR"
