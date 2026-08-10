#!/usr/bin/env bash
# Measure Recompute, Direct reuse, and K-only correction on one real 8K Skill.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SCRIPT_DIR="$ROOT/scripts/08_lmcache_mp/paper_experiment/latency"
PYTHON_BIN="${PYTHON_BIN:-python}"
VLLM_BIN="${VLLM_BIN:-vllm}"
MODEL_PATH="${VLLM_MODEL_PATH:-/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B}"
SERVED_MODEL="${VLLM_SERVED_NAME:-Qwen3}"
POOL_DIR="${SKILL_SAVE_POOL_DIR:-/mnt/Large_Language_Model_Lab_1/wsh/skill_save_pool/Qwen3-14B}"
CACHE_ID="${CACHE_ID:-Auto-claude-code-research-in-sleep/paper-write}"
CACHE_KV_DIR="$POOL_DIR/$CACHE_ID/kv"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUTPUT_ROOT="${SEGMENTIA_LATENCY_OUTPUT_ROOT:-/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/08_lmcache_mp/paper_experiment_latency}"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
RESULT_DIR="${SEGMENTIA_LATENCY_RESULT_DIR:-$ROOT/results/problem_exploration/skill_latency_8k}"
PORT="${VLLM_PORT:-8120}"
API_KEY="${VLLM_API_KEY:-EMPTY}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
READY_ATTEMPTS="${VLLM_READY_MAX_ATTEMPTS:-450}"
READY_INTERVAL="${VLLM_READY_INTERVAL:-2}"
REPLICAS="${REPLICAS:-3}"
WARMUPS="${WARMUPS:-2}"
MEASUREMENTS="${MEASUREMENTS:-10}"
PREFIX_TOKENS="${PREFIX_TOKENS:-1024}"
SUFFIX_TOKENS="${SUFFIX_TOKENS:-32}"
RESUME="${RESUME:-1}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"
DRY_RUN="${DRY_RUN:-0}"
PYTHONPATH_VALUE="$SCRIPT_DIR:$ROOT/vllm:$ROOT/LMCache${PYTHONPATH:+:$PYTHONPATH}"
MODES=(full direct correction)

usage() {
  echo "usage: RUN_ID=<id> bash $0 [--dry-run]"
  echo "env: REPLICAS=3 WARMUPS=2 MEASUREMENTS=10 PREFIX_TOKENS=1024 SUFFIX_TOKENS=32"
}

while (($#)); do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[error] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "${CONDA_DEFAULT_ENV:-}" != "opencode" ]]; then
  echo "[error] activate the opencode conda environment first" >&2
  exit 2
fi
if ! [[ "$REPLICAS" =~ ^[1-9][0-9]*$ && "$WARMUPS" =~ ^[0-9]+$ && "$MEASUREMENTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] invalid REPLICAS/WARMUPS/MEASUREMENTS" >&2
  exit 2
fi
if [[ ! -f "$POOL_DIR/$CACHE_ID/manifest.json" || ! -d "$CACHE_KV_DIR" ]]; then
  echo "[error] cached Skill is unavailable: $POOL_DIR/$CACHE_ID" >&2
  exit 2
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

common_workload_args=(
  --pool-dir "$POOL_DIR"
  --cache-id "$CACHE_ID"
  --model "$SERVED_MODEL"
  --prefix-tokens "$PREFIX_TOKENS"
  --suffix-tokens "$SUFFIX_TOKENS"
  --warmups "$WARMUPS"
  --measurements "$MEASUREMENTS"
)

if [[ "$DRY_RUN" == "1" ]]; then
  PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$SCRIPT_DIR/workload.py" dry-run \
    "${common_workload_args[@]}" --replica 0
  exit 0
fi

for command in "$PYTHON_BIN" "$VLLM_BIN" curl; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "[error] required command is unavailable: $command" >&2
    exit 2
  fi
done

PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" -c '
from pathlib import Path
import lmcache
import vllm
root = Path("'"$ROOT"'").resolve()
actual = {"vllm": Path(vllm.__file__).resolve(), "lmcache": Path(lmcache.__file__).resolve()}
for name, path in actual.items():
    expected = root / ("LMCache" if name == "lmcache" else "vllm")
    if not path.is_relative_to(expected):
        raise RuntimeError(f"{name} resolved to {path}, expected under {expected}")
print(f"[sources] vllm={actual['"'"'vllm'"'"']} lmcache={actual['"'"'lmcache'"'"']}")
'

mkdir -p "$RUN_DIR/.work"

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
  local mode="$1"
  local log_path="$2"
  local -a connector_args=()
  unset LMCACHE_LOCAL_DISK LMCACHE_EXTRA_CONFIG LMCACHE_FORCE_SKIP_SAVE || true
  export LMCACHE_CHUNK_SIZE=256
  export LMCACHE_USE_LAYERWISE=True
  export LMCACHE_MAX_LOCAL_DISK_SIZE="${LMCACHE_MAX_LOCAL_DISK_SIZE:-1000}"
  export LMCACHE_LOG_LEVEL="${LMCACHE_LOG_LEVEL:-INFO}"
  if [[ "$mode" == "full" ]]; then
    export LMCACHE_ENABLE_SEGMENTIA=False
    export LMCACHE_LOCAL_CPU=False
    export LMCACHE_LOCAL_DISK_REHYDRATE=False
  else
    export LMCACHE_ENABLE_SEGMENTIA=True
    export LMCACHE_LOCAL_CPU=True
    export LMCACHE_MAX_LOCAL_CPU_SIZE="${LMCACHE_MAX_LOCAL_CPU_SIZE:-8}"
    export LMCACHE_LOCAL_DISK_REHYDRATE=True
    export LMCACHE_LOCAL_DISK="file://${CACHE_KV_DIR}/"
    export LMCACHE_FORCE_SKIP_SAVE=1
    connector_args+=(--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}')
    if [[ "$mode" == "direct" ]]; then
      export LMCACHE_EXTRA_CONFIG='{"segmentia_direct_reuse":true,"segmentia_cpu_prefetch":true}'
    elif [[ "$mode" == "correction" ]]; then
      export LMCACHE_EXTRA_CONFIG='{"segmentia_prefix_correction":true,"segmentia_prefix_apply_correction":true,"segmentia_prefix_correction_alpha":0.6,"segmentia_cpu_prefetch":true}'
    else
      echo "[error] unsupported mode: $mode" >&2
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
    >"$log_path" 2>&1 &
  SERVER_PID=$!
  wait_vllm_ready
}

for ((replica=0; replica<REPLICAS; replica++)); do
  for ((offset=0; offset<${#MODES[@]}; offset++)); do
    mode="${MODES[$(((offset + replica) % ${#MODES[@]}))]}"
    leaf="$RUN_DIR/replica_$replica/$mode"
    if [[ -d "$leaf" ]]; then
      if [[ "$RESUME" != "1" ]]; then
        echo "[error] completed leaf already exists: $leaf" >&2
        exit 2
      fi
      PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$SCRIPT_DIR/validate_leaf.py" \
        --leaf "$leaf" --log "$leaf/vllm.log" --mode "$mode"
      echo "[skip-valid] replica=$replica mode=$mode"
      continue
    fi
    work="$RUN_DIR/.work/replica_${replica}-${mode}-$$-$(date +%s%N)"
    mkdir -p "$work"
    echo "[server] boundary=(replica=$replica, mode=$mode, task=paper-write-8k) log=$work/vllm.log"
    start_server "$mode" "$work/vllm.log"
    PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$SCRIPT_DIR/workload.py" measure \
      "${common_workload_args[@]}" \
      --mode "$mode" \
      --replica "$replica" \
      --output-dir "$work" \
      --base-url "http://127.0.0.1:$PORT" \
      --api-key "$API_KEY"
    cleanup_server
    PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$SCRIPT_DIR/validate_leaf.py" \
      --leaf "$work" --log "$work/vllm.log" --mode "$mode"
    mkdir -p "$(dirname "$leaf")"
    mv "$work" "$leaf"
    echo "[published] replica=$replica mode=$mode leaf=$leaf"
  done
done

if [[ "$RUN_ANALYSIS" == "1" ]]; then
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/segmentia-mpl}" PYTHONPATH="$PYTHONPATH_VALUE" \
    "$PYTHON_BIN" "$SCRIPT_DIR/analyze_latency.py" \
      --run-dir "$RUN_DIR" --output-dir "$RESULT_DIR" --replicas "$REPLICAS"
fi

echo "[completed] run=$RUN_DIR summary=$RESULT_DIR"
