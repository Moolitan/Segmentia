#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SCRIPT_DIR="$ROOT/scripts/08_lmcache_mp/paper_motivation/3.2"
VLLM_ROOT="$ROOT/vllm"
LMCACHE_ROOT="$ROOT/LMCache"
MODEL_PATH="${VLLM_MODEL_PATH:-/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B}"
POOL_DIR="${SKILL_SAVE_POOL_DIR:-/mnt/Large_Language_Model_Lab_1/wsh/skill_save_pool/Qwen3-14B}"
OUTPUT_ROOT="${SEGMENTIA_OUTPUT_DIR:-/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/08_lmcache_mp/paper_motivation_3_2_context_free_residual}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
PORT="${VLLM_PORT:-8015}"
API_KEY="${VLLM_API_KEY:-EMPTY}"
SERVED_MODEL="${VLLM_SERVED_NAME:-Qwen3}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.9}"
PYTHON_BIN="${PYTHON_BIN:-python}"
VLLM_BIN="${VLLM_BIN:-vllm}"
PYTHONPATH_VALUE="$VLLM_ROOT:$LMCACHE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
CASES="$SCRIPT_DIR/cases.json"
ANALYSIS_ONLY=""

if [[ $# -gt 0 ]]; then
  if [[ $# -ne 2 || "$1" != "--analysis-only" ]]; then
    echo "usage: $0 [--analysis-only EXISTING_RUN_DIR]" >&2
    exit 2
  fi
  ANALYSIS_ONLY="$2"
  RUN_DIR="$(realpath "$ANALYSIS_ONLY")"
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "opencode" ]]; then
  echo "[error] activate conda environment opencode first" >&2
  exit 2
fi
if [[ -z "$ANALYSIS_ONLY" && -e "$RUN_DIR" ]]; then
  echo "[error] immutable run directory already exists: $RUN_DIR" >&2
  exit 2
fi
if [[ -n "$ANALYSIS_ONLY" ]]; then
  if [[ ! -d "$RUN_DIR/analysis" ]]; then
    echo "[error] existing run has no analysis directory: $RUN_DIR" >&2
    exit 2
  fi
else
  mkdir -p "$RUN_DIR"
fi
export MPLCONFIGDIR="$RUN_DIR/.matplotlib"
mkdir -p "$MPLCONFIGDIR"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
export OPENHANDS_SUPPRESS_BANNER=1
export PYTHONHASHSEED=0
export LMCACHE_CHUNK_SIZE=256
export LMCACHE_USE_LAYERWISE=True
export LMCACHE_ENABLE_SEGMENTIA=True
export LMCACHE_EXTRA_CONFIG='{}'
# Every case starts with a fresh directory, so enabling rehydration cannot load
# an earlier request.  The flag is required here because the local-disk backend
# uses it to persist the per-layer metadata sidecars consumed by the analyzer.
export LMCACHE_LOCAL_DISK_REHYDRATE=True
export LMCACHE_LOCAL_CPU=False
export LMCACHE_MAX_LOCAL_CPU_SIZE=5
export LMCACHE_MAX_LOCAL_DISK_SIZE=100
unset LMCACHE_FORCE_SKIP_SAVE || true

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
  for _ in $(seq 1 450); do
    if [[ -n "$SERVER_PID" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
      return 1
    fi
    code="$(curl -sS -o /dev/null -w '%{http_code}' --connect-timeout 3 --max-time 10 \
      -H "Authorization: Bearer $API_KEY" "http://127.0.0.1:$PORT/v1/models" 2>/dev/null || true)"
    [[ "$code" == "200" ]] && return 0
    sleep 2
  done
  return 1
}

if [[ -z "$ANALYSIS_ONLY" ]]; then
  mapfile -t rows < <(jq -r '.cases[] | [.case_id,.skill,.task_prompt] | @tsv' "$CASES")
  for row in "${rows[@]}"; do
    IFS=$'\t' read -r case_id skill task_prompt <<<"$row"
    case_dir="$RUN_DIR/$case_id"
    kv_dir="$case_dir/online_full_kv"
    workspace="$case_dir/workspace"
    mkdir -p "$kv_dir" "$workspace"

    # Isolation boundary is (mode=online_full, task=case_id): every task gets a
    # fresh vLLM process and therefore an empty vLLM prefix cache.
    export LMCACHE_LOCAL_DISK="file://${kv_dir}/"
    log="$case_dir/vllm.log"
    echo "[server] mode=online_full task=$case_id"
    PYTHONPATH="$PYTHONPATH_VALUE" "$VLLM_BIN" serve "$MODEL_PATH" \
      --served-model-name "$SERVED_MODEL" --api-key "$API_KEY" --port "$PORT" \
      --dtype auto --max-model-len 32768 --gpu-memory-utilization "$GPU_UTIL" \
      --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
      --enforce-eager --no-enable-log-requests --enable-prefix-caching \
      --no-async-scheduling --enable-auto-tool-choice --tool-call-parser hermes \
      --reasoning-parser qwen3 >"$log" 2>&1 &
    SERVER_PID=$!
    if ! wait_ready; then
      tail -n 80 "$log" >&2 || true
      exit 1
    fi

    PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$SCRIPT_DIR/capture_online_full.py" \
      --skill "$skill" --task-prompt "$ROOT/$task_prompt" \
      --output "$case_dir/capture.json" --kv-dir "$kv_dir" \
      --workspace "$workspace" --pool-dir "$POOL_DIR" --model-path "$MODEL_PATH" \
      --served-model "$SERVED_MODEL" --base-url "http://127.0.0.1:$PORT" \
      --api-key "$API_KEY" >"$case_dir/agent.log" 2>&1
    stop_server
  done

else
  echo "[analysis-only] source=$RUN_DIR"
fi

"$PYTHON_BIN" "$SCRIPT_DIR/analyze_token_axis.py" \
  --run-dir "$RUN_DIR" --pool-dir "$POOL_DIR" --cases "$CASES" \
  --output-dir "$RUN_DIR/analysis" \
  --observation-start 132 --observation-end 256 --evaluation-start 256
"$PYTHON_BIN" "$SCRIPT_DIR/plot_token_axis.py" \
  --fidelity-csv "$RUN_DIR/analysis/token_axis_fidelity.csv" \
  --commonality-csv "$RUN_DIR/analysis/token_residual_commonality.csv" \
  --output-dir "$RUN_DIR/figures"
"$PYTHON_BIN" "$SCRIPT_DIR/publish_token_axis.py" \
  --run-dir "$RUN_DIR" --cases "$CASES" \
  --result-dir "$ROOT/results/problem_exploration/cskcache_token_axis_correction"
echo "[done] $RUN_DIR"
