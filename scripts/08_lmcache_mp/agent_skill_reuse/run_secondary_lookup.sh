#!/usr/bin/env bash
# Restart vLLM per task and run the real OpenHands secondary-lookup driver.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${VLLM_MODEL_PATH:-/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B}"
SERVED_MODEL="${VLLM_SERVED_NAME:-Qwen3}"
PORT="${VLLM_PORT:-8100}"
API_KEY="${VLLM_API_KEY:-EMPTY}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${LMCACHE_AGENT_RUN_DIR:-/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/08_lmcache_mp/agent_skill_secondary_lookup_runs/$RUN_ID}"
TASKS="${TASKS:-doc_coauthoring_design_doc}"
READY_ATTEMPTS="${VLLM_READY_MAX_ATTEMPTS:-450}"
READY_INTERVAL="${VLLM_READY_INTERVAL:-2}"
PYTHONPATH_VALUE="$ROOT/vllm:$ROOT/LMCache${PYTHONPATH:+:$PYTHONPATH}"
CACHE_SESSION_ID="${LMCACHE_CACHE_SESSION_ID:-$(date +%Y%m%d-%H%M%S)-$$}"
LMCACHE_DISK_ROOT="${LMCACHE_DISK_DIR:-$RUN_DIR/lmcache_sessions/$CACHE_SESSION_ID}"

OVERWRITE=0
DRY_RUN=0
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --overwrite) OVERWRITE=1 ;;
    --dry-run) DRY_RUN=1 ;;
    *) echo "usage: $0 [--overwrite] [--dry-run]" >&2; exit 2 ;;
  esac
  shift
done

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
export VLLM_API_KEY="$API_KEY"
export VLLM_SERVED_NAME="$SERVED_MODEL"
export OPENHANDS_SUPPRESS_BANNER=1
export LITELLM_LOCAL_MODEL_COST_MAP=True
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

export LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-256}"
export LMCACHE_ENABLE_BLENDING="${LMCACHE_ENABLE_BLENDING:-True}"
export LMCACHE_BLEND_SPECIAL_STR="${LMCACHE_BLEND_SPECIAL_STR:-<|fim_pad|><|repo_name|>}"
export LMCACHE_USE_LAYERWISE="${LMCACHE_USE_LAYERWISE:-True}"
export LMCACHE_BLEND_CHECK_LAYERS="${LMCACHE_BLEND_CHECK_LAYERS:-1}"
export LMCACHE_BLEND_RECOMPUTE_RATIOS="${LMCACHE_BLEND_RECOMPUTE_RATIOS:-0.15}"
export LMCACHE_LOCAL_CPU="${LMCACHE_LOCAL_CPU:-True}"
export LMCACHE_MAX_LOCAL_CPU_SIZE="${LMCACHE_MAX_LOCAL_CPU_SIZE:-5}"
export LMCACHE_MAX_LOCAL_DISK_SIZE="${LMCACHE_MAX_LOCAL_DISK_SIZE:-50}"

KV_TRANSFER_CONFIG='{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
SERVER_PID=""

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
      echo "[vLLM] ready port=$PORT"
      return 0
    fi
    attempt=$((attempt + 1))
    sleep "$READY_INTERVAL"
  done
  echo "[error] vLLM readiness timeout port=$PORT" >&2
  return 1
}

verify_local_sources() {
  PYTHONPATH="$PYTHONPATH_VALUE" python -c '
from pathlib import Path
import lmcache
import vllm
root = Path("'"$ROOT"'").resolve()
paths = {"vllm": Path(vllm.__file__).resolve(), "lmcache": Path(lmcache.__file__).resolve()}
expected = {"vllm": root / "vllm", "lmcache": root / "LMCache"}
for name, path in paths.items():
    if not path.is_relative_to(expected[name]):
        raise RuntimeError(f"{name} resolved to {path}, expected under {expected[name]}")
print(f"[sources] vllm={paths['"'"'vllm'"'"']} lmcache={paths['"'"'lmcache'"'"']}")
'
}

run_driver() {
  local task="$1"
  local vllm_log="$RUN_DIR/vllm_${task}.log"
  local args=(
    --benchmark-repo "$task"
    --bench-root "$ROOT/anthropic_skill_benchmark"
    --skills-dir "$ROOT/skills"
    --run-dir "$RUN_DIR"
    --tokenizer-path "$MODEL_PATH"
    --vllm-log "$vllm_log"
    --model "$SERVED_MODEL"
    --vllm-port "$PORT"
  )
  [[ "$OVERWRITE" == "1" ]] && args+=(--overwrite)
  [[ "$DRY_RUN" == "1" ]] && args+=(--dry-run)
  env -u LD_LIBRARY_PATH PYTHONPATH="$PYTHONPATH_VALUE" \
    python "$SCRIPT_DIR/run_secondary_lookup_agent.py" "${args[@]}"
}

IFS=',' read -ra TASK_LIST <<< "$TASKS"
verify_local_sources
if [[ "$DRY_RUN" == "1" ]]; then
  for task in "${TASK_LIST[@]}"; do
    task="${task// /}"
    [[ -z "$task" ]] || run_driver "$task"
  done
  exit 0
fi

mkdir -p "$RUN_DIR" "$LMCACHE_DISK_ROOT"
failures=0
for task in "${TASK_LIST[@]}"; do
  task="${task// /}"
  [[ -z "$task" ]] && continue
  if [[ -e "$RUN_DIR/$task" && "$OVERWRITE" == "0" ]]; then
    if [[ -f "$RUN_DIR/$task/_summary.json" ]] &&
       grep -q '"status": "completed"' "$RUN_DIR/$task/_summary.json"; then
      echo "[skipped_existing] task=$task"
    else
      echo "[failed_existing] task=$task rerun requires --overwrite" >&2
      failures=$((failures + 1))
    fi
    continue
  fi

  cleanup
  TASK_LMCACHE_DISK_DIR="$LMCACHE_DISK_ROOT/$task"
  mkdir -p "$TASK_LMCACHE_DISK_DIR"
  export LMCACHE_LOCAL_DISK="file://${TASK_LMCACHE_DISK_DIR}/"
  echo "[vLLM] restart boundary=(mode=lmcache-secondary-lookup, task=$task) cache=$TASK_LMCACHE_DISK_DIR"
  PYTHONPATH="$PYTHONPATH_VALUE" vllm serve "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL" \
    --api-key "$API_KEY" \
    --port "$PORT" \
    --dtype auto \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --enable-prefix-caching \
    --no-async-scheduling \
    --enforce-eager \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --reasoning-parser qwen3 \
    --no-enable-log-requests \
    --kv-transfer-config "$KV_TRANSFER_CONFIG" \
    >"$RUN_DIR/vllm_${task}.log" 2>&1 &
  SERVER_PID=$!

  if ! wait_vllm_ready; then
    failures=$((failures + 1))
    cleanup
    continue
  fi
  set +e
  run_driver "$task"
  status=$?
  set -e
  cleanup
  if [[ "$status" != "0" ]]; then
    failures=$((failures + 1))
    echo "[failed] task=$task status=$status"
  else
    echo "[completed] task=$task"
  fi
done

if [[ "$failures" != "0" ]]; then
  echo "$failures task(s) failed" >&2
  exit 1
fi
echo "[done] run_dir=$RUN_DIR"
