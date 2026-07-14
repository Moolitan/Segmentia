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
KV_DIR="${CSKCACHE_KV_DIR:-/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/07_cskcache/offline_skill_kv}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${CSKCACHE_AGENT_RUN_DIR:-/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/07_cskcache/real_agent_runs/$RUN_ID}"
PROFILE_ENABLED="${CSKCACHE_PROFILE_ENABLED:-0}"
PROFILE_JSONL_OVERRIDE="${CSKCACHE_PROFILE_JSONL:-}"
TASKS="${TASKS:-internal_comms_incident_update,doc_coauthoring_design_doc,mcp_server_and_spec,web_artifact_with_theme,launch_poster_page_pack,slack_launch_pack}"
READY_ATTEMPTS="${VLLM_READY_MAX_ATTEMPTS:-450}"
READY_INTERVAL="${VLLM_READY_INTERVAL:-2}"

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
export PYTHONPATH="$CSKCACHE_ROOT:$VLLM_ROOT:$ROOT/software-agent-sdk/openhands-sdk:$ROOT/software-agent-sdk/openhands-tools${PYTHONPATH:+:$PYTHONPATH}"
export CSKCACHE_DISK_DIR="$KV_DIR"
export CSKCACHE_CAPTURE_ONLY=1
unset CSKCACHE_CPU_MAX_BYTES || true
export VLLM_API_KEY="$API_KEY"
export OPENHANDS_SUPPRESS_BANNER=1
export LITELLM_LOCAL_MODEL_COST_MAP=True
unset LD_LIBRARY_PATH

KV_TRANSFER_CONFIG='{"kv_connector":"CSKCacheConnectorV1","kv_connector_module_path":"cskcache.integration.vllm.v1_connector","kv_role":"kv_both"}'
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

run_driver() {
  local task="$1"
  local args=(
    --benchmark-repo "$task"
    --bench-root "$ROOT/anthropic_skill_benchmark"
    --skills-dir "$ROOT/skills"
    --kv-dir "$KV_DIR"
    --run-dir "$RUN_DIR"
    --model-path "$MODEL_PATH"
    --model "$SERVED_MODEL"
    --vllm-port "$PORT"
  )
  [[ "$OVERWRITE" == "1" ]] && args+=(--overwrite)
  [[ "$DRY_RUN" == "1" ]] && args+=(--dry-run)
  python "$ROOT/scripts/07_cskcache/run_real_agent.py" "${args[@]}"
}

IFS=',' read -ra TASK_LIST <<< "$TASKS"
if [[ "$DRY_RUN" == "1" ]]; then
  for task in "${TASK_LIST[@]}"; do
    task="${task// /}"
    [[ -z "$task" ]] || run_driver "$task"
  done
  exit 0
fi

mkdir -p "$RUN_DIR"
failures=0
for task in "${TASK_LIST[@]}"; do
  task="${task// /}"
  [[ -z "$task" ]] && continue
  if [[ -f "$RUN_DIR/$task/_summary.json" && "$OVERWRITE" == "0" ]] &&
     grep -q '"status": "completed"' "$RUN_DIR/$task/_summary.json"; then
    echo "[skipped_existing] task=$task"
    continue
  fi
  if [[ -e "$RUN_DIR/$task" && "$OVERWRITE" == "0" ]]; then
    failures=$((failures + 1))
    echo "[failed_existing] task=$task requires --overwrite" >&2
    continue
  fi

  cleanup
  echo "[vLLM] restart boundary=(mode=cskcache, task=$task)"
  export CSKCACHE_PROFILE_ENABLED="$PROFILE_ENABLED"
  if [[ -n "$PROFILE_JSONL_OVERRIDE" ]]; then
    export CSKCACHE_PROFILE_JSONL="$PROFILE_JSONL_OVERRIDE"
  elif [[ "${PROFILE_ENABLED,,}" =~ ^(1|true|yes|on)$ ]]; then
    export CSKCACHE_PROFILE_JSONL="$RUN_DIR/profile_${task}.jsonl"
  else
    unset CSKCACHE_PROFILE_JSONL || true
  fi
  cd "$VLLM_ROOT"
  LD_LIBRARY_PATH="${CONDA_PREFIX}/lib" \
  python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL" \
    --api-key "$API_KEY" \
    --port "$PORT" \
    --dtype auto \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --enable-prefix-caching \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --reasoning-parser qwen3 \
    --no-enable-log-requests \
    --kv-transfer-config "$KV_TRANSFER_CONFIG" \
    >"$RUN_DIR/vllm_${task}.log" 2>&1 &
  SERVER_PID=$!

  cd "$ROOT"
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
