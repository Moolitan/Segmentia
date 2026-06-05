#!/usr/bin/env bash
# Trace-driven recompute-vs-reuse CKSim.
#
# Runs independent vLLM lifecycles per task. For each task, dump full-context
# recompute KV, restart vLLM for RoPE reuse, then restart vLLM again for
# direct/no-rope reuse. Finally computes CKSim across tasks.
#
# Usage:
#   bash run_cksim.sh
#   TASKS=slack_launch_pack bash run_cksim.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$ROOT"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true

export VLLM_PORT="${VLLM_PORT:-8000}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export VLLM_SERVED_NAME="${VLLM_SERVED_NAME:-Qwen3}"
export VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.86}"

KV_SAVE_DIR="${KV_SAVE_DIR:-$ROOT/results/05_context_segment_agent_kv/CKSim/kv_cache_trace}"
TASKS="${TASKS:-all}"

DEFAULT_TASKS=(
  internal_comms_incident_update
  doc_coauthoring_design_doc
  # mcp_server_and_spec
  web_artifact_with_theme
  # launch_poster_page_pack
  # slack_launch_pack
)

if [[ "$TASKS" == "all" ]]; then
  TASK_LIST=("${DEFAULT_TASKS[@]}")
else
  IFS=',' read -r -a TASK_LIST <<< "$TASKS"
fi

rm -rf "$KV_SAVE_DIR"
mkdir -p "$KV_SAVE_DIR"

wait_for_vllm() {
  echo "[vLLM] wait for readiness"
  for i in $(seq 1 180); do
    code="$(curl -sS -o /dev/null -w '%{http_code}' --connect-timeout 3 --max-time 10 \
      -H "Authorization: Bearer ${VLLM_API_KEY}" \
      "http://127.0.0.1:${VLLM_PORT}/v1/models" 2>/dev/null || true)"
    [[ "$code" == "200" ]] && { echo "[vLLM] ready"; return 0; }
    if grep -qE "OutOfMemoryError|Engine core initialization failed" "$ROOT/log/vllm.log" 2>/dev/null; then
      echo "[error] vLLM failed during startup; recent log:" >&2
      tail -n 80 "$ROOT/log/vllm.log" >&2 || true
      return 1
    fi
    sleep 2
  done
  echo "[error] vLLM not ready at http://127.0.0.1:${VLLM_PORT}" >&2
  return 1
}

start_vllm_for_cksim() {
  export VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR="$KV_SAVE_DIR"
  # Do not preload the accumulated .pt files on each restart. Each phase
  # recollects its own source KV in-process, while summarize reads .pt files
  # offline after vLLM is stopped.
  unset VLLM_CONTEXT_SEGMENT_KV_DIR
  bash "$ROOT/scripts/vllm_stop.sh" || true
  sleep "${VLLM_RESTART_SLEEP:-5}"
  bash "$ROOT/scripts/vllm_start.sh"
  wait_for_vllm
}

export VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR="$KV_SAVE_DIR"

for task in "${TASK_LIST[@]}"; do
  task="${task#"${task%%[![:space:]]*}"}"
  task="${task%"${task##*[![:space:]]}"}"
  [[ -z "$task" ]] && continue

  echo "[task] $task"
  echo "[phase] recompute"
  start_vllm_for_cksim
  python "$SCRIPT_DIR/trace_reuse_cksim.py" \
    --phase recompute \
    --tasks "$task" \
    --vllm-port "$VLLM_PORT" \
    --model "$VLLM_SERVED_NAME" \
    --kv-dir "$KV_SAVE_DIR"

  echo "[phase] reuse"
  start_vllm_for_cksim
  python "$SCRIPT_DIR/trace_reuse_cksim.py" \
    --phase reuse \
    --tasks "$task" \
    --vllm-port "$VLLM_PORT" \
    --model "$VLLM_SERVED_NAME" \
    --kv-dir "$KV_SAVE_DIR"

  echo "[phase] reuse_no_rope"
  start_vllm_for_cksim
  python "$SCRIPT_DIR/trace_reuse_cksim.py" \
    --phase reuse_no_rope \
    --tasks "$task" \
    --vllm-port "$VLLM_PORT" \
    --model "$VLLM_SERVED_NAME" \
    --kv-dir "$KV_SAVE_DIR"
done

bash "$ROOT/scripts/vllm_stop.sh" || true

echo "[phase] summarize"
python "$SCRIPT_DIR/trace_reuse_cksim.py" \
  --phase summarize \
  --tasks "$TASKS" \
  --vllm-port "$VLLM_PORT" \
  --model "$VLLM_SERVED_NAME" \
  --kv-dir "$KV_SAVE_DIR"
