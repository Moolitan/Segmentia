#!/usr/bin/env bash
# Collect task-specific occurrence-1 skill KV with strict per-task isolation.
#
# Lifecycle:
#   for task:
#     restart vLLM (clears prefix cache and in-process KV registry)
#     collect that task's skills in first-invocation order
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true

export VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B}"
export VLLM_SERVED_NAME="${VLLM_SERVED_NAME:-Qwen3}"
export VLLM_PORT="${VLLM_PORT:-8000}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"

DEFAULT_CONTEXTUAL_KV_DIR="$(
  PYTHONPATH="$SCRIPT_DIR/module" python -c \
    'from config import DEFAULT_CONTEXTUAL_KV_DIR; print(DEFAULT_CONTEXTUAL_KV_DIR)'
)"
KV_DIR="${KV_DIR:-$DEFAULT_CONTEXTUAL_KV_DIR}"
MANIFEST_DIR="${MANIFEST_DIR:-$KV_DIR/manifests}"
TASKS="${TASKS:-internal_comms_incident_update,doc_coauthoring_design_doc,mcp_server_and_spec,web_artifact_with_theme,launch_poster_page_pack,slack_launch_pack}"
MAX_TOKENS="${MAX_TOKENS:-1}"

mkdir -p "$KV_DIR" "$MANIFEST_DIR"

cleanup() {
  bash "$ROOT/scripts/vllm_stop.sh" || true
}
trap cleanup EXIT

start_vllm_for_task() {
  local task="$1"

  echo ""
  echo "[vLLM] restart boundary=(mode=contextual_occ1_prefill, task=$task)"
  unset VLLM_CONTEXT_SEGMENT_KV_DIR || true
  export VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR="$KV_DIR"

  bash "$ROOT/scripts/vllm_stop.sh" || true
  bash "$ROOT/scripts/vllm_start.sh"

  local ready=0
  local code=""
  for _poll_i in $(seq 1 600); do
    code="$(curl -sS -o /dev/null -w '%{http_code}' \
      --connect-timeout 3 --max-time 10 \
      -H "Authorization: Bearer ${VLLM_API_KEY}" \
      "http://127.0.0.1:${VLLM_PORT}/v1/models" 2>/dev/null || true)"
    if [[ "$code" == "200" ]]; then
      ready=1
      echo "[vLLM] ready task=$task"
      break
    fi
    sleep 2
  done
  if [[ "$ready" != "1" ]]; then
    echo "[error] vLLM start timeout for task=$task" >&2
    exit 1
  fi
}

IFS=',' read -ra TASK_LIST <<< "$TASKS"

for task in "${TASK_LIST[@]}"; do
  task="${task// /}"
  [[ -z "$task" ]] && continue

  start_vllm_for_task "$task"
  python "$SCRIPT_DIR/prefill_contextual_skill_kv.py" \
    --tasks "$task" \
    --vllm-port "$VLLM_PORT" \
    --model "$VLLM_SERVED_NAME" \
    --api-key "$VLLM_API_KEY" \
    --kv-dir "$KV_DIR" \
    --manifest "$MANIFEST_DIR/${task}.json" \
    --max-tokens "$MAX_TOKENS"
done

echo ""
echo "[done] contextual occurrence-1 KV: $KV_DIR"
echo "[done] per-task manifests: $MANIFEST_DIR"
