#!/usr/bin/env bash
# Free-generation thinking -> action boundary diagnostic for Segmentia.
#
# This wrapper restarts vLLM per (task, mode) to keep prefix-cache state
# isolated. It is meant to be run by the user, not automatically by the agent.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true

export VLLM_PORT="${VLLM_PORT:-8000}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export VLLM_SERVED_NAME="${VLLM_SERVED_NAME:-Qwen3}"

RES="$ROOT/results/problem_exploration"
OUT_DIR="$RES/thinking_to_action_divergence"
SEGMENTIA_OUTPUT_DIR="${SEGMENTIA_OUTPUT_DIR:-/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/06_context_free_segment_cache}"

ALL_TASKS="internal_comms_incident_update doc_coauthoring_design_doc mcp_server_and_spec web_artifact_with_theme launch_poster_page_pack slack_launch_pack"
TASKS="${TASKS:-all}"
MODES="${MODES:-recompute,rope}"
OCCURRENCES="${OCCURRENCES:-2,3}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
TOP_LOGPROBS="${TOP_LOGPROBS:-10}"
KV_DIR="${KV_DIR:-$SEGMENTIA_OUTPUT_DIR/offline_skill_kv}"
FREE_JSONL="${FREE_JSONL:-$OUT_DIR/data/free_generation_rows.jsonl}"
TOKEN_JSONL="${TOKEN_JSONL:-$OUT_DIR/data/token_logprob_rows.jsonl}"
CASE_CSV="${CASE_CSV:-$OUT_DIR/tables/thinking_action_case_summary.csv}"

if [[ "$TASKS" == "all" ]]; then
  IFS=' ' read -ra TASK_LIST <<< "$ALL_TASKS"
else
  IFS=',' read -ra TASK_LIST <<< "$TASKS"
fi

IFS=',' read -ra MODE_LIST <<< "$MODES"

start_vllm() {
  local label="$1"   # recompute | injection
  echo ""
  echo "[vLLM] restart for $label (empty prefix cache)"
  unset VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR || true
  if [[ "$label" == "recompute" ]]; then
    unset VLLM_CONTEXT_SEGMENT_KV_DIR || true
  else
    export VLLM_CONTEXT_SEGMENT_KV_DIR="$KV_DIR"
  fi

  bash "$ROOT/scripts/vllm_stop.sh" || true
  bash "$ROOT/scripts/vllm_start.sh"

  local ready=0
  for _poll_i in $(seq 1 600); do
    code="$(curl -sS -o /dev/null -w '%{http_code}' --connect-timeout 3 --max-time 10 \
      -H "Authorization: Bearer ${VLLM_API_KEY}" \
      "http://127.0.0.1:${VLLM_PORT}/v1/models" 2>/dev/null || true)"
    if [[ "$code" == "200" ]]; then
      echo "[vLLM] ready"
      ready=1
      break
    fi
    sleep 2
  done
  if [[ "$ready" != "1" ]]; then
    echo "[vLLM] ERROR: server did not become ready within timeout (label=$label)" >&2
    exit 1
  fi
}

mkdir -p "$OUT_DIR/data" "$OUT_DIR/tables" "$OUT_DIR/figures"

for task in "${TASK_LIST[@]}"; do
  task="${task//,/}"
  task="${task// /}"
  [[ -z "$task" ]] && continue

  echo ""
  echo "========== task=$task =========="

  for mode in "${MODE_LIST[@]}"; do
    mode="${mode// /}"
    [[ -z "$mode" ]] && continue

    if [[ "$mode" == "recompute" ]]; then
      start_vllm recompute
    else
      start_vllm injection
    fi

    echo "--- thinking/action diagnostic mode=$mode task=$task ---"
    python "$SCRIPT_DIR/run_thinking_action_diagnostic.py" \
      --tasks "$task" \
      --modes "$mode" \
      --occurrences "$OCCURRENCES" \
      --vllm-port "$VLLM_PORT" \
      --model "$VLLM_SERVED_NAME" \
      --api-key "$VLLM_API_KEY" \
      --kv-dir "$KV_DIR" \
      --free-jsonl "$FREE_JSONL" \
      --token-jsonl "$TOKEN_JSONL" \
      --case-csv "$CASE_CSV" \
      --max-tokens "$MAX_TOKENS" \
      --top-logprobs "$TOP_LOGPROBS" \
      --append \
      --skip-existing
  done
done

echo ""
echo "[done] free rows: $FREE_JSONL"
echo "[done] token rows: $TOKEN_JSONL"
echo "[done] case summary: $CASE_CSV"
