#!/usr/bin/env bash
# Replay saved agent traces (src/traces/) against a modified vLLM, driving
# skill KV reuse via ContextSegmentKV. No agent framework involved.
#
# Usage:
#   bash run_replay.sh                       # recompute + reuse, all tasks
#   REUSE_SCOPE=cross-task bash run_replay.sh # cross-task skill reuse
#   TASKS=internal_comms_incident_update,slack_launch_pack bash run_replay.sh
#   MODE=reuse bash run_replay.sh
#   MODE=recompute bash run_replay.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$ROOT"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true

export VLLM_PORT="${VLLM_PORT:-8000}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export VLLM_SERVED_NAME="${VLLM_SERVED_NAME:-Qwen3}"

TASKS="${TASKS:-all}"
MODE="${MODE:-all}"
REUSE_SCOPE="${REUSE_SCOPE:-per-task}"
OUTPUT="${OUTPUT:-$ROOT/results/05_context_segment_agent_kv/replay/replay_${REUSE_SCOPE}_${MODE}.json}"

# Restart vLLM so the ContextSegmentKV in-memory registry starts empty. Source
# KVs are collected online during the run; no .pt persistence is used here.
if [[ "${RESTART_VLLM:-1}" == "1" ]]; then
  echo "[vLLM] restart (empty in-memory ContextSegmentKV registry)"
  unset VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR
  unset VLLM_CONTEXT_SEGMENT_KV_DIR
  bash "$ROOT/scripts/vllm_stop.sh" || true
  bash "$ROOT/scripts/vllm_start.sh"
  # Wait for readiness.
  for i in $(seq 1 180); do
    code="$(curl -sS -o /dev/null -w '%{http_code}' --connect-timeout 3 --max-time 10 \
      -H "Authorization: Bearer ${VLLM_API_KEY}" \
      "http://127.0.0.1:${VLLM_PORT}/v1/models" 2>/dev/null || true)"
    [[ "$code" == "200" ]] && { echo "[vLLM] ready"; break; }
    sleep 2
  done
fi

python "$SCRIPT_DIR/replay_trace_context_segment.py" \
  --tasks "$TASKS" \
  --mode "$MODE" \
  --reuse-scope "$REUSE_SCOPE" \
  --vllm-port "$VLLM_PORT" \
  --model "$VLLM_SERVED_NAME" \
  --output "$OUTPUT"

echo "[done] result: $OUTPUT"
