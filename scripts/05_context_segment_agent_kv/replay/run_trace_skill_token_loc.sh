#!/usr/bin/env bash
# Collect skill occurrence token locations from each task's final cumulative
# trace JSON. By default this calls vLLM /tokenize, so vLLM must be running.
#
# Usage:
#   bash run_trace_skill_token_loc.sh
#   TASKS=internal_comms_incident_update bash run_trace_skill_token_loc.sh
#   CHAR_ONLY=1 bash run_trace_skill_token_loc.sh
#   RESTART_VLLM=0 bash run_trace_skill_token_loc.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$ROOT"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true

export VLLM_PORT="${VLLM_PORT:-8000}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export VLLM_SERVED_NAME="${VLLM_SERVED_NAME:-Qwen3}"

TASKS="${TASKS:-all}"
CHAR_ONLY="${CHAR_ONLY:-0}"
OUTPUT="${OUTPUT:-$ROOT/results/05_context_segment_agent_kv/replay/skill_token_locations.json}"
SYSTEM_PREFIX="${SYSTEM_PREFIX:-}"

if [[ "$CHAR_ONLY" != "1" ]]; then
  if [[ "${RESTART_VLLM:-1}" == "1" ]]; then
    echo "[vLLM] restart for tokenizer"
    bash "$ROOT/scripts/vllm_stop.sh" || true
    bash "$ROOT/scripts/vllm_start.sh"
  fi

  echo "[vLLM] wait for readiness"
  ready=0
  for _ in $(seq 1 180); do
    code="$(curl -sS -o /dev/null -w '%{http_code}' --connect-timeout 3 --max-time 10 \
      -H "Authorization: Bearer ${VLLM_API_KEY}" \
      "http://127.0.0.1:${VLLM_PORT}/v1/models" 2>/dev/null || true)"
    if [[ "$code" == "200" ]]; then
      ready=1
      echo "[vLLM] ready"
      break
    fi
    sleep 2
  done

  if [[ "$ready" != "1" ]]; then
    echo "[error] vLLM not ready at http://127.0.0.1:${VLLM_PORT}" >&2
    exit 1
  fi
fi

args=(
  "$SCRIPT_DIR/trace_skill_token_loc.py"
  --tasks "$TASKS"
  --vllm-port "$VLLM_PORT"
  --model "$VLLM_SERVED_NAME"
  --output "$OUTPUT"
)

if [[ -n "$SYSTEM_PREFIX" ]]; then
  args+=(--system-prefix "$SYSTEM_PREFIX")
fi

if [[ "$CHAR_ONLY" == "1" ]]; then
  args+=(--char-only)
fi

python "${args[@]}"

echo "[done] result: $OUTPUT"
