#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

unset LD_LIBRARY_PATH
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy ftp_proxy FTP_PROXY

export VLLM_PORT="${VLLM_PORT:-8000}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
export VLLM_SERVED_NAME="${VLLM_SERVED_NAME:-Qwen3}"

RUN_ROOT="${RUN_ROOT:-$ROOT/results/05_context_segment_agent_kv/internal_comms_incident_update}"
export VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR="${VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR:-$RUN_ROOT/kv_cache}"
export VLLM_CONTEXT_SEGMENT_KV_DIR="${VLLM_CONTEXT_SEGMENT_KV_DIR:-$VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR}"
mkdir -p "$VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR" "$ROOT/log/05_context_segment_agent_kv"

VLLM_READY_MAX_ATTEMPTS="${VLLM_READY_MAX_ATTEMPTS:-180}"
VLLM_READY_INTERVAL="${VLLM_READY_INTERVAL:-2}"

vllm_probe_ok() {
  local code_m code_h
  code_m="$(
    curl -sS -o /dev/null -w '%{http_code}' \
      --connect-timeout 5 --max-time 15 \
      -H "Authorization: Bearer ${VLLM_API_KEY}" \
      "http://127.0.0.1:${VLLM_PORT}/v1/models" 2>/dev/null || true
  )"
  [[ -z "$code_m" ]] && code_m="000"
  if [[ "$code_m" == "200" || "$code_m" == "401" || "$code_m" == "403" ]]; then
    return 0
  fi
  code_h="$(
    curl -sS -o /dev/null -w '%{http_code}' \
      --connect-timeout 5 --max-time 15 \
      "http://127.0.0.1:${VLLM_PORT}/health" 2>/dev/null || true
  )"
  [[ -z "$code_h" ]] && code_h="000"
  [[ "$code_h" == "200" ]]
}

wait_vllm_ready() {
  local attempt=0
  echo "[vLLM] waiting on port ${VLLM_PORT}"
  while (( attempt < VLLM_READY_MAX_ATTEMPTS )); do
    if vllm_probe_ok; then
      echo "[vLLM] ready"
      return 0
    fi
    attempt=$((attempt + 1))
    if (( attempt % 15 == 0 )); then
      echo "[vLLM] still waiting ${attempt}/${VLLM_READY_MAX_ATTEMPTS}"
    fi
    sleep "$VLLM_READY_INTERVAL"
  done
  echo "[vLLM] timeout; check $ROOT/log/vllm.log" >&2
  return 1
}

echo "[vLLM] stop existing server"
bash "$ROOT/scripts/vllm_stop.sh"

echo "[vLLM] start for real multiturn ContextSegmentKV run"
echo "  VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR=$VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR"
echo "  VLLM_CONTEXT_SEGMENT_KV_DIR=$VLLM_CONTEXT_SEGMENT_KV_DIR"
bash "$ROOT/scripts/vllm_start.sh"
sleep 2
wait_vllm_ready

python "$SCRIPT_DIR/run_real_multurn_context_segment.py" \
  --benchmark-repo internal_comms_incident_update \
  --bench-root "$ROOT/anthropic_skill_benchmark_8_repos_explicit_skills" \
  --workspace "$ROOT/workspace/05_context_segment_agent_kv/internal_comms_incident_update" \
  --skills-dir "$ROOT/skills" \
  --output "$RUN_ROOT/multiturn_sequence_traces.json" \
  --log-dir "$ROOT/log/05_context_segment_agent_kv" \
  --vllm-port "$VLLM_PORT" \
  --model "$VLLM_SERVED_NAME" \
  "$@"

echo "[done] result: $RUN_ROOT/multiturn_sequence_traces.json"
echo "[done] kv dir: $VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR"
