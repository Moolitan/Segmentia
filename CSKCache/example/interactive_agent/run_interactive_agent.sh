#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
VLLM_ROOT="$ROOT/vllm"
LMCACHE_ROOT="$ROOT/LMCache"
CSKCACHE_ROOT="$ROOT/CSKCache"
PYTHON_BIN="${PYTHON_BIN:-python}"
VLLM_BIN="${VLLM_BIN:-vllm}"

if (($#)); then
  echo "[error] this launcher accepts no arguments; edit config.py instead" >&2
  exit 2
fi
eval "$("$PYTHON_BIN" "$SCRIPT_DIR/config_loader.py")"

MODEL_PATH="$VLLM_MODEL_PATH"
SERVED_MODEL="$VLLM_SERVED_NAME"
PORT="$VLLM_PORT"
API_KEY="$VLLM_API_KEY"
WORKSPACE="$INTERACTIVE_WORKSPACE"
POOL_DIR="$INTERACTIVE_POOL_DIR"
PYTHONPATH_VALUE="$SCRIPT_DIR:$VLLM_ROOT:$LMCACHE_ROOT:$CSKCACHE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "opencode" ]]; then
  echo "[error] activate conda environment opencode first" >&2
  exit 2
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
export OPENHANDS_SUPPRESS_BANNER=1
export PYTHONHASHSEED=0
export LMCACHE_USE_LAYERWISE=True
export LMCACHE_FORCE_SKIP_SAVE=1

if [[ "$INTERACTIVE_MODE" == "cskcache" ]]; then
  export LMCACHE_LOCAL_CPU=True
  export VLLM_CSK_T0_PREFETCH=1
  if [[ "$INTERACTIVE_STORAGE_BACKEND" == "local_disk" ]]; then
    export LMCACHE_LOCAL_DISK
  else
    unset LMCACHE_LOCAL_DISK || true
  fi
else
  export LMCACHE_LOCAL_CPU=False
  unset VLLM_CSK_T0_PREFETCH LMCACHE_LOCAL_DISK || true
fi

mkdir -p "$WORKSPACE"
SERVER_LOG="$WORKSPACE/${INTERACTIVE_MODE}_vllm.log"
AGENT_CHECK_LOG="$WORKSPACE/${INTERACTIVE_MODE}_agent_check.log"
AGENT_RUN_LOG="$WORKSPACE/${INTERACTIVE_MODE}_agent_run.log"
export VLLM_SCHEDULE_WINDOW_TRACE_PATH="$WORKSPACE/${INTERACTIVE_MODE}_scheduler_admission.jsonl"
export VLLM_SKILL_ACTION_TRACE_PATH="$WORKSPACE/${INTERACTIVE_MODE}_skill_action_ready.jsonl"
: > "$VLLM_SCHEDULE_WINDOW_TRACE_PATH"
: > "$VLLM_SKILL_ACTION_TRACE_PATH"

if [[ "$CSKCACHE_PROFILE" == "1" ]]; then
  export CSKCACHE_PROFILE_TRACE_PATH="$WORKSPACE/cskcache_profile.jsonl"
  : > "$CSKCACHE_PROFILE_TRACE_PATH"
else
  unset CSKCACHE_PROFILE_TRACE_PATH || true
fi
if [[ "$CSKCACHE_FINE_TIMELINE" == "1" ]]; then
  export CSKCACHE_SKILL_EXECUTION_TRACE_PATH="$WORKSPACE/cskcache_skill_execution.jsonl"
  : > "$CSKCACHE_SKILL_EXECUTION_TRACE_PATH"
else
  unset CSKCACHE_SKILL_EXECUTION_TRACE_PATH || true
fi

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[agent] checking config.py and offline Catalog; log: $AGENT_CHECK_LOG"
PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" -c \
  'from interactive_agent import main; main(check=True)' \
  >"$AGENT_CHECK_LOG" 2>&1

echo "[server] mode=$INTERACTIVE_MODE backend=$INTERACTIVE_STORAGE_BACKEND"
echo "[server] pool=$POOL_DIR log=$SERVER_LOG"
PYTHONPATH="$PYTHONPATH_VALUE" "$VLLM_BIN" serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL" \
  --api-key "$API_KEY" \
  --port "$PORT" \
  --dtype auto \
  --max-model-len "$VLLM_MAX_MODEL_LEN" \
  --gpu-memory-utilization "$VLLM_GPU_UTIL" \
  --kv-transfer-config "$VLLM_KV_TRANSFER_CONFIG" \
  --enforce-eager \
  --no-enable-log-requests \
  --enable-prefix-caching \
  --no-async-scheduling \
  --enable-auto-tool-choice \
  --tool-call-parser "$VLLM_TOOL_CALL_PARSER" \
  --reasoning-parser "$VLLM_REASONING_PARSER" \
  >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

ready=0
for _ in $(seq 1 450); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[error] vLLM exited before readiness" >&2
    tail -n 60 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  code="$(curl -sS -o /dev/null -w '%{http_code}' \
    --connect-timeout 3 --max-time 10 \
    -H "Authorization: Bearer $API_KEY" \
    "http://127.0.0.1:$PORT/v1/models" 2>/dev/null || true)"
  if [[ "$code" == "200" ]]; then
    ready=1
    break
  fi
  sleep 2
done
if [[ "$ready" != "1" ]]; then
  echo "[error] vLLM readiness timeout; log: $SERVER_LOG" >&2
  exit 1
fi

echo "[agent] starting interactive run; log: $AGENT_RUN_LOG"
PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" \
  "$SCRIPT_DIR/interactive_agent.py" 2>&1 | tee "$AGENT_RUN_LOG"
