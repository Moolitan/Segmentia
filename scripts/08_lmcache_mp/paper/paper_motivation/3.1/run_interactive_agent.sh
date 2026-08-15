#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
SCRIPT_DIR="$ROOT/scripts/08_lmcache_mp/paper/paper_motivation/3.1"
VLLM_ROOT="$ROOT/vllm"
LMCACHE_ROOT="$ROOT/LMCache"
PYTHON_BIN="${PYTHON_BIN:-python}"
VLLM_BIN="${VLLM_BIN:-vllm}"
MODEL_PATH="${VLLM_MODEL_PATH:-/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B}"
SERVED_MODEL="${VLLM_SERVED_NAME:-Qwen3}"
SKILL_SAVE_POOL_ROOT="${SKILL_SAVE_POOL_ROOT:-/mnt/990_pro/skill_save_pool}"
POOL_DIR="$SKILL_SAVE_POOL_ROOT/Qwen3-14B/raw"
SKILLS_DIR="${OPENHANDS_SKILLS_DIR:-$ROOT/skills/Auto-claude-code-research-in-sleep/skills}"
EXTRA_SKILLS_DIR="${OPENHANDS_EXTRA_SKILLS_DIR:-$ROOT/skills}"
WORKSPACE="${OPENHANDS_WORKSPACE:-$ROOT/workspace/08_lmcache_mp/interactive_agent}"
PORT="${VLLM_PORT:-8014}"
API_KEY="${VLLM_API_KEY:-EMPTY}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.9}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
TOOL_CALL_PARSER="${VLLM_TOOL_CALL_PARSER:-hermes}"
REASONING_PARSER="${VLLM_REASONING_PARSER:-qwen3}"
PYTHONPATH_VALUE="$VLLM_ROOT:$LMCACHE_ROOT:$ROOT/CSKCache${PYTHONPATH:+:$PYTHONPATH}"
SERVER_LOG="$POOL_DIR/interactive_vllm.log"
AGENT_CHECK_LOG="$POOL_DIR/interactive_agent_check.log"
AGENT_RUN_LOG="$POOL_DIR/interactive_agent_run.log"
AGENT_SCRIPT="$SCRIPT_DIR/interactive_agent.py"
AGENT_MODE_ARGS=()
SCHEDULE_WINDOW_TRACE=""
SKILL_ACTION_TRACE=""
SKILL_EXECUTION_TRACE=""

if [[ "${CONDA_DEFAULT_ENV:-}" != "opencode" ]]; then
  echo "[error] activate conda environment opencode first" >&2
  exit 2
fi
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
export OPENHANDS_SUPPRESS_BANNER=1
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-256}"
export LMCACHE_USE_LAYERWISE=True
if [[ "${CSKCACHE_FINE_TIMELINE:-0}" == "1" ]]; then
  SKILL_EXECUTION_TRACE="$WORKSPACE/cskcache_skill_execution.jsonl"
  export CSKCACHE_SKILL_EXECUTION_TRACE_PATH="$SKILL_EXECUTION_TRACE"
else
  unset CSKCACHE_SKILL_EXECUTION_TRACE_PATH || true
fi

# CSKCache 模式选择（直接改这里即可，不用记命令行）
#   no_reuse : 关闭 CSKCache，走普通 prefill
#   cskcache : T0 raw-block 预取、请求认证、逐层加载和K纠错
CSKCACHE_MODE="${CSKCACHE_MODE:-cskcache}"

# 避免继承上一次服务残留的 CSKCache 配置。
unset VLLM_CSK_T0_PREFETCH LMCACHE_STORAGE_PLUGINS CSKCACHE_PROFILE || true

case "$CSKCACHE_MODE" in
  no_reuse)
    export LMCACHE_EXTRA_CONFIG='{}'
    export LMCACHE_LOCAL_DISK_REHYDRATE=False
    AGENT_SCRIPT="$SCRIPT_DIR/interactive_agent_no_reuse.py"
    SERVER_LOG="$WORKSPACE/no_reuse_vllm.log"
    AGENT_CHECK_LOG="$WORKSPACE/no_reuse_agent_check.log"
    AGENT_RUN_LOG="$WORKSPACE/no_reuse_agent_run.log"
    SCHEDULE_WINDOW_TRACE="$WORKSPACE/normal_prefill_scheduler_admission.jsonl"
    SKILL_ACTION_TRACE="$WORKSPACE/normal_prefill_skill_action_ready.jsonl"
    export VLLM_SCHEDULE_WINDOW_TRACE_PATH="$SCHEDULE_WINDOW_TRACE"
    export VLLM_SKILL_ACTION_TRACE_PATH="$SKILL_ACTION_TRACE"
    ;;
  cskcache)
    export LMCACHE_LOCAL_DISK_REHYDRATE=False
    export LMCACHE_LOCAL_CPU=True
    export LMCACHE_STORAGE_PLUGINS=raw_block
    export VLLM_CSK_T0_PREFETCH=1
    export CSKCACHE_PROFILE=1
    LMCACHE_EXTRA_CONFIG="$(
      SKILL_SAVE_POOL_ROOT="$SKILL_SAVE_POOL_ROOT" \
      PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" \
        "$SCRIPT_DIR/manage_raw_catalog.py" serve-lmcache-config
    )"
    export LMCACHE_EXTRA_CONFIG
    SCHEDULE_WINDOW_TRACE="$WORKSPACE/cskcache_scheduler_admission.jsonl"
    SKILL_ACTION_TRACE="$WORKSPACE/cskcache_skill_action_ready.jsonl"
    export VLLM_SCHEDULE_WINDOW_TRACE_PATH="$SCHEDULE_WINDOW_TRACE"
    export VLLM_SKILL_ACTION_TRACE_PATH="$SKILL_ACTION_TRACE"
    ;;
  *)
    echo "[error] unknown CSKCACHE_MODE: $CSKCACHE_MODE" >&2
    echo "[error] valid modes: no_reuse, cskcache" >&2
    exit 2
    ;;
esac

if [[ "$CSKCACHE_MODE" != "no_reuse" ]]; then
  if [[ ! -d "$POOL_DIR" ]]; then
    echo "[error] offline Skill pool does not exist: $POOL_DIR" >&2
    exit 2
  fi
  AGENT_MODE_ARGS+=(
    --pool-dir "$POOL_DIR"
    --model-path "$MODEL_PATH"
  )
fi
export LMCACHE_LOCAL_CPU="${LMCACHE_LOCAL_CPU:-False}"
export LMCACHE_MAX_LOCAL_CPU_SIZE="${LMCACHE_MAX_LOCAL_CPU_SIZE:-5}"
export LMCACHE_MAX_LOCAL_DISK_SIZE="${LMCACHE_MAX_LOCAL_DISK_SIZE:-1000}"
if [[ "$CSKCACHE_MODE" == "no_reuse" ]]; then
  export LMCACHE_LOCAL_DISK="file://${WORKSPACE}/.lmcache_no_reuse/"
elif [[ "$CSKCACHE_MODE" == "cskcache" ]]; then
  unset LMCACHE_LOCAL_DISK
else
  export LMCACHE_LOCAL_DISK="file://${POOL_DIR}/"
fi
export LMCACHE_FORCE_SKIP_SAVE=1

SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

mkdir -p "$WORKSPACE"
if [[ -n "$SCHEDULE_WINDOW_TRACE" ]]; then
  : > "$SCHEDULE_WINDOW_TRACE"
fi
if [[ -n "$SKILL_ACTION_TRACE" ]]; then
  : > "$SKILL_ACTION_TRACE"
fi
if [[ -n "$SKILL_EXECUTION_TRACE" ]]; then
  : > "$SKILL_EXECUTION_TRACE"
fi
echo "[agent] running pre-flight check; log: $AGENT_CHECK_LOG"
PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$AGENT_SCRIPT" \
  --skills-dir "$SKILLS_DIR" \
  --extra-skills-dir "$EXTRA_SKILLS_DIR" \
  --served-model "$SERVED_MODEL" \
  --base-url "http://127.0.0.1:$PORT" \
  --api-key "$API_KEY" \
  --workspace "$WORKSPACE" \
  "${AGENT_MODE_ARGS[@]}" \
  "$@" \
  --check >"$AGENT_CHECK_LOG" 2>&1

echo "[server] starting vLLM; log: $SERVER_LOG"
PYTHONPATH="$PYTHONPATH_VALUE" "$VLLM_BIN" serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL" \
  --api-key "$API_KEY" \
  --port "$PORT" \
  --dtype auto \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
  --enforce-eager \
  --no-enable-log-requests \
  --enable-prefix-caching \
  --no-async-scheduling \
  --enable-auto-tool-choice \
  --tool-call-parser "$TOOL_CALL_PARSER" \
  --reasoning-parser "$REASONING_PARSER" \
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
PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$AGENT_SCRIPT" \
  --skills-dir "$SKILLS_DIR" \
  --extra-skills-dir "$EXTRA_SKILLS_DIR" \
  --served-model "$SERVED_MODEL" \
  --base-url "http://127.0.0.1:$PORT" \
  --api-key "$API_KEY" \
  --workspace "$WORKSPACE" \
  "${AGENT_MODE_ARGS[@]}" \
  "$@" 2>&1 | tee "$AGENT_RUN_LOG"
