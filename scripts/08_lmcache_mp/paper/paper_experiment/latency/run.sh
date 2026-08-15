#!/usr/bin/env bash
# Real-Agent latency comparison: normal Recompute versus current CSKCache.
# All experiment parameters live in config.py; run this file without arguments.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
SCRIPT_DIR="$ROOT/scripts/08_lmcache_mp/paper/paper_experiment/latency"
MOTIVATION_DIR="$ROOT/scripts/08_lmcache_mp/paper/paper_motivation/3.1"
PYTHON_BIN="${PYTHON_BIN:-python}"
VLLM_BIN="${VLLM_BIN:-vllm}"
PYTHONPATH_VALUE="$SCRIPT_DIR:$ROOT/vllm:$ROOT/LMCache:$ROOT/CSKCache:$MOTIVATION_DIR${PYTHONPATH:+:$PYTHONPATH}"

if (($#)); then
  echo "[error] this experiment takes no command-line arguments; edit config.py" >&2
  exit 2
fi
if [[ "${CONDA_DEFAULT_ENV:-}" != "opencode" ]]; then
  echo "[error] activate conda environment opencode first" >&2
  exit 2
fi

eval "$(PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" - <<'PY'
import shlex
from config import (
    API_KEY, GPU_MEMORY_UTILIZATION, MAX_AGENT_ITERATIONS, MAX_MODEL_LEN,
    MEASURE_CASES, MODEL_PATH, OUTPUT_ROOT, PORT, PROMPT_FILE, RAW_POOL_DIR,
    REPLICAS, SERVED_MODEL, SKILL, SKILL_SAVE_POOL_ROOT, WARMUP_CASES,
)
values = locals()
for name in (
    "API_KEY", "GPU_MEMORY_UTILIZATION", "MAX_AGENT_ITERATIONS",
    "MAX_MODEL_LEN", "MEASURE_CASES", "MODEL_PATH", "OUTPUT_ROOT", "PORT",
    "PROMPT_FILE", "RAW_POOL_DIR", "REPLICAS", "SERVED_MODEL", "SKILL",
    "SKILL_SAVE_POOL_ROOT", "WARMUP_CASES",
):
    print(f"{name}={shlex.quote(str(values[name]))}")
PY
)"

RUN_ID="${CSKCACHE_LATENCY_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
mkdir -p "$RUN_DIR"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
export OPENHANDS_SUPPRESS_BANNER=1
export PYTHONHASHSEED=0
export LMCACHE_CHUNK_SIZE=256
export LMCACHE_USE_LAYERWISE=True
export LMCACHE_FORCE_SKIP_SAVE=1
export LMCACHE_MAX_LOCAL_CPU_SIZE=5
export LMCACHE_MAX_LOCAL_DISK_SIZE=1000
export CSKCACHE_DISABLE_VISUALIZER=1
export CSKCACHE_FINE_TIMELINE=0
# vLLM intentionally registers cache-control endpoints only in server dev
# mode.  This experiment needs /reset_prefix_cache between independent cases;
# production request handling and the CSKCache data path remain unchanged.
export VLLM_SERVER_DEV_MODE=1

SERVER_PID=""
stop_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap stop_server EXIT INT TERM

wait_ready() {
  for _ in $(seq 1 450); do
    if [[ -n "$SERVER_PID" ]] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
      return 1
    fi
    code="$(curl -sS -o /dev/null -w '%{http_code}' --connect-timeout 3 \
      --max-time 10 -H "Authorization: Bearer $API_KEY" \
      "http://127.0.0.1:$PORT/v1/models" 2>/dev/null || true)"
    [[ "$code" == "200" ]] && return 0
    sleep 2
  done
  return 1
}

reset_prefix_cache() {
  local response_file
  local status="000"
  local body=""
  response_file="$(mktemp)"
  for _ in $(seq 1 20); do
    status="$(curl -sS -o "$response_file" -w '%{http_code}' -X POST --max-time 30 \
      -H "Authorization: Bearer $API_KEY" \
      "http://127.0.0.1:$PORT/reset_prefix_cache?reset_external=false" || true)"
    body="$(<"$response_file")"
    if [[ "$status" == "200" && "$body" == *'"success":true'* ]]; then
      rm -f "$response_file"
      return 0
    fi
    sleep 1
  done
  rm -f "$response_file"
  echo "[error] vLLM prefix cache reset did not succeed: HTTP $status body=$body" >&2
  return 1
}

start_server() {
  local mode="$1"
  local leaf="$2"
  unset VLLM_CSK_T0_PREFETCH LMCACHE_STORAGE_PLUGINS CSKCACHE_PROFILE \
    CSKCACHE_PROFILE_TRACE_PATH LMCACHE_LOCAL_DISK || true
  export VLLM_REQUEST_TIMELINE_PATH="$leaf/vllm_request_timeline.jsonl"
  : > "$VLLM_REQUEST_TIMELINE_PATH"
  export LMCACHE_LOCAL_DISK_REHYDRATE=False
  if [[ "$mode" == "cskcache" ]]; then
    export LMCACHE_LOCAL_CPU=True
    export LMCACHE_STORAGE_PLUGINS=raw_block
    export VLLM_CSK_T0_PREFETCH=1
    export CSKCACHE_PROFILE=1
    export CSKCACHE_PROFILE_TRACE_PATH="$leaf/cskcache_profile.jsonl"
    : > "$CSKCACHE_PROFILE_TRACE_PATH"
    export LMCACHE_EXTRA_CONFIG
    LMCACHE_EXTRA_CONFIG="$(
      SKILL_SAVE_POOL_ROOT="$SKILL_SAVE_POOL_ROOT" \
      PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" \
        "$MOTIVATION_DIR/manage_raw_catalog.py" serve-lmcache-config
    )"
  else
    export LMCACHE_LOCAL_CPU=False
    export LMCACHE_EXTRA_CONFIG='{}'
  fi

  PYTHONPATH="$PYTHONPATH_VALUE" "$VLLM_BIN" serve "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL" \
    --api-key "$API_KEY" \
    --port "$PORT" \
    --dtype auto \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
    --enforce-eager \
    --no-enable-log-requests \
    --enable-prefix-caching \
    --no-async-scheduling \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --reasoning-parser qwen3 \
    >"$leaf/vllm.log" 2>&1 &
  SERVER_PID=$!
  if ! wait_ready; then
    echo "[error] vLLM failed to become ready: $leaf/vllm.log" >&2
    tail -n 80 "$leaf/vllm.log" >&2 || true
    return 1
  fi
}

run_case() {
  local mode="$1"
  local replica="$2"
  local ordinal="$3"
  local kind="$4"
  local leaf="$5"
  local case_id="r${replica}-${mode}-${kind}-${ordinal}"
  local case_dir="$leaf/cases/$case_id"
  mkdir -p "$case_dir/workspace"
  reset_prefix_cache
  export CSKCACHE_LATENCY_CASE_ID="$case_id"
  export CSKCACHE_AGENT_TIMELINE_PATH="$case_dir/agent_timeline.jsonl"
  : > "$CSKCACHE_AGENT_TIMELINE_PATH"
  local agent_script="$MOTIVATION_DIR/interactive_agent_no_reuse.py"
  local -a cache_args=()
  if [[ "$mode" == "cskcache" ]]; then
    agent_script="$MOTIVATION_DIR/interactive_agent.py"
    cache_args=(--pool-dir "$RAW_POOL_DIR" --model-path "$MODEL_PATH")
  fi
  set +e
  PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$agent_script" \
    --skills-dir "$ROOT/skills/Auto-claude-code-research-in-sleep/skills" \
    --extra-skills-dir "$ROOT/skills" \
    --served-model "$SERVED_MODEL" \
    --base-url "http://127.0.0.1:$PORT" \
    --api-key "$API_KEY" \
    --workspace "$case_dir/workspace" \
    --skill "$SKILL" \
    --prompt-file "$PROMPT_FILE" \
    --max-iterations "$MAX_AGENT_ITERATIONS" \
    "${cache_args[@]}" \
    >"$case_dir/agent.log" 2>&1
  rc=$?
  set -e
  if ! grep -q '"post_skill":true' "$CSKCACHE_AGENT_TIMELINE_PATH"; then
    echo "[error] case did not produce request B: $case_id (agent rc=$rc)" >&2
    tail -n 60 "$case_dir/agent.log" >&2 || true
    return 1
  fi
  printf '{"case_id":"%s","mode":"%s","replica":%s,"ordinal":%s,"kind":"%s","agent_exit_code":%s}\n' \
    "$case_id" "$mode" "$replica" "$ordinal" "$kind" "$rc" \
    > "$case_dir/case.json"
  echo "[captured] $case_id"
}

for ((replica=0; replica<REPLICAS; replica++)); do
  if ((replica % 2 == 0)); then modes=(recompute cskcache); else modes=(cskcache recompute); fi
  for mode in "${modes[@]}"; do
    leaf="$RUN_DIR/replica_${replica}/$mode"
    mkdir -p "$leaf/cases"
    echo "[server] replica=$replica mode=$mode"
    start_server "$mode" "$leaf"
    for ((i=0; i<WARMUP_CASES; i++)); do
      run_case "$mode" "$replica" "$i" warmup "$leaf"
    done
    # for ((i=0; i<MEASURE_CASES; i++)); do
    #   run_case "$mode" "$replica" "$i" measure "$leaf"
    # done
    stop_server
  done
done

CSKCACHE_LATENCY_RUN_DIR="$RUN_DIR" PYTHONPATH="$PYTHONPATH_VALUE" \
  "$PYTHON_BIN" "$SCRIPT_DIR/analyze_latency.py"
echo "[completed] raw=$RUN_DIR"
