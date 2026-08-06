#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
SCRIPT_DIR="$ROOT/scripts/08_lmcache_mp/paper_motivation/3.1"
LMCACHE_ROOT="$ROOT/LMCache"
VLLM_ROOT="${VLLM_ROOT:-$ROOT/vllm}"
PYTHON_BIN="${PYTHON_BIN:-python}"
VLLM_BIN="${VLLM_BIN:-vllm}"
MODEL_PATH="${VLLM_MODEL_PATH:-/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B}"
SERVED_MODEL="${VLLM_SERVED_NAME:-Qwen3}"
OUTPUT_ROOT="${SKILL_SAVE_POOL_ROOT:-/mnt/Large_Language_Model_Lab_1/wsh/skill_save_pool}"
OUTPUT_DIR="$OUTPUT_ROOT/${SKILL_POOL_MODEL_DIR_NAME:-Qwen3-14B}"
PORT="${VLLM_PORT:-8013}"
API_KEY="${VLLM_API_KEY:-EMPTY}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.9}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
READY_ATTEMPTS="${VLLM_READY_MAX_ATTEMPTS:-450}"
READY_INTERVAL="${VLLM_READY_INTERVAL:-2}"
EXPECTED_LAYERS="${SEGMENTIA_MODEL_NUM_LAYERS:-40}"
PYTHONPATH_VALUE="$VLLM_ROOT:$LMCACHE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

COLLECTION=""
SKILL=""
OVERWRITE=0
DRY_RUN=0
while (($#)); do
  case "$1" in
    --collection)
      COLLECTION="${2:?--collection requires a value}"
      shift 2
      ;;
    --skill)
      SKILL="${2:?--skill requires a value}"
      shift 2
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      echo "usage: $0 [--collection NAME] [--skill CACHE_ID_OR_NAME] [--overwrite] [--dry-run]"
      exit 0
      ;;
    *)
      echo "[error] unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$DRY_RUN" == "0" && "${CONDA_DEFAULT_ENV:-}" != "opencode" ]]; then
  echo "[error] activate conda environment opencode first" >&2
  exit 2
fi
if [[ ! -d "$LMCACHE_ROOT/lmcache" ]]; then
  echo "[error] local LMCache source is missing: $LMCACHE_ROOT" >&2
  exit 2
fi
if [[ ! -d "$VLLM_ROOT/vllm" ]]; then
  echo "[error] local vLLM source is missing: $VLLM_ROOT" >&2
  exit 2
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-256}"
export LMCACHE_USE_LAYERWISE="${LMCACHE_USE_LAYERWISE:-True}"
export LMCACHE_ENABLE_SEGMENTIA="${LMCACHE_ENABLE_SEGMENTIA:-True}"
export LMCACHE_SEGMENTIA_CHECK_LAYERS="${LMCACHE_SEGMENTIA_CHECK_LAYERS:-1}"
export LMCACHE_LOCAL_CPU="${LMCACHE_LOCAL_CPU:-False}"
export LMCACHE_MAX_LOCAL_CPU_SIZE="${LMCACHE_MAX_LOCAL_CPU_SIZE:-5}"
export LMCACHE_MAX_LOCAL_DISK_SIZE="${LMCACHE_MAX_LOCAL_DISK_SIZE:-1000}"
export LMCACHE_LOCAL_DISK_REHYDRATE="${LMCACHE_LOCAL_DISK_REHYDRATE:-True}"

SERVER_PID=""
cleanup_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap cleanup_server EXIT INT TERM

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
      return 0
    fi
    attempt=$((attempt + 1))
    sleep "$READY_INTERVAL"
  done
  echo "[error] vLLM readiness timeout on port $PORT" >&2
  return 1
}

start_server() {
  local cache_dir="$1"
  local server_log="$2"
  export LMCACHE_LOCAL_DISK="file://${cache_dir}/"
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
    --no-enable-prefix-caching \
    --no-async-scheduling \
    >"$server_log" 2>&1 &
  SERVER_PID=$!
  wait_vllm_ready
}

list_args=(--list --skills-dir "$ROOT/skills")
[[ -z "$COLLECTION" ]] || list_args+=(--collection "$COLLECTION")
[[ -z "$SKILL" ]] || list_args+=(--skill "$SKILL")
mapfile -t skill_rows < <(
  PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" \
    "$SCRIPT_DIR/prefill_skill_pool.py" "${list_args[@]}"
)
if ((${#skill_rows[@]} == 0)); then
  echo "[error] no matching Skills" >&2
  exit 2
fi

if [[ "$DRY_RUN" == "0" ]]; then
  mkdir -p "$OUTPUT_DIR" || exit 2
fi
failures=0
completed=0
skipped=0
STAGING_DIR="$OUTPUT_DIR/.staging"
POOL_INDEX="$OUTPUT_DIR/pool_index.json"

if [[ "$DRY_RUN" == "0" ]]; then
  rm -rf -- "$STAGING_DIR"
  mkdir -p "$STAGING_DIR"
  SERVER_LOG="$OUTPUT_DIR/vllm.log"
  echo "[server] starting one shared vLLM; log: $SERVER_LOG"
  if ! start_server "$STAGING_DIR" "$SERVER_LOG"; then
    cleanup_server
    echo "[error] vLLM startup failed; log: $SERVER_LOG" >&2
    tail -n 40 "$SERVER_LOG" >&2 || true
    exit 1
  fi
fi

for row in "${skill_rows[@]}"; do
  IFS=$'\t' read -r cache_id skill_path <<< "$row"
  [[ -n "$cache_id" && -n "$skill_path" ]] || continue
  skill_dir="$OUTPUT_DIR/$cache_id"
  cache_dir="$skill_dir/kv"
  manifest="$skill_dir/manifest.json"
  completed_marker="$skill_dir/COMPLETED"

  if [[ "$DRY_RUN" == "0" && "$OVERWRITE" == "0" && -f "$completed_marker" ]]; then
    echo "[skipped] $cache_id (COMPLETED exists)"
    skipped=$((skipped + 1))
    continue
  fi

  driver_args=(
    --skills-dir "$ROOT/skills"
    --cache-id "$cache_id"
    --skill-path "$skill_path"
    --cache-dir "$cache_dir"
    --staging-dir "$STAGING_DIR"
    --pool-index "$POOL_INDEX"
    --manifest "$manifest"
    --model-path "$MODEL_PATH"
    --served-model "$SERVED_MODEL"
    --base-url "http://127.0.0.1:$PORT"
    --api-key "$API_KEY"
    --expected-layers "$EXPECTED_LAYERS"
  )
  if [[ "$DRY_RUN" == "1" ]]; then
    PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" \
      "$SCRIPT_DIR/prefill_skill_pool.py" "${driver_args[@]}" --dry-run || \
      failures=$((failures + 1))
    continue
  fi

  rm -rf -- "$cache_dir"
  rm -f -- "$manifest" "$completed_marker"
  echo "[prefill] $cache_id"
  if PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" \
    "$SCRIPT_DIR/prefill_skill_pool.py" "${driver_args[@]}"; then
    completed=$((completed + 1))
  else
    failures=$((failures + 1))
    echo "[error] aborting after Skill failure: $cache_id" >&2
    break
  fi
done

cleanup_server
if ((failures == 0)) && [[ "$DRY_RUN" == "0" ]]; then
  rm -rf -- "$STAGING_DIR"
fi
echo "[done] completed=$completed skipped=$skipped failures=$failures output=$OUTPUT_DIR"
((failures == 0))
