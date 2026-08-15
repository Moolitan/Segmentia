#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
LMCACHE_ROOT="$ROOT/LMCache"
CSKCACHE_ROOT="$ROOT/CSKCache"
VLLM_ROOT="${VLLM_ROOT:-$ROOT/vllm}"
PYTHON_BIN="${PYTHON_BIN:-python}"
VLLM_BIN="${VLLM_BIN:-vllm}"
MODEL_PATH="${VLLM_MODEL_PATH:-/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B}"
SERVED_MODEL="${VLLM_SERVED_NAME:-Qwen3}"
OUTPUT_ROOT="${SKILL_SAVE_POOL_ROOT:-/mnt/990_pro/skill_save_pool}"
OUTPUT_DIR="$OUTPUT_ROOT/${SKILL_POOL_MODEL_DIR_NAME:-Qwen3-14B}"
RAW_ROOT="$OUTPUT_DIR/raw"
PENDING_DIR="$RAW_ROOT/.pending"
CATALOG="$RAW_ROOT/catalog.json"
PORT="${VLLM_PORT:-8013}"
API_KEY="${VLLM_API_KEY:-EMPTY}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.9}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
READY_ATTEMPTS="${VLLM_READY_MAX_ATTEMPTS:-450}"
READY_INTERVAL="${VLLM_READY_INTERVAL:-2}"
SHUTDOWN_TIMEOUT="${VLLM_SHUTDOWN_TIMEOUT:-30}"
EXPECTED_LAYERS="${CSKCACHE_MODEL_NUM_LAYERS:-40}"
PYTHONPATH_VALUE="$VLLM_ROOT:$LMCACHE_ROOT:$CSKCACHE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

COLLECTION=""
SKILL=""
EXCLUDED_SKILLS=()
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
    --exclude-skill)
      EXCLUDED_SKILLS+=("${2:?--exclude-skill requires a value}")
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
      echo "usage: $0 [--collection NAME] [--skill CACHE_ID_OR_NAME] [--exclude-skill CACHE_ID]... [--overwrite] [--dry-run]"
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
if [[ ! -d "$CSKCACHE_ROOT/cskcache" ]]; then
  echo "[error] local CSKCache source is missing: $CSKCACHE_ROOT" >&2
  exit 2
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-256}"
export LMCACHE_USE_LAYERWISE="${LMCACHE_USE_LAYERWISE:-True}"
export LMCACHE_LOCAL_CPU="${LMCACHE_LOCAL_CPU:-True}"
export LMCACHE_MAX_LOCAL_CPU_SIZE="${LMCACHE_MAX_LOCAL_CPU_SIZE:-8}"
export LMCACHE_STORAGE_PLUGINS="raw_block"
export LMCACHE_STORE_LOCATION="raw_block"
unset LMCACHE_LOCAL_DISK LMCACHE_FORCE_SKIP_SAVE || true

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
  local server_log="$1"
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
    --shutdown-timeout "$SHUTDOWN_TIMEOUT" \
    >"$server_log" 2>&1 &
  SERVER_PID=$!
  wait_vllm_ready
}

list_args=(--list --skills-dir "$ROOT/skills")
[[ -z "$COLLECTION" ]] || list_args+=(--collection "$COLLECTION")
[[ -z "$SKILL" ]] || list_args+=(--skill "$SKILL")
for excluded_skill in "${EXCLUDED_SKILLS[@]}"; do
  list_args+=(--exclude-skill "$excluded_skill")
done
mapfile -t skill_rows < <(
  PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" \
    "$SCRIPT_DIR/prefill_skill_to_raw.py" "${list_args[@]}"
)
if ((${#skill_rows[@]} == 0)); then
  echo "[error] no matching Skills" >&2
  exit 2
fi

if [[ "$DRY_RUN" == "0" ]]; then
  if ! PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" -c \
    'import cskcache; import lmcache.integration.vllm.vllm_v1_adapter; import vllm'; then
    echo "[error] failed to import local vLLM, LMCache, or CSKCache source" >&2
    exit 2
  fi
  export SKILL_SAVE_POOL_ROOT="$OUTPUT_ROOT"
  export SKILL_POOL_MODEL_DIR_NAME="${SKILL_POOL_MODEL_DIR_NAME:-Qwen3-14B}"
  if ! PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" \
    "$SCRIPT_DIR/manage_raw_catalog.py" initialize; then
    echo "[error] failed to initialize the direct raw Skill pool" >&2
    exit 1
  fi
  # A prior process may have completed raw writes but crashed before Catalog
  # publication.  Recover that transaction before accepting new requests;
  # successful publication removes its pending records.
  recovery_args=(finalize)
  if [[ "$OVERWRITE" == "1" ]]; then
    # --overwrite authorizes rebuilding an unpublished transaction whose raw
    # bytes exist but whose LMCache key index was never checkpointed.  The
    # helper preserves those pending records under raw/.failed/; it does not
    # delete an already published Catalog object or hide any other error.
    recovery_args+=(--quarantine-unrecoverable)
  fi
  if ! PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" \
    "$SCRIPT_DIR/manage_raw_catalog.py" "${recovery_args[@]}"; then
    echo "[error] failed to recover pending direct raw objects" >&2
    exit 1
  fi
  LMCACHE_EXTRA_CONFIG="$(PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" \
    "$SCRIPT_DIR/manage_raw_catalog.py" lmcache-config)" || exit 1
  export LMCACHE_EXTRA_CONFIG
fi
failures=0
completed=0
skipped=0
if [[ "$DRY_RUN" == "0" ]]; then
  SERVER_LOG="$OUTPUT_DIR/vllm.log"
  echo "[server] starting one shared vLLM; log: $SERVER_LOG"
  if ! start_server "$SERVER_LOG"; then
    cleanup_server
    echo "[error] vLLM startup failed; log: $SERVER_LOG" >&2
    tail -n 40 "$SERVER_LOG" >&2 || true
    exit 1
  fi
fi

for row in "${skill_rows[@]}"; do
  IFS=$'\t' read -r cache_id skill_path <<< "$row"
  [[ -n "$cache_id" && -n "$skill_path" ]] || continue

  driver_args=(
    --skills-dir "$ROOT/skills"
    --cache-id "$cache_id"
    --skill-path "$skill_path"
    --pending-dir "$PENDING_DIR"
    --catalog "$CATALOG"
    --model-path "$MODEL_PATH"
    --served-model "$SERVED_MODEL"
    --base-url "http://127.0.0.1:$PORT"
    --api-key "$API_KEY"
    --expected-layers "$EXPECTED_LAYERS"
  )
  if [[ "$DRY_RUN" == "1" ]]; then
    PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" \
      "$SCRIPT_DIR/prefill_skill_to_raw.py" "${driver_args[@]}" --dry-run || \
      failures=$((failures + 1))
    continue
  fi

  if [[ "$OVERWRITE" == "1" ]]; then
    driver_args+=(--force-recompute)
  fi

  echo "[prefill] $cache_id"
  if PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" \
    "$SCRIPT_DIR/prefill_skill_to_raw.py" "${driver_args[@]}"; then
    completed=$((completed + 1))
  else
    failures=$((failures + 1))
    echo "[error] aborting after Skill failure: $cache_id" >&2
    break
  fi
done

cleanup_server
if [[ "$DRY_RUN" == "0" ]]; then
  if ! PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" \
    "$SCRIPT_DIR/manage_raw_catalog.py" finalize; then
    failures=$((failures + 1))
    echo "[error] direct raw objects were not published to the Catalog" >&2
  fi
fi
echo "[done] completed=$completed skipped=$skipped failures=$failures output=$OUTPUT_DIR"
((failures == 0))
