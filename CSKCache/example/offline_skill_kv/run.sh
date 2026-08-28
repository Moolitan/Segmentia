#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
LMCACHE_ROOT="$ROOT/LMCache"
CSKCACHE_ROOT="$ROOT/CSKCache"
VLLM_ROOT="${VLLM_ROOT:-$ROOT/vllm}"
PYTHON_BIN="${PYTHON_BIN:-python}"
VLLM_BIN="${VLLM_BIN:-vllm}"

if (($#)); then
  echo "[error] this launcher accepts no arguments; edit config.py instead" >&2
  exit 2
fi
eval "$("$PYTHON_BIN" "$SCRIPT_DIR/config_loader.py")"

MODEL_PATH="$VLLM_MODEL_PATH"
SERVED_MODEL="$VLLM_SERVED_NAME"
OUTPUT_ROOT="$SKILL_SAVE_POOL_ROOT"
OUTPUT_DIR="$OUTPUT_ROOT/$SKILL_POOL_MODEL_DIR_NAME"
RAW_ROOT="$OUTPUT_DIR/raw"
LOCAL_ROOT="$OUTPUT_DIR/layer_files"
STORAGE_BACKEND="$OFFLINE_STORAGE_BACKEND"
PORT="$VLLM_PORT"
API_KEY="$VLLM_API_KEY"
GPU_UTIL="$VLLM_GPU_UTIL"
MAX_MODEL_LEN="$VLLM_MAX_MODEL_LEN"
READY_ATTEMPTS="$VLLM_READY_MAX_ATTEMPTS"
READY_INTERVAL="$VLLM_READY_INTERVAL"
SHUTDOWN_TIMEOUT="$VLLM_SHUTDOWN_TIMEOUT"
EXPECTED_LAYERS="$CSKCACHE_MODEL_NUM_LAYERS"
PYTHONPATH_VALUE="$VLLM_ROOT:$LMCACHE_ROOT:$CSKCACHE_ROOT${PYTHONPATH:+:$PYTHONPATH}"

COLLECTION="$OFFLINE_COLLECTION"
OVERWRITE="$OFFLINE_OVERWRITE"
DRY_RUN="$OFFLINE_DRY_RUN"
DEDUPLICATE_CONTENT="$OFFLINE_DEDUPLICATE_CONTENT"
SKILLS=()
EXCLUDED_SKILLS=()
while IFS= read -r skill; do
  [[ -z "$skill" ]] || SKILLS+=("$skill")
done <<< "$OFFLINE_SKILLS"
while IFS= read -r skill; do
  [[ -z "$skill" ]] || EXCLUDED_SKILLS+=("$skill")
done <<< "$OFFLINE_EXCLUDED_SKILLS"

if [[ "$STORAGE_BACKEND" == "raw_block" ]]; then
  BACKEND_ROOT="$RAW_ROOT"
  CATALOG_MANAGER="$SCRIPT_DIR/manage_raw_catalog.py"
elif [[ "$STORAGE_BACKEND" == "local_disk" ]]; then
  BACKEND_ROOT="$LOCAL_ROOT"
  CATALOG_MANAGER="$SCRIPT_DIR/manage_local_catalog.py"
else
  echo "[error] storage backend must be raw_block or local_disk" >&2
  exit 2
fi
PENDING_DIR="$BACKEND_ROOT/.pending"
CATALOG="$BACKEND_ROOT/catalog.json"

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
if [[ "$STORAGE_BACKEND" == "raw_block" ]]; then
  export LMCACHE_STORAGE_PLUGINS="raw_block"
  export LMCACHE_STORE_LOCATION="raw_block"
  unset LMCACHE_LOCAL_DISK LMCACHE_FORCE_SKIP_SAVE || true
else
  export LMCACHE_STORAGE_PLUGINS=""
  export LMCACHE_STORE_LOCATION="LocalDiskBackend"
  export LMCACHE_LOCAL_DISK="$LOCAL_ROOT"
  export LMCACHE_MAX_LOCAL_DISK_SIZE="${LMCACHE_MAX_LOCAL_DISK_SIZE:-1000}"
  unset LMCACHE_FORCE_SKIP_SAVE || true
fi

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

list_args=(--list --skills-dir "$OFFLINE_SKILLS_DIR")
if [[ "$DEDUPLICATE_CONTENT" == "1" ]]; then
  list_args+=(--deduplicate-content)
fi
[[ -z "$COLLECTION" ]] || list_args+=(--collection "$COLLECTION")
for skill in "${SKILLS[@]}"; do
  list_args+=(--skill "$skill")
done
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
    "$CATALOG_MANAGER" initialize; then
    echo "[error] failed to initialize the $STORAGE_BACKEND Skill pool" >&2
    exit 1
  fi
  # A prior process may have completed raw writes but crashed before Catalog
  # publication.  Recover that transaction before accepting new requests;
  # successful publication removes its pending records.
  recovery_args=(finalize)
  if [[ "$STORAGE_BACKEND" == "raw_block" && "$OVERWRITE" == "1" ]]; then
    # --overwrite authorizes rebuilding an unpublished transaction whose raw
    # bytes exist but whose LMCache key index was never checkpointed.  The
    # helper preserves those pending records under raw/.failed/; it does not
    # delete an already published Catalog object or hide any other error.
    recovery_args+=(--quarantine-unrecoverable)
  fi
  if ! PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" \
    "$CATALOG_MANAGER" "${recovery_args[@]}"; then
    echo "[error] failed to recover pending $STORAGE_BACKEND objects" >&2
    exit 1
  fi
  LMCACHE_EXTRA_CONFIG="$(PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" \
    "$CATALOG_MANAGER" lmcache-config)" || exit 1
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
    --skills-dir "$OFFLINE_SKILLS_DIR"
    --cache-id "$cache_id"
    --skill-path "$skill_path"
    --pending-dir "$PENDING_DIR"
    --catalog "$CATALOG"
    --model-path "$MODEL_PATH"
    --served-model "$SERVED_MODEL"
    --base-url "http://127.0.0.1:$PORT"
    --api-key "$API_KEY"
    --expected-layers "$EXPECTED_LAYERS"
    --storage-backend "$STORAGE_BACKEND"
  )
  if [[ "$STORAGE_BACKEND" == "local_disk" ]]; then
    driver_args+=(--local-disk-root "$LOCAL_ROOT")
  fi
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
    "$CATALOG_MANAGER" finalize; then
    failures=$((failures + 1))
    echo "[error] $STORAGE_BACKEND objects were not published to the Catalog" >&2
  fi
fi
echo "[done] backend=$STORAGE_BACKEND completed=$completed skipped=$skipped failures=$failures output=$BACKEND_ROOT"
((failures == 0))
