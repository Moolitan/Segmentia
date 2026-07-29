#!/usr/bin/env bash
# Capture source contextual Skill KV, prove cross-request SSD reuse, then
# capture an isolated full target-context KV ground truth. This script is
# intentionally immutable/no-resume: every run must use a fresh RUN_ID.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
VLLM_BIN="${VLLM_BIN:-vllm}"
MODEL_PATH="${VLLM_MODEL_PATH:-/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B}"
SERVED_MODEL="${VLLM_SERVED_NAME:-Qwen3}"
PORT="${VLLM_PORT:-8100}"
API_KEY="${VLLM_API_KEY:-EMPTY}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
READY_ATTEMPTS="${VLLM_READY_MAX_ATTEMPTS:-450}"
READY_INTERVAL="${VLLM_READY_INTERVAL:-2}"
NUM_LAYERS="${SEGMENTIA_MODEL_NUM_LAYERS:-40}"
KV_HEADS="${SEGMENTIA_MODEL_KV_HEADS:-8}"
HEAD_DIM="${SEGMENTIA_MODEL_HEAD_DIM:-128}"
DTYPE_BYTES="${SEGMENTIA_KV_DTYPE_BYTES:-2}"
ACTUAL_ANCHOR_TOKENS="${SEGMENTIA_ACTUAL_ANCHOR_TOKENS:-0}"
ANCHOR_LAYOUT="${SEGMENTIA_ANCHOR_LAYOUT:-centered}"
ANCHOR_CORRECTION="${SEGMENTIA_ANCHOR_CORRECTION:-none}"
PREFIX_CORRECTION="${SEGMENTIA_PREFIX_CORRECTION:-0}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_ROOT="${SEGMENTIA_CAPTURE_OUTPUT_ROOT:-/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/08_lmcache_mp/cross_request_kv_capture_runs}"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
REUSE_BASE_RUN="${SEGMENTIA_REUSE_BASE_RUN:-}"
REUSE_DIRECTION="${SEGMENTIA_REUSE_DIRECTION:-forward}"
SEPARATOR="${LMCACHE_BLEND_SPECIAL_STR:-<|fim_pad|><|repo_name|>}"
PYTHONPATH_VALUE="$ROOT/vllm:$ROOT/LMCache${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "opencode" ]]; then
  echo "[error] activate the opencode conda environment before running" >&2
  exit 2
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[error] PYTHON_BIN is unavailable in opencode: $PYTHON_BIN" >&2
  exit 2
fi
if ! command -v "$VLLM_BIN" >/dev/null 2>&1; then
  echo "[error] VLLM_BIN is unavailable in opencode: $VLLM_BIN" >&2
  exit 2
fi
if ! command -v nvcc >/dev/null 2>&1; then
  echo "[error] nvcc is unavailable in opencode; FlashInfer warmup cannot run" >&2
  exit 2
fi
if [[ -e "$RUN_DIR" ]]; then
  echo "[error] immutable run directory already exists: $RUN_DIR" >&2
  exit 2
fi
if [[ "$PREFIX_CORRECTION" == "1" && "$ACTUAL_ANCHOR_TOKENS" != "0" ]]; then
  echo "[error] prefix correction and fixed anchors are mutually exclusive" >&2
  exit 2
fi

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export VLLM_API_KEY="$API_KEY"
export VLLM_SERVED_NAME="$SERVED_MODEL"
export LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-256}"
export LMCACHE_BLEND_SPECIAL_STR="$SEPARATOR"
export LMCACHE_USE_LAYERWISE="${LMCACHE_USE_LAYERWISE:-True}"
export LMCACHE_ENABLE_SEGMENTIA="${LMCACHE_ENABLE_SEGMENTIA:-True}"
export LMCACHE_SEGMENTIA_CHECK_LAYERS="${LMCACHE_SEGMENTIA_CHECK_LAYERS:-1}"
export LMCACHE_SEGMENTIA_RECOMPUTE_RATIOS="${LMCACHE_SEGMENTIA_RECOMPUTE_RATIOS:-0.15}"
# Keep the CPU backend as LMCache's staging allocator, but disable it as a
# hot-cache tier. This makes a successful target reuse an actual SSD read.
export LMCACHE_LOCAL_CPU="${LMCACHE_LOCAL_CPU:-False}"
export LMCACHE_MAX_LOCAL_CPU_SIZE="${LMCACHE_MAX_LOCAL_CPU_SIZE:-5}"
export LMCACHE_MAX_LOCAL_DISK_SIZE="${LMCACHE_MAX_LOCAL_DISK_SIZE:-50}"
export LMCACHE_LOCAL_DISK_REHYDRATE="${LMCACHE_LOCAL_DISK_REHYDRATE:-True}"
export SEGMENTIA_PROFILE=1

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
  echo "[error] vLLM readiness timeout port=$PORT" >&2
  return 1
}

verify_local_sources() {
  PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" -c '
from pathlib import Path
import lmcache
import vllm
root = Path("'"$ROOT"'").resolve()
actual = {"vllm": Path(vllm.__file__).resolve(), "lmcache": Path(lmcache.__file__).resolve()}
expected = {"vllm": root / "vllm", "lmcache": root / "LMCache"}
for name, path in actual.items():
    if not path.is_relative_to(expected[name]):
        raise RuntimeError(f"{name} resolved to {path}, expected under {expected[name]}")
print(f"[sources] vllm={actual['"'"'vllm'"'"']} lmcache={actual['"'"'lmcache'"'"']}")
'
}

start_server() {
  local cache_dir="$1"
  local log_path="$2"
  local anchor_capture_dir="${3:-}"
  mkdir -p "$(dirname "$log_path")"
  export LMCACHE_LOCAL_DISK="file://${cache_dir}/"
  if [[ "$PREFIX_CORRECTION" == "1" ]] && [[ -n "$anchor_capture_dir" ]]; then
    export LMCACHE_EXTRA_CONFIG
    LMCACHE_EXTRA_CONFIG="$(jq -cn \
      --arg capture_dir "$anchor_capture_dir" \
      '{segmentia_prefix_correction:true,
        segmentia_prefix_capture_dir:$capture_dir}')"
  elif ((ACTUAL_ANCHOR_TOKENS > 0)) && [[ -n "$anchor_capture_dir" ]]; then
    export LMCACHE_EXTRA_CONFIG
    LMCACHE_EXTRA_CONFIG="$(jq -cn \
      --argjson tokens "$ACTUAL_ANCHOR_TOKENS" \
      --arg capture_dir "$anchor_capture_dir" \
      --arg anchor_layout "$ANCHOR_LAYOUT" \
      --arg anchor_correction "$ANCHOR_CORRECTION" \
      '{segmentia_fixed_anchor_tokens:$tokens,
        segmentia_anchor_capture_dir:$capture_dir,
        segmentia_anchor_layout:$anchor_layout,
        segmentia_anchor_correction:$anchor_correction}')"
  else
    unset LMCACHE_EXTRA_CONFIG || true
  fi
  PYTHONPATH="$PYTHONPATH_VALUE" "$VLLM_BIN" serve "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL" \
    --api-key "$API_KEY" \
    --port "$PORT" \
    --dtype auto \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --reasoning-parser qwen3 \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
    --enforce-eager \
    --no-enable-log-requests \
    --enable-prefix-caching \
    --no-async-scheduling \
    >"$log_path" 2>&1 &
  SERVER_PID=$!
  wait_vllm_ready
}

send_phase() {
  local case_id="$1"
  local phase="$2"
  local endpoint_name="${3:-}"
  local phase_dir="$RUN_DIR/$case_id/$phase"
  mkdir -p "$phase_dir"
  local replay_args=(
    --prepared-cases "$RUN_DIR/prepared_cases.json"
    --case-id "$case_id"
    --phase "$phase"
    --output "$phase_dir/request.json"
    --tokenizer-path "$MODEL_PATH"
    --base-url "http://127.0.0.1:$PORT"
    --model "$SERVED_MODEL"
    --api-key "$API_KEY"
    --separator "$SEPARATOR"
  )
  if [[ "$PREFIX_CORRECTION" == "1" ]]; then
    replay_args+=(--correction-mode prefix_k_headwise)
  fi
  if [[ -n "$endpoint_name" ]]; then
    replay_args+=(--endpoint-name "$endpoint_name")
  fi
  PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$SCRIPT_DIR/replay_request.py" \
    "${replay_args[@]}"
}

wait_for_ssd() {
  local request_path="$1"
  local cache_dir="$2"
  PYTHONPATH="$SCRIPT_DIR" "$PYTHON_BIN" "$SCRIPT_DIR/wait_for_cache.py" \
    --request "$request_path" \
    --cache-dir "$cache_dir" \
    --layers "$NUM_LAYERS" \
    --kv-heads "$KV_HEADS" \
    --head-dim "$HEAD_DIM" \
    --dtype-bytes "$DTYPE_BYTES"
}

mkdir -p "$RUN_DIR"
prepare_args=(--output "$RUN_DIR/prepared_cases.json")
if [[ -n "${CASE_IDS:-}" ]]; then
  IFS=',' read -ra selected_case_ids <<< "$CASE_IDS"
  for case_id in "${selected_case_ids[@]}"; do
    case_id="${case_id// /}"
    [[ -z "$case_id" ]] || prepare_args+=(--case-id "$case_id")
  done
fi
verify_local_sources
PYTHONPATH="$SCRIPT_DIR" "$PYTHON_BIN" "$SCRIPT_DIR/prepare_cases.py" "${prepare_args[@]}"

mapfile -t case_ids < <(jq -r '.cases[].case_id' "$RUN_DIR/prepared_cases.json")
reuse_capture_name="actual_anchor"
if [[ "$PREFIX_CORRECTION" == "1" ]]; then
  reuse_capture_name="prefix_correction"
fi
for case_id in "${case_ids[@]}"; do
  case_dir="$RUN_DIR/$case_id"
  if [[ -n "$REUSE_BASE_RUN" ]]; then
    base_case_dir="$REUSE_BASE_RUN/$case_id"
    base_manifest="$base_case_dir/manifest.json"
    if [[ "$REUSE_DIRECTION" == "forward" ]]; then
      base_cache_ssd="$base_case_dir/shared_ssd"
      replay_endpoint="target"
    elif [[ "$REUSE_DIRECTION" == "reverse" ]]; then
      base_cache_ssd="$base_case_dir/target_full_ssd"
      replay_endpoint="source"
    else
      echo "[error] invalid SEGMENTIA_REUSE_DIRECTION: $REUSE_DIRECTION" >&2
      exit 2
    fi
    if [[ ! -f "$base_manifest" ]] || \
       [[ "$(jq -r '.status // empty' "$base_manifest")" != "completed" ]]; then
      echo "[error] base case has no completed manifest: $base_case_dir" >&2
      exit 2
    fi
    if [[ ! -d "$base_cache_ssd" ]]; then
      echo "[error] base case has no cache SSD: $base_cache_ssd" >&2
      exit 2
    fi
    echo "[reuse-only] case=$case_id direction=$REUSE_DIRECTION restart -> rehydrate -> $reuse_capture_name"
    start_server \
      "$base_cache_ssd" \
      "$case_dir/target_reuse/vllm.log" \
      "$case_dir/target_reuse/$reuse_capture_name"
    send_phase "$case_id" target_reuse "$replay_endpoint"
    cleanup_server
    continue
  fi

  shared_ssd="$case_dir/shared_ssd"
  target_full_ssd="$case_dir/target_full_ssd"
  mkdir -p "$shared_ssd" "$target_full_ssd"

  echo "[phase 1/3] case=$case_id source cold miss -> shared SSD"
  start_server "$shared_ssd" "$case_dir/source/vllm.log" ""
  send_phase "$case_id" source
  wait_for_ssd "$case_dir/source/request.json" "$shared_ssd"
  cleanup_server

  echo "[phase 2/3] case=$case_id restart -> rehydrate -> target SSD reuse"
  start_server \
    "$shared_ssd" \
    "$case_dir/target_reuse/vllm.log" \
    "$case_dir/target_reuse/$reuse_capture_name"
  send_phase "$case_id" target_reuse
  cleanup_server

  echo "[phase 3/3] case=$case_id target isolated full recompute"
  start_server "$target_full_ssd" "$case_dir/target_full/vllm.log" ""
  send_phase "$case_id" target_full
  wait_for_ssd "$case_dir/target_full/request.json" "$target_full_ssd"
  cleanup_server

  PYTHONPATH="$SCRIPT_DIR" "$PYTHON_BIN" "$SCRIPT_DIR/validate_capture.py" \
    --case-dir "$case_dir" \
    --layers "$NUM_LAYERS" \
    --kv-heads "$KV_HEADS" \
    --head-dim "$HEAD_DIM" \
    --dtype-bytes "$DTYPE_BYTES"
done

echo "[completed] run_dir=$RUN_DIR"
