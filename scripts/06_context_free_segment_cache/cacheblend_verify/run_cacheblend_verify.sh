#!/usr/bin/env bash
# Run two deterministic chunk-pair recompute vs full-KV-reuse comparisons.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$ROOT"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true

export VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B}"
export VLLM_SERVED_NAME="${VLLM_SERVED_NAME:-Qwen3}"
export VLLM_PORT="${VLLM_PORT:-8000}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export PREFIX_CACHING=0
# 采样参数：Qwen3 thinking 默认（不再贪心）。可用环境变量覆盖。
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
MIN_P="${MIN_P:-0.0}"

TEMP_TAG="${TEMP_TAG:-temp06}"

RUN_DIR="${RUN_DIR:-/mnt/Large_Language_Model_Lab_1/wsh/CacheBlend/output/two_chunk_full_reuse_verify/pilot_${TEMP_TAG}}"
KV_DIR="$RUN_DIR/kv"
RAW_DIR="$RUN_DIR/raw"
SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-$ROOT/results/problem_exploration/cacheblend_full_kv_reuse_verify/data/comparison_${TEMP_TAG}.json}"
MAX_TOKENS="${MAX_TOKENS:-512}"
SEED="${SEED:-1111}"
BASE_URL="http://127.0.0.1:${VLLM_PORT}"

mkdir -p "$KV_DIR" "$RAW_DIR" "$(dirname "$SUMMARY_OUTPUT")"

cleanup() {
  bash "$ROOT/scripts/vllm_stop.sh" || true
}
trap cleanup EXIT

start_vllm() {
  local stage="$1"
  echo ""
  echo "[vLLM] restart boundary=(experiment=cacheblend_verify, stage=$stage)"
  bash "$ROOT/scripts/vllm_stop.sh" || true
  bash "$ROOT/scripts/vllm_start.sh"

  local ready=0
  local code=""
  for _poll_i in $(seq 1 600); do
    code="$(curl -sS -o /dev/null -w '%{http_code}' \
      --connect-timeout 3 --max-time 10 \
      -H "Authorization: Bearer ${VLLM_API_KEY}" \
      "$BASE_URL/v1/models" 2>/dev/null || true)"
    if [[ "$code" == "200" ]]; then
      ready=1
      echo "[vLLM] ready stage=$stage"
      break
    fi
    sleep 2
  done
  if [[ "$ready" != "1" ]]; then
    echo "[error] vLLM start timeout at stage=$stage" >&2
    exit 1
  fi
}

# Stage 1: each of the three chunks is a separate source request. Prefix
# caching is disabled, so no chunk can inherit state from another chunk.
unset VLLM_CONTEXT_SEGMENT_KV_DIR || true
export VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR="$KV_DIR"
start_vllm "collect_independent_chunks"
python "$SCRIPT_DIR/run_cacheblend_verify.py" collect \
  --base-url "$BASE_URL" \
  --model "$VLLM_SERVED_NAME" \
  --api-key "$VLLM_API_KEY" \
  --kv-dir "$KV_DIR"

CASES=(alias_only explicit_full_name)
MODES=(recompute full_kv_reuse)

# Each (case, mode) gets a fresh server. This prevents prefix state from one
# pair or arm from covering a later target span.
for case_name in "${CASES[@]}"; do
  mkdir -p "$RAW_DIR/$case_name"
  for mode in "${MODES[@]}"; do
    unset VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR || true
    if [[ "$mode" == "full_kv_reuse" ]]; then
      export VLLM_CONTEXT_SEGMENT_KV_DIR="$KV_DIR"
    else
      unset VLLM_CONTEXT_SEGMENT_KV_DIR || true
    fi
    start_vllm "case=$case_name,mode=$mode"
    python "$SCRIPT_DIR/run_cacheblend_verify.py" decode \
      --case "$case_name" \
      --mode "$mode" \
      --base-url "$BASE_URL" \
      --model "$VLLM_SERVED_NAME" \
      --api-key "$VLLM_API_KEY" \
      --kv-dir "$KV_DIR" \
      --output "$RAW_DIR/$case_name/$mode.json" \
      --max-tokens "$MAX_TOKENS" \
      --temperature "$TEMPERATURE" \
      --top-p "$TOP_P" \
      --top-k "$TOP_K" \
      --min-p "$MIN_P" \
      --seed "$SEED"
  done
done

python "$SCRIPT_DIR/run_cacheblend_verify.py" summarize \
  --run-dir "$RUN_DIR" \
  --summary-output "$SUMMARY_OUTPUT"

echo ""
echo "[done] raw run artifacts: $RUN_DIR"
echo "[done] lightweight comparison: $SUMMARY_OUTPUT"
