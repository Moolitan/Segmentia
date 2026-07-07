#!/usr/bin/env bash
# Run the three occurrence-1 cross-task contextual KV cases documented in the
# Python entrypoint. The vLLM lifecycle boundary is (mode, target_task).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$ROOT"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true

export VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B}"
export VLLM_SERVED_NAME="${VLLM_SERVED_NAME:-Qwen3}"
export VLLM_PORT="${VLLM_PORT:-8000}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"

DEFAULT_CONTEXTUAL_KV_DIR="$(
  PYTHONPATH="$ROOT/scripts/06_context_free_segment_cache/module" python -c \
    'from config import DEFAULT_CONTEXTUAL_KV_DIR; print(DEFAULT_CONTEXTUAL_KV_DIR)'
)"
KV_DIR="${KV_DIR:-$DEFAULT_CONTEXTUAL_KV_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results/problem_exploration/cross_task_contextual_occ1_reuse}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1}"
SEED="${SEED:-1111}"
RESUME="${RESUME:-1}"

MODES=(recompute cross_contextual_rope)
TARGET_TASKS=(mcp_server_and_spec slack_launch_pack launch_poster_page_pack)

cleanup() {
  bash "$ROOT/scripts/vllm_stop.sh" || true
}
trap cleanup EXIT

start_vllm() {
  local mode="$1"
  local target_task="$2"

  echo ""
  echo "[vLLM] restart boundary=(mode=$mode, target_task=$target_task)"
  unset VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR || true
  if [[ "$mode" == "recompute" ]]; then
    unset VLLM_CONTEXT_SEGMENT_KV_DIR || true
  else
    export VLLM_CONTEXT_SEGMENT_KV_DIR="$KV_DIR"
  fi

  bash "$ROOT/scripts/vllm_stop.sh" || true
  bash "$ROOT/scripts/vllm_start.sh"

  local ready=0
  local code=""
  for _poll_i in $(seq 1 600); do
    code="$(curl -sS -o /dev/null -w '%{http_code}' \
      --connect-timeout 3 --max-time 10 \
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
    echo "[error] vLLM start timeout for mode=$mode task=$target_task" >&2
    exit 1
  fi
}

for mode in "${MODES[@]}"; do
  sequence_dir="$OUTPUT_DIR/$mode"
  manifest="$OUTPUT_DIR/$mode/manifest.jsonl"
  mkdir -p "$sequence_dir"

  for target_task in "${TARGET_TASKS[@]}"; do
    start_vllm "$mode" "$target_task"
    args=(
      --target-task "$target_task"
      --mode "$mode"
      --sequence-dir "$sequence_dir"
      --manifest "$manifest"
      --kv-dir "$KV_DIR"
      --vllm-port "$VLLM_PORT"
      --model "$VLLM_SERVED_NAME"
      --api-key "$VLLM_API_KEY"
      --max-tokens "$MAX_TOKENS"
      --temperature "$TEMPERATURE"
      --top-p "$TOP_P"
      --seed "$SEED"
    )
    if [[ "$RESUME" == "1" ]]; then
      args+=(--resume)
    fi
    python "$SCRIPT_DIR/run_cross_task_contextual_occ1_reuse.py" "${args[@]}"
  done
done

echo ""
echo "[done] cross-task contextual occurrence-1 results: $OUTPUT_DIR"
