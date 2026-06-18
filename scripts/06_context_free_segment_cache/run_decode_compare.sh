#!/usr/bin/env bash
# Run decode comparison with per-task vLLM restarts to prevent prefix cache
# from bleeding across modes.
#
# For each task the order is:
#   1. restart vLLM (no KV dir)  → run recompute for this task
#   2. restart vLLM (with KV dir) → run direct   for this task
#   3. restart vLLM (with KV dir) → run rope     for this task
#
# Usage:
#   bash run_decode_compare.sh
#   TASKS=internal_comms_incident_update,slack_launch_pack bash run_decode_compare.sh
#   OCCURRENCES=2,3 bash run_decode_compare.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true

export VLLM_PORT="${VLLM_PORT:-8000}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export VLLM_SERVED_NAME="${VLLM_SERVED_NAME:-Qwen3}"

# All tasks in order; override with TASKS=a,b,c
ALL_TASKS="internal_comms_incident_update doc_coauthoring_design_doc mcp_server_and_spec web_artifact_with_theme launch_poster_page_pack slack_launch_pack"
TASKS="${TASKS:-all}"
# Override which modes to run per task; default runs all three.
# Example: MODES=direct or MODES=direct,rope
ALL_MODES="recompute direct rope"
MODES="${MODES:-all}"
OCCURRENCES="${OCCURRENCES:-2,3}"
RES="$ROOT/results/problem_exploration"
SEGMENTIA_OUTPUT_DIR="${SEGMENTIA_OUTPUT_DIR:-/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/06_context_free_segment_cache}"
OUTPUT="${OUTPUT:-$RES/headline_semantic_action_gap/data/decode_outputs.jsonl}"
KV_DIR="${KV_DIR:-$SEGMENTIA_OUTPUT_DIR/offline_skill_kv}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
# Repeat decoding to separate stable trajectory divergence from sampling noise.
# REPEATS>1 is only meaningful with TEMPERATURE>0 (set e.g. TEMPERATURE=0.7).
REPEATS="${REPEATS:-1}"
TEMPERATURE="${TEMPERATURE:-0.0}"
SEED_BASE="${SEED_BASE:-}"

# Resolve task list
if [[ "$TASKS" == "all" ]]; then
  IFS=' ' read -ra TASK_LIST <<< "$ALL_TASKS"
else
  IFS=',' read -ra TASK_LIST <<< "$TASKS"
fi

start_vllm() {
  local label="$1"   # "recompute" or "injection"
  echo ""
  echo "[vLLM] restart for $label (empty prefix cache)"
  unset VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR || true
  if [[ "$label" == "recompute" ]]; then
    unset VLLM_CONTEXT_SEGMENT_KV_DIR || true
  else
    export VLLM_CONTEXT_SEGMENT_KV_DIR="$KV_DIR"
  fi
  bash "$ROOT/scripts/vllm_stop.sh" || true
  bash "$ROOT/scripts/vllm_start.sh"
  # NOTE: this loop variable must not be named `i` -- the caller's repeat-round
  # loop also uses a global `i` (bash for-loop vars are not function-local),
  # and a name collision here previously clobbered the round counter/sample
  # index with this poll loop's leftover value.
  # 600*2s = 20min: VLLM_CONTEXT_SEGMENT_KV_DIR is eager-loaded into GPU memory
  # at worker startup (gpu_model_runner.py load_dir), and the repair-arm KV
  # dir alone is ~16GB across 72 files, which can take well over the previous
  # 360s budget.
  local ready=0
  for _poll_i in $(seq 1 600); do
    code="$(curl -sS -o /dev/null -w '%{http_code}' --connect-timeout 3 --max-time 10 \
      -H "Authorization: Bearer ${VLLM_API_KEY}" \
      "http://127.0.0.1:${VLLM_PORT}/v1/models" 2>/dev/null || true)"
    if [[ "$code" == "200" ]]; then
      echo "[vLLM] ready"
      ready=1
      break
    fi
    sleep 2
  done
  if [[ "$ready" != "1" ]]; then
    echo "[vLLM] ERROR: server did not become ready within timeout (label=$label)" >&2
    exit 1
  fi
}

run_mode() {
  local task="$1"
  local mode="$2"
  local sample_index="$3"   # "" = single pass using --repeats as-is; N = one repeat round
  local seed_args=()
  [[ -n "$SEED_BASE" ]] && seed_args=(--seed-base "$SEED_BASE")
  local repeat_args=(--repeats "$REPEATS")
  if [[ -n "$sample_index" ]]; then
    repeat_args=(--repeats 1 --sample-index "$sample_index")
  fi
  python "$SCRIPT_DIR/run_decode_compare.py" \
    --tasks "$task" \
    --modes "$mode" \
    --occurrences "$OCCURRENCES" \
    --vllm-port "$VLLM_PORT" \
    --model "$VLLM_SERVED_NAME" \
    --api-key "$VLLM_API_KEY" \
    --output "$OUTPUT" \
    --kv-dir "$KV_DIR" \
    --max-tokens "$MAX_TOKENS" \
    "${repeat_args[@]}" \
    --temperature "$TEMPERATURE" \
    "${seed_args[@]}" \
    --append \
    --skip-existing
}

# Clear output file before starting
if [[ -f "$OUTPUT" ]]; then
  rm "$OUTPUT"
  echo "[init] cleared $OUTPUT"
fi

if [[ "$MODES" == "all" ]]; then
  IFS=' ' read -ra MODE_LIST <<< "$ALL_MODES"
else
  IFS=',' read -ra MODE_LIST <<< "$MODES"
fi

if [[ "$REPEATS" -gt 1 ]]; then
  # Stability phase: round outermost, then task, then mode.
  # One round = all tasks × all modes, each (task, mode) gets a fresh vLLM
  # restart (same 36 restarts as before, just ordered round-first so that
  # sample_index=0 covers all tasks before sample_index=1 begins).
  for ((round_i = 0; round_i < REPEATS; round_i++)); do
    echo ""
    echo "========== ROUND $((round_i + 1))/$REPEATS =========="

    for task in "${TASK_LIST[@]}"; do
      task="${task//,/}"
      task="${task// /}"
      [[ -z "$task" ]] && continue

      echo "--- task=$task ---"

      for mode in "${MODE_LIST[@]}"; do
        mode="${mode// /}"
        [[ -z "$mode" ]] && continue

        if [[ "$mode" == "recompute" ]]; then
          start_vllm recompute
        else
          start_vllm injection
        fi
        echo "  $mode (round $((round_i + 1))/$REPEATS)"
        run_mode "$task" "$mode" "$round_i"
      done
    done
  done
else
  # Single pass (headline phase): task outermost, fresh vLLM per (task, mode).
  for task in "${TASK_LIST[@]}"; do
    task="${task//,/}"
    task="${task// /}"
    [[ -z "$task" ]] && continue

    echo ""
    echo "========== task=$task =========="

    for mode in "${MODE_LIST[@]}"; do
      mode="${mode// /}"
      [[ -z "$mode" ]] && continue

      if [[ "$mode" == "recompute" ]]; then
        start_vllm recompute
      else
        start_vllm injection
      fi
      echo "--- $mode ---"
      run_mode "$task" "$mode" ""
    done
  done
fi

echo ""
echo "[done] all tasks complete. output: $OUTPUT"
