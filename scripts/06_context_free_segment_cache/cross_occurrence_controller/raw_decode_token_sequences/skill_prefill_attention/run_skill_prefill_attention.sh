#!/usr/bin/env bash
# Recompute-only prefill attention capture for occurrence-3 Skill spans.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../../.." && pwd)"
cd "$ROOT"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true

export VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B}"
export VLLM_SERVED_NAME="${VLLM_SERVED_NAME:-Qwen3}"
export VLLM_PORT="${VLLM_PORT:-8000}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
# Preserve natural same-task prefix reuse; task boundaries are isolated by restart.
export PREFIX_CACHING=1

TASKS="${TASKS:-internal_comms_incident_update,doc_coauthoring_design_doc,mcp_server_and_spec,web_artifact_with_theme,launch_poster_page_pack,slack_launch_pack}"
SEGMENTIA_OUTPUT_DIR="${SEGMENTIA_OUTPUT_DIR:-/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/06_context_free_segment_cache}"
ATTN_OUT_DIR="${ATTN_OUT_DIR:-$SEGMENTIA_OUTPUT_DIR/skill_prefill_attention/occ3}"
DUMP_DIR="$ATTN_OUT_DIR/dumps"
PROBE_MARKER="$ATTN_OUT_DIR/current_attention_probe.json"
MANIFEST="$ATTN_OUT_DIR/capture_manifest.jsonl"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results/problem_exploration/skill_prefill_attention}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-0}"

export VLLM_SEGMENTIA_ATTENTION_PROBE_LAYERS="${VLLM_SEGMENTIA_ATTENTION_PROBE_LAYERS:-all}"
export VLLM_SEGMENTIA_ATTENTION_PROBE_OUT_DIR="$DUMP_DIR"
export VLLM_SEGMENTIA_ATTENTION_PROBE_CURRENT="$PROBE_MARKER"
export VLLM_SEGMENTIA_ATTENTION_PROBE_MAX_WINDOW_TOKENS="${VLLM_SEGMENTIA_ATTENTION_PROBE_MAX_WINDOW_TOKENS:-4096}"
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"

mkdir -p "$DUMP_DIR"
shopt -s nullglob
existing_dumps=("$DUMP_DIR"/attention_local_windows_*.jsonl)
shopt -u nullglob
if [[ -f "$MANIFEST" || ${#existing_dumps[@]} -gt 0 ]]; then
  if [[ "$CLEAN_OUTPUT" != "1" ]]; then
    echo "[error] output already exists under $ATTN_OUT_DIR" >&2
    echo "Set CLEAN_OUTPUT=1 to replace this derived capture." >&2
    exit 1
  fi
  rm -f "$MANIFEST" "$PROBE_MARKER"
  rm -f "$DUMP_DIR"/attention_local_windows_*.jsonl
  rm -f "$DUMP_DIR"/attention_region_mass_*.jsonl
fi

start_vllm() {
  local task="$1"

  echo "[vLLM] restart boundary=(mode=recompute, task=$task)"
  unset VLLM_CONTEXT_SEGMENT_KV_DIR || true
  unset VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR || true
  bash "$ROOT/scripts/vllm_stop.sh" || true
  bash "$ROOT/scripts/vllm_start.sh"

  local ready=0
  local code
  for _poll_i in $(seq 1 600); do
    code="$(curl -sS -o /dev/null -w '%{http_code}' \
      --connect-timeout 3 --max-time 10 \
      -H "Authorization: Bearer ${VLLM_API_KEY}" \
      "http://127.0.0.1:${VLLM_PORT}/v1/models" 2>/dev/null || true)"
    if [[ "$code" == "200" ]]; then
      ready=1
      break
    fi
    sleep 2
  done
  if [[ "$ready" != "1" ]]; then
    echo "[error] vLLM start timeout for task=$task" >&2
    exit 1
  fi
}

IFS=',' read -ra TASK_LIST <<< "$TASKS"
for task in "${TASK_LIST[@]}"; do
  task="${task// /}"
  [[ -z "$task" ]] && continue
  start_vllm "$task"
  python "$SCRIPT_DIR/run_skill_prefill_attention.py" \
    --task "$task" \
    --probe-marker "$PROBE_MARKER" \
    --manifest "$MANIFEST" \
    --vllm-port "$VLLM_PORT" \
    --model "$VLLM_SERVED_NAME" \
    --api-key "$VLLM_API_KEY"
done

python "$SCRIPT_DIR/plot_skill_prefill_attention.py" \
  --dump-dir "$DUMP_DIR" \
  --manifest "$MANIFEST" \
  --output-dir "$RESULTS_DIR"

echo "[done] raw dumps: $DUMP_DIR"
echo "[done] manifest: $MANIFEST"
echo "[done] figures: $RESULTS_DIR/figures"
