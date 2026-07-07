#!/usr/bin/env bash
# Attention probe: capture attention at the first thinking token for
# recompute vs rope comparison.
#
# Two runs:
#   recompute   seed=1111   (full recomputation, no KV reuse)
#   rope        seed=1111   (KV reuse with RoPE position correction)
#
# Same seed ensures identical sampling RNG — any attention difference
# is purely from the KV cache mechanism.
#
# No external CSV dependency: the query position (prompt_tokens + 1) is
# deterministic from the prompt alone.
#
# IMPORTANT: vLLM must run with VLLM_ENFORCE_EAGER=1 (no CUDA graphs)
# for the attention probe to work.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$ROOT"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true

export VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B}"
export VLLM_SERVED_NAME="${VLLM_SERVED_NAME:-Qwen3}"
export VLLM_PORT="${VLLM_PORT:-8000}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"

# Sampling parameters.
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
MIN_P="${MIN_P:-0}"

TEMP_TAG="${TEMP_TAG:-temp${TEMPERATURE}}"
RUN_LABEL="${RUN_LABEL:-without_occ12}"

SEGMENTIA_OUTPUT_DIR="${SEGMENTIA_OUTPUT_DIR:-/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/06_context_free_segment_cache}"
KV_DIR="${KV_DIR:-$SEGMENTIA_OUTPUT_DIR/offline_skill_kv}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
TASKS="${TASKS:-internal_comms_incident_update,doc_coauthoring_design_doc,mcp_server_and_spec,web_artifact_with_theme,launch_poster_page_pack,slack_launch_pack}"

# Query positions to probe, as token offsets after the first thinking token
# (0 = first thinking token, i.e. prompt_tokens+1). All positions are captured
# in a single decode pass, then split into per-offset directories.
QUERY_OFFSETS="${QUERY_OFFSETS:-0,30,60,90,120}"

# Attention probe output.
ATTN_OUT_DIR="${ATTN_OUT_DIR:-$SEGMENTIA_OUTPUT_DIR/cross_occurrence_function_vector/attention_divergence/$TEMP_TAG/$RUN_LABEL}"
PROBE_MARKER="${PROBE_MARKER:-$ATTN_OUT_DIR/current_attention_probe.json}"

# All offsets dump into one combined dir during generation, then split by
# offset into $ATTN_OUT_DIR/offset_<NNN>/dumps below.
COMBINED_DIR="$ATTN_OUT_DIR/_combined"
COMBINED_DUMP_DIR="$COMBINED_DIR/dumps"
MANIFEST="$COMBINED_DIR/query_position_manifest.jsonl"

# Per-offset figures + summary CSV land here (one subdir per offset).
RESULTS_BASE="${RESULTS_BASE:-$ROOT/results/problem_exploration/raw_decode_token_sequences/attention_divergence/$TEMP_TAG}"

# Two runs: recompute + rope, same seed.
RUN_MODES=(recompute rope)
RUN_SEEDS=(1111 1111)
RUN_NAMES=(recompute rope)

# ── vLLM attention probe environment ──
# All layers (Qwen3-14B has 40 layers: 0-39).
export VLLM_SEGMENTIA_ATTENTION_PROBE_LAYERS="${VLLM_SEGMENTIA_ATTENTION_PROBE_LAYERS:-all}"
export VLLM_SEGMENTIA_ATTENTION_PROBE_OUT_DIR="$COMBINED_DUMP_DIR"
export VLLM_SEGMENTIA_ATTENTION_PROBE_CURRENT="$PROBE_MARKER"
export VLLM_SEGMENTIA_ATTENTION_PROBE_MAX_WINDOW_TOKENS="${VLLM_SEGMENTIA_ATTENTION_PROBE_MAX_WINDOW_TOKENS:-4096}"
# CUDA graphs must be disabled for the attention probe hook.
export VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"

mkdir -p "$COMBINED_DUMP_DIR"
# Fresh manifest + combined dumps each run so offset splitting is deterministic.
rm -f "$MANIFEST"
rm -f "$COMBINED_DUMP_DIR"/attention_region_mass_*.jsonl \
      "$COMBINED_DUMP_DIR"/attention_local_windows_*.jsonl

start_vllm() {
  local mode="$1"
  local task="$2"

  echo ""
  echo "[vLLM] restart boundary=(mode=$mode, task=$task)"
  unset VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR || true
  if [[ "$mode" == "recompute" ]]; then
    unset VLLM_CONTEXT_SEGMENT_KV_DIR || true
  else
    export VLLM_CONTEXT_SEGMENT_KV_DIR="$KV_DIR"
  fi

  bash "$ROOT/scripts/vllm_stop.sh" || true
  bash "$ROOT/scripts/vllm_start.sh"

  local ready=0
  for _poll_i in $(seq 1 600); do
    code="$(curl -sS -o /dev/null -w '%{http_code}' \
      --connect-timeout 3 --max-time 10 \
      -H "Authorization: Bearer ${VLLM_API_KEY}" \
      "http://127.0.0.1:${VLLM_PORT}/v1/models" 2>/dev/null || true)"
    if [[ "$code" == "200" ]]; then
      ready=1
      echo "[vLLM] ready (ENFORCE_EAGER=$VLLM_ENFORCE_EAGER)"
      break
    fi
    sleep 2
  done
  if [[ "$ready" != "1" ]]; then
    echo "[error] vLLM start timeout for mode=$mode task=$task" >&2
    exit 1
  fi
}

IFS=',' read -ra TASK_LIST <<< "$TASKS"

for idx in "${!RUN_MODES[@]}"; do
  mode="${RUN_MODES[$idx]}"
  seed="${RUN_SEEDS[$idx]}"
  run_name="${RUN_NAMES[$idx]}"

  echo ""
  echo "########## attention probe: run=$run_name (mode=$mode, seed=$seed) ##########"

  for task in "${TASK_LIST[@]}"; do
    task="${task// /}"
    [[ -z "$task" ]] && continue
    start_vllm "$mode" "$task"

    probe_args=(
      --task "$task"
      --mode "$mode"
      --run-name "$run_name"
      --probe-marker "$PROBE_MARKER"
      --vllm-port "$VLLM_PORT"
      --model "$VLLM_SERVED_NAME"
      --api-key "$VLLM_API_KEY"
      --kv-dir "$KV_DIR"
      --max-tokens "$MAX_TOKENS"
      --temperature "$TEMPERATURE"
      --top-p "$TOP_P"
      --seed "$seed"
      --query-offsets "$QUERY_OFFSETS"
      --manifest "$MANIFEST"
    )
    [[ -n "$TOP_K" ]] && probe_args+=(--top-k "$TOP_K")
    [[ -n "$MIN_P" ]] && probe_args+=(--min-p "$MIN_P")
    python "$SCRIPT_DIR/run_attention_divergence_probe.py" "${probe_args[@]}"
  done
done

# ── Split combined dumps into per-offset dirs, then plot each offset ──
echo ""
echo "########## split dumps by query offset ##########"
python "$SCRIPT_DIR/split_dumps_by_query_offset.py" \
  --combined-dump-dir "$COMBINED_DUMP_DIR" \
  --manifest "$MANIFEST" \
  --out-base "$ATTN_OUT_DIR" \
  --offsets "$QUERY_OFFSETS"

echo ""
echo "########## plot per offset ##########"
IFS=',' read -ra OFFSET_LIST <<< "$QUERY_OFFSETS"
for off in "${OFFSET_LIST[@]}"; do
  off="${off// /}"
  [[ -z "$off" ]] && continue
  off_tag="$(printf 'offset_%03d' "$off")"
  dump_dir="$ATTN_OUT_DIR/$off_tag/dumps"
  out_dir="$RESULTS_BASE/$off_tag"
  echo "[plot] offset=$off  dumps=$dump_dir  figures=$out_dir"
  python "$SCRIPT_DIR/plot_attention_divergence_probe.py" \
    --dump-dir "$dump_dir" \
    --output-dir "$out_dir"
done

echo ""
echo "[done] runs: ${RUN_NAMES[*]}"
echo "[done] combined dumps: $COMBINED_DUMP_DIR"
echo "[done] per-offset dumps: $ATTN_OUT_DIR/offset_<NNN>/dumps"
echo "[done] per-offset figures: $RESULTS_BASE/offset_<NNN>"
