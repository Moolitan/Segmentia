#!/usr/bin/env bash
# Value-side repair controlled experiment (2x2 oracle ablation).
#
# Goal: separate how much of the reuse behavior gap is caused by a stale KEY
# (position mismatch, already addressed by RoPE) versus a stale VALUE
# (representational drift from the no-context skill prefill).
#
# Arms decoded against the recompute reference:
#   rope   = skill key (RoPE-corrected)      + skill value          (current best reuse)
#   vrep   = skill key (RoPE-corrected)      + recompute oracle value
#   krep   = recompute oracle key            + skill value
#   oracle = recompute oracle key            + recompute oracle value  (~= recompute; splice sanity)
#
# Reading: rope->vrep isolates the VALUE repair effect, rope->krep isolates the
# KEY repair effect, oracle->recompute checks the splice path is faithful.
#
# Phases (per-task vLLM restarts keep the prefix cache from bleeding across modes):
#   A. recompute + dump in-context oracle KV (needs cksim SAVE dir, no KV dir)
#   B. build mixed per-case KV files offline (no server)
#   C. decode rope (skill KV dir) and vrep/krep/oracle (repair KV dir)
#
# Usage:
#   bash run_value_repair_compare.sh
#   TASKS=internal_comms_incident_update bash run_value_repair_compare.sh
#   REPEATS=5 TEMPERATURE=0.7 SEED_BASE=1000 bash run_value_repair_compare.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true

export VLLM_PORT="${VLLM_PORT:-8000}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export VLLM_SERVED_NAME="${VLLM_SERVED_NAME:-Qwen3}"
# Lower GPU utilization than the default 0.9: the repair KV dir is eager-loaded
# into GPU at worker startup on top of the model weights. At 0.9 the largest
# task (mcp_server_and_spec, 4.8 GB) overflows the remaining VRAM. 0.80 leaves
# ~9.5 GiB headroom on a 47 GiB card, safely above the largest per-task dir.
export VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.80}"

RES="$ROOT/results/problem_exploration"
SEGMENTIA_OUTPUT_DIR="${SEGMENTIA_OUTPUT_DIR:-/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/06_context_free_segment_cache}"
ALL_TASKS="internal_comms_incident_update doc_coauthoring_design_doc mcp_server_and_spec web_artifact_with_theme launch_poster_page_pack slack_launch_pack"
TASKS="${TASKS:-all}"
OCCURRENCES="${OCCURRENCES:-2,3}"
OUTPUT="${OUTPUT:-$RES/value_repair_key_value_diagnosis/data/decode_outputs_value_repair.jsonl}"
SKILL_KV_DIR="${SKILL_KV_DIR:-$SEGMENTIA_OUTPUT_DIR/offline_skill_kv}"
CKSIM_KV_DIR="${CKSIM_KV_DIR:-$SEGMENTIA_OUTPUT_DIR/cksim_kv}"
REPAIR_KV_DIR="${REPAIR_KV_DIR:-$SEGMENTIA_OUTPUT_DIR/repair_arms_kv}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
REPEATS="${REPEATS:-1}"
TEMPERATURE="${TEMPERATURE:-0.0}"
SEED_BASE="${SEED_BASE:-}"
RUN_PHASE_A="${RUN_PHASE_A:-1}"
RUN_PHASE_B="${RUN_PHASE_B:-1}"
RUN_PHASE_C="${RUN_PHASE_C:-1}"

if [[ "$TASKS" == "all" ]]; then
  IFS=' ' read -ra TASK_LIST <<< "$ALL_TASKS"
else
  IFS=',' read -ra TASK_LIST <<< "$TASKS"
fi

start_vllm() {
  local kind="$1"   # none | skill | repair ; always saves cksim dumps for recompute
  local task="${2:-}"  # required for kind=repair, selects the per-task subdir
  echo ""
  echo "[vLLM] restart ($kind KV dir)"
  unset VLLM_CONTEXT_SEGMENT_KV_DIR || true
  unset VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR || true
  case "$kind" in
    none)   export VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR="$CKSIM_KV_DIR" ;;  # dump oracle KV
    skill)  export VLLM_CONTEXT_SEGMENT_KV_DIR="$SKILL_KV_DIR" ;;
    repair)
      # build_repair_arms_kv.py writes under <REPAIR_KV_DIR>/<task>/ so each
      # restart only eager-loads this task's ~6 files instead of all tasks'
      # ~72 files (~16GB) -- see the comment there for why eager-load exists.
      [[ -n "$task" ]] || { echo "start_vllm repair requires a task arg" >&2; exit 1; }
      export VLLM_CONTEXT_SEGMENT_KV_DIR="$REPAIR_KV_DIR/$task"
      ;;
  esac
  bash "$ROOT/scripts/vllm_stop.sh" || true
  bash "$ROOT/scripts/vllm_start.sh"
  # 600*2s = 20min: VLLM_CONTEXT_SEGMENT_KV_DIR is eager-loaded into GPU memory
  # at worker startup (gpu_model_runner.py load_dir). The repair-arm KV dir
  # (kind=repair) is ~16GB across 72 files, which can take well over the
  # previous 360s budget -- the old code silently gave up and let the script
  # fall through to send requests at a server that was never actually ready,
  # producing a wall of "connection refused" rows instead of a clear failure.
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
    echo "[vLLM] ERROR: server did not become ready within timeout (kind=$kind)" >&2
    exit 1
  fi
}

run_mode() {
  local task="$1"; local mode="$2"; shift 2
  local extra=("$@")
  local seed_args=()
  [[ -n "$SEED_BASE" ]] && seed_args=(--seed-base "$SEED_BASE")
  python "$SCRIPT_DIR/run_decode_compare.py" \
    --tasks "$task" \
    --modes "$mode" \
    --occurrences "$OCCURRENCES" \
    --vllm-port "$VLLM_PORT" \
    --model "$VLLM_SERVED_NAME" \
    --api-key "$VLLM_API_KEY" \
    --output "$OUTPUT" \
    --kv-dir "$SKILL_KV_DIR" \
    --cksim-kv-dir "$CKSIM_KV_DIR" \
    --max-tokens "$MAX_TOKENS" \
    --repeats "$REPEATS" \
    --temperature "$TEMPERATURE" \
    "${seed_args[@]}" \
    --append --skip-existing \
    "${extra[@]}"
}

# Only wipe OUTPUT when Phase A will run; if Phase A/B are skipped (e.g. rerunning
# just Phase C after a partial failure), --append --skip-existing in run_mode
# already handles incremental resume against the existing file.
if [[ "$RUN_PHASE_A" == "1" && -f "$OUTPUT" ]]; then
  rm "$OUTPUT"
  echo "[init] cleared $OUTPUT"
fi

if [[ "$RUN_PHASE_A" == "1" ]]; then
  echo "########## Phase A: recompute + dump oracle KV ##########"
  for task in "${TASK_LIST[@]}"; do
    task="${task//,/}"; task="${task// /}"; [[ -z "$task" ]] && continue
    echo "========== A task=$task =========="
    start_vllm none
    run_mode "$task" recompute --dump-kv-for-cksim
  done
else
  echo "########## Phase A: skipped (RUN_PHASE_A=0) ##########"
fi

echo ""
if [[ "$RUN_PHASE_B" == "1" ]]; then
  echo "########## Phase B: build mixed repair-arm KV (offline) ##########"
  python "$SCRIPT_DIR/build_repair_arms_kv.py" \
    --tasks "$TASKS" \
    --occurrences "$OCCURRENCES" \
    --kv-dir "$SKILL_KV_DIR" \
    --cksim-kv-dir "$CKSIM_KV_DIR" \
    --out-dir "$REPAIR_KV_DIR"
else
  echo "########## Phase B: skipped (RUN_PHASE_B=0) ##########"
fi

echo ""
if [[ "$RUN_PHASE_C" == "1" ]]; then
  echo "########## Phase C: decode reuse arms ##########"
  for task in "${TASK_LIST[@]}"; do
    task="${task//,/}"; task="${task// /}"; [[ -z "$task" ]] && continue
    echo "========== C task=$task =========="

    start_vllm skill
    echo "--- rope (baseline) ---"
    run_mode "$task" rope

    start_vllm repair "$task"
    echo "--- vrep (value repaired) ---"
    run_mode "$task" vrep

    start_vllm repair "$task"
    echo "--- krep (key repaired) ---"
    run_mode "$task" krep

    start_vllm repair "$task"
    echo "--- oracle (both, splice sanity) ---"
    run_mode "$task" oracle
  done
else
  echo "########## Phase C: skipped (RUN_PHASE_C=0) ##########"
fi

echo ""
echo "[done] value-repair experiment complete. output: $OUTPUT"
echo "[next] evaluate:"
echo "  python $SCRIPT_DIR/evaluate_outputs.py --input $OUTPUT \\"
echo "    --cksim-kv-dir $CKSIM_KV_DIR \\"
echo "    --metrics-csv $RES/value_repair_key_value_diagnosis/tables/value_repair_metrics_rows.csv \\"
echo "    --stability-csv $RES/value_repair_key_value_diagnosis/tables/value_repair_stability_rows.csv \\"
echo "    --metrics-json $RES/value_repair_key_value_diagnosis/data/value_repair_summary.json"
