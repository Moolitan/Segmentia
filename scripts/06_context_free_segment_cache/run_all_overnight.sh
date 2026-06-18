#!/usr/bin/env bash
# One-shot overnight driver for experiment 06. Runs everything sequentially on a
# single GPU (decode and the embedding metrics cannot share the GPU, so this is
# NOT parallel), then evaluates and plots.
#
# Phases:
#   1. headline  (a,b): recompute/direct/rope, temperature 0, 1 sample
#                       -> reproduces the original comparison with the new
#                          reasoning + action-level metrics.
#   2. stability (c)   : recompute/direct/rope, temperature>0, N samples
#                       -> measures whether trajectory divergences are stable or
#                          just sampling noise (action_self_consistency floor).
#   3. value-repair (d): rope/vrep/krep/oracle 2x2 oracle ablation
#                       -> isolates whether the residual gap is a key or a value
#                          problem.
#   4. evaluate + plot all three.
#
# Each sub-script restarts vLLM between modes to stop prefix-cache bleed; that is
# why this cannot be a single python call.
#
# Launch (detached, survives logout):
#   cd /home/wsh/openhands_code_research
#   nohup bash scripts/06_context_free_segment_cache/run_all_overnight.sh \
#     > log/06_overnight_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# Toggle phases: RUN_HEADLINE/RUN_STABILITY/RUN_VALUE_REPAIR=0
# NOT -e: a failed phase must not abort the rest.
# NOT -u: `conda activate` runs the env's (de)activate hooks, which reference
# unset variables and would trip nounset; all vars below use ${VAR:-default}.
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RES="$ROOT/results/problem_exploration"
SEGMENTIA_OUTPUT_DIR="${SEGMENTIA_OUTPUT_DIR:-/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/06_context_free_segment_cache}"
cd "$ROOT"

# --- conda: vllm_start.sh relies on the active env (python + CONDA_PREFIX) ---
source /home/wsh/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate opencode 2>/dev/null || true
echo "[env] python=$(which python)  CONDA_PREFIX=${CONDA_PREFIX:-unset}"
if ! python -c "import torch" 2>/dev/null; then
  echo "[env] ERROR: opencode env not active (torch import failed); aborting." >&2
  exit 1
fi

export VLLM_PORT="${VLLM_PORT:-8000}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export VLLM_SERVED_NAME="${VLLM_SERVED_NAME:-Qwen3}"

TASKS="${TASKS:-all}"
OCCURRENCES="${OCCURRENCES:-2,3}"
MAX_TOKENS="${MAX_TOKENS:-4096}"

RUN_HEADLINE="${RUN_HEADLINE:-1}"
RUN_STABILITY="${RUN_STABILITY:-1}"
RUN_VALUE_REPAIR="${RUN_VALUE_REPAIR:-1}"

STAB_REPEATS="${STAB_REPEATS:-1}"
STAB_TEMP="${STAB_TEMP:-0.7}"
STAB_SEED_BASE="${STAB_SEED_BASE:-1000}"

ts() { date +'%Y-%m-%d %H:%M:%S'; }
banner() { echo ""; echo "############################################################"; echo "# [$(ts)] $*"; echo "############################################################"; }

# ---------------------------------------------------------------- phase 1
if [[ "$RUN_HEADLINE" == "1" ]]; then
  banner "PHASE 1 headline (a,b): recompute/direct/rope @ temp=0"
  TASKS="$TASKS" OCCURRENCES="$OCCURRENCES" MAX_TOKENS="$MAX_TOKENS" \
    REPEATS=1 TEMPERATURE=0.0 \
    OUTPUT="$RES/headline_semantic_action_gap/data/decode_outputs.jsonl" \
    bash "$SCRIPT_DIR/run_decode_compare.sh" \
    || echo "[$(ts)] WARN phase 1 returned non-zero"
fi

# ---------------------------------------------------------------- phase 2
if [[ "$RUN_STABILITY" == "1" ]]; then
  banner "PHASE 2 stability (c): recompute/direct/rope @ temp=$STAB_TEMP x$STAB_REPEATS"
  TASKS="$TASKS" OCCURRENCES="$OCCURRENCES" MAX_TOKENS="$MAX_TOKENS" \
    REPEATS="$STAB_REPEATS" TEMPERATURE="$STAB_TEMP" SEED_BASE="$STAB_SEED_BASE" \
    OUTPUT="$RES/stability_systematic_vs_noise/data/decode_outputs_stability.jsonl" \
    bash "$SCRIPT_DIR/run_decode_compare.sh" \
    || echo "[$(ts)] WARN phase 2 returned non-zero"
fi

# ---------------------------------------------------------------- phase 3
if [[ "$RUN_VALUE_REPAIR" == "1" ]]; then
  banner "PHASE 3 value-repair (d): rope/vrep/krep/oracle 2x2 @ temp=0"
  TASKS="$TASKS" OCCURRENCES="$OCCURRENCES" MAX_TOKENS="$MAX_TOKENS" \
    REPEATS=1 TEMPERATURE=0.0 \
    OUTPUT="$RES/value_repair_key_value_diagnosis/data/decode_outputs_value_repair.jsonl" \
    bash "$SCRIPT_DIR/run_value_repair_compare.sh" \
    || echo "[$(ts)] WARN phase 3 returned non-zero"
fi

# ---------------------------------------------------------------- free the GPU
banner "Stopping vLLM before GPU-bound evaluation"
bash "$ROOT/scripts/vllm_stop.sh" || true
sleep 5

# ---------------------------------------------------------------- evaluate
# Prefer full (embedding cosine) eval; fall back to BLEU/ROUGE/action/cksim only.
eval_run() {
  local input="$1"; local cksim="$2"; local metrics_csv="$3"; local stability_csv="$4"; local metrics_json="$5"
  [[ -f "$input" ]] || { echo "[$(ts)] skip eval, missing $input"; return; }
  echo "[$(ts)] evaluating $input"
  python "$SCRIPT_DIR/evaluate_outputs.py" \
    --input "$input" \
    --cksim-kv-dir "$cksim" \
    --metrics-csv "$metrics_csv" \
    --stability-csv "$stability_csv" \
    --metrics-json "$metrics_json" \
  || python "$SCRIPT_DIR/evaluate_outputs.py" \
    --input "$input" \
    --cksim-kv-dir "$cksim" \
    --metrics-csv "$metrics_csv" \
    --stability-csv "$stability_csv" \
    --metrics-json "$metrics_json" \
    --skip-embedding
}

banner "PHASE 4 evaluate + plot"
eval_run "$RES/headline_semantic_action_gap/data/decode_outputs.jsonl" \
  "$SEGMENTIA_OUTPUT_DIR/cksim_kv" \
  "$RES/headline_semantic_action_gap/tables/headline_metrics_rows.csv" \
  "$RES/headline_semantic_action_gap/tables/headline_stability_rows.csv" \
  "$RES/headline_semantic_action_gap/data/headline_summary.json"
eval_run "$RES/stability_systematic_vs_noise/data/decode_outputs_stability.jsonl" \
  "$SEGMENTIA_OUTPUT_DIR/cksim_kv" \
  "$RES/stability_systematic_vs_noise/tables/stability_metrics_rows.csv" \
  "$RES/stability_systematic_vs_noise/tables/stability_stability_rows.csv" \
  "$RES/stability_systematic_vs_noise/data/stability_summary.json"
eval_run "$RES/value_repair_key_value_diagnosis/data/decode_outputs_value_repair.jsonl" \
  "$SEGMENTIA_OUTPUT_DIR/cksim_kv" \
  "$RES/value_repair_key_value_diagnosis/tables/value_repair_metrics_rows.csv" \
  "$RES/value_repair_key_value_diagnosis/tables/value_repair_stability_rows.csv" \
  "$RES/value_repair_key_value_diagnosis/data/value_repair_summary.json"

[[ -f "$RES/headline_semantic_action_gap/tables/headline_metrics_rows.csv" ]] && \
  python "$SCRIPT_DIR/plot_action_fidelity.py" \
    --input "$RES/headline_semantic_action_gap/tables/headline_metrics_rows.csv" \
    --output "$RES/headline_semantic_action_gap/figures/headline_action_fidelity.png" \
  || echo "[$(ts)] skip plot for headline"
[[ -f "$RES/value_repair_key_value_diagnosis/tables/value_repair_metrics_rows.csv" ]] && \
  python "$SCRIPT_DIR/plot_action_fidelity.py" \
    --input "$RES/value_repair_key_value_diagnosis/tables/value_repair_metrics_rows.csv" \
    --output "$RES/value_repair_key_value_diagnosis/figures/value_repair_action_fidelity.png" \
  || echo "[$(ts)] skip plot for value_repair"

banner "DONE. Key outputs in $RES :"
echo "  headline_semantic_action_gap/   (a,b)"
echo "  stability_systematic_vs_noise/  (c: action_self_consistency vs recompute floor)"
echo "  value_repair_key_value_diagnosis/   (d: rope vs vrep vs krep vs oracle)"
echo "[$(ts)] overnight run complete."
