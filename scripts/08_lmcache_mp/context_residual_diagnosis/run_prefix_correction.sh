#!/usr/bin/env bash
# Run online continuous-prefix K correction on three long Skills plus one
# short-Skill length-gate control. The user supplies a fresh outer RUN_ID.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CAPTURE="$ROOT/scripts/08_lmcache_mp/cross_request_kv_capture/run_capture.sh"
ANALYZER="$ROOT/scripts/08_lmcache_mp/context_residual_diagnosis/validate_prefix_correction.py"
DOC_BASE="${SEGMENTIA_PREFIX_DOC_BASE:-/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/08_lmcache_mp/cross_request_kv_capture_runs/20260728-rehydration-pilot10}"
HELDOUT_BASE="${SEGMENTIA_PREFIX_HELDOUT_BASE:-/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/08_lmcache_mp/cross_request_kv_capture_runs/20260728-m1a-actual-anchor-v1}"
EXPERIMENT_RUN_ID="${RUN_ID:?Set a fresh RUN_ID for the prefix-correction experiment}"
CAPTURE_ROOT="${SEGMENTIA_PREFIX_CAPTURE_ROOT:-/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/08_lmcache_mp/prefix_correction_capture_runs}/$EXPERIMENT_RUN_ID"
RESULT_DIR="${SEGMENTIA_PREFIX_RESULT_DIR:-$ROOT/results/problem_exploration/contextual_skill_kv_residual_structure/prefix_correction_validation}"
RUN_DOC="${RUN_DOC:-1}"
RUN_HELDOUT="${RUN_HELDOUT:-1}"
RUN_SHORT="${RUN_SHORT:-1}"
RUN_ANALYSIS="${RUN_ANALYSIS:-1}"

doc_case="doc_coauthoring_to_mcp_doc_coauthoring"
theme_case="web_artifact_to_launch_theme_factory"
web_case="web_artifact_to_launch_web_artifacts_builder"
short_case="internal_comms_to_slack_internal_comms"

for required in \
  "$DOC_BASE/$doc_case/manifest.json" \
  "$HELDOUT_BASE/$theme_case/manifest.json" \
  "$HELDOUT_BASE/$web_case/manifest.json" \
  "$HELDOUT_BASE/$short_case/manifest.json"; do
  if [[ "$(jq -r '.status // empty' "$required" 2>/dev/null)" != "completed" ]]; then
    echo "[error] missing completed base manifest: $required" >&2
    exit 2
  fi
done

run_group() {
  local child_run_id="$1"
  local case_ids="$2"
  local base_run="$3"
  RUN_ID="$child_run_id" \
  CASE_IDS="$case_ids" \
  SEGMENTIA_CAPTURE_OUTPUT_ROOT="$CAPTURE_ROOT" \
  SEGMENTIA_REUSE_BASE_RUN="$base_run" \
  SEGMENTIA_REUSE_DIRECTION=forward \
  SEGMENTIA_PREFIX_CORRECTION=1 \
  SEGMENTIA_ACTUAL_ANCHOR_TOKENS=0 \
    bash "$CAPTURE"
}

if [[ "$RUN_DOC" == "1" ]]; then
  echo "[phase 1/4] long Skill: doc-coauthoring"
  run_group doc "$doc_case" "$DOC_BASE"
fi

if [[ "$RUN_HELDOUT" == "1" ]]; then
  echo "[phase 2/4] long Skills: theme-factory and web-artifacts-builder"
  run_group heldout_long "$theme_case,$web_case" "$HELDOUT_BASE"
fi

if [[ "$RUN_SHORT" == "1" ]]; then
  echo "[phase 3/4] short Skill: verify full-local length fallback"
  run_group short_control "$short_case" "$HELDOUT_BASE"
fi

if [[ "$RUN_ANALYSIS" == "1" ]]; then
  echo "[phase 4/4] validate online prefix correction and short fallback"
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/segmentia-mpl}" \
    python "$ANALYZER" \
      --base-case-dir "$DOC_BASE/$doc_case" \
      --prefix-case-dir "$CAPTURE_ROOT/doc/$doc_case" \
      --base-case-dir "$HELDOUT_BASE/$theme_case" \
      --prefix-case-dir "$CAPTURE_ROOT/heldout_long/$theme_case" \
      --base-case-dir "$HELDOUT_BASE/$web_case" \
      --prefix-case-dir "$CAPTURE_ROOT/heldout_long/$web_case" \
      --short-case-dir "$CAPTURE_ROOT/short_control/$short_case" \
      --output-dir "$RESULT_DIR"
fi

echo "[completed] prefix_capture=$CAPTURE_ROOT analysis=$RESULT_DIR"
