#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$ROOT/scripts/08_lmcache_mp/shared_skill_attention_feasibility"
RUN_ROOT="/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/08_lmcache_mp/prefix_correction_capture_runs/20260728-m2a-prefix-v1"
DOC_BASE="/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/08_lmcache_mp/cross_request_kv_capture_runs/20260728-rehydration-pilot10"
HELDOUT_BASE="/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/08_lmcache_mp/cross_request_kv_capture_runs/20260728-m1a-actual-anchor-v1"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results/problem_exploration/shared_skill_attention_feasibility}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DOC_CAPTURE="$(find "$RUN_ROOT/doc" -path '*/prefix_correction/*.pt' -type f -print -quit)"
THEME_CAPTURE="$(find "$RUN_ROOT/heldout_long/web_artifact_to_launch_theme_factory" -path '*/prefix_correction/*.pt' -type f -print -quit)"
WEB_CAPTURE="$(find "$RUN_ROOT/heldout_long/web_artifact_to_launch_web_artifacts_builder" -path '*/prefix_correction/*.pt' -type f -print -quit)"
DOC_SOURCE_REQUEST="$DOC_BASE/doc_coauthoring_to_mcp_doc_coauthoring/source/request.json"
DOC_TARGET_REQUEST="$RUN_ROOT/doc/doc_coauthoring_to_mcp_doc_coauthoring/target_reuse/request.json"
THEME_SOURCE_REQUEST="$HELDOUT_BASE/web_artifact_to_launch_theme_factory/source/request.json"
THEME_TARGET_REQUEST="$RUN_ROOT/heldout_long/web_artifact_to_launch_theme_factory/target_reuse/request.json"
WEB_SOURCE_REQUEST="$HELDOUT_BASE/web_artifact_to_launch_web_artifacts_builder/source/request.json"
WEB_TARGET_REQUEST="$RUN_ROOT/heldout_long/web_artifact_to_launch_web_artifacts_builder/target_reuse/request.json"

for capture in "$DOC_CAPTURE" "$THEME_CAPTURE" "$WEB_CAPTURE"; do
  if [[ -z "$capture" || ! -f "$capture" ]]; then
    echo "[error] missing Prefix-256 capture: $capture" >&2
    exit 2
  fi
done
for request in \
  "$DOC_SOURCE_REQUEST" "$DOC_TARGET_REQUEST" \
  "$THEME_SOURCE_REQUEST" "$THEME_TARGET_REQUEST" \
  "$WEB_SOURCE_REQUEST" "$WEB_TARGET_REQUEST"; do
  if [[ ! -f "$request" ]]; then
    echo "[error] missing request metadata: $request" >&2
    exit 2
  fi
done

DOC_SOURCE_START="$(jq -er '.segment_start' "$DOC_SOURCE_REQUEST")"
DOC_TARGET_START="$(jq -er '.segment_start' "$DOC_TARGET_REQUEST")"
THEME_SOURCE_START="$(jq -er '.segment_start' "$THEME_SOURCE_REQUEST")"
THEME_TARGET_START="$(jq -er '.segment_start' "$THEME_TARGET_REQUEST")"
WEB_SOURCE_START="$(jq -er '.segment_start' "$WEB_SOURCE_REQUEST")"
WEB_TARGET_START="$(jq -er '.segment_start' "$WEB_TARGET_REQUEST")"

PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" "$SCRIPT_DIR/validate_equivalence.py" \
  --case doc_coauthoring "$DOC_CAPTURE" "$DOC_SOURCE_START" "$DOC_TARGET_START" \
  --case theme_factory "$THEME_CAPTURE" "$THEME_SOURCE_START" "$THEME_TARGET_START" \
  --case web_artifacts_builder "$WEB_CAPTURE" "$WEB_SOURCE_START" "$WEB_TARGET_START" \
  --output-dir "$OUTPUT_DIR" \
  --overwrite
