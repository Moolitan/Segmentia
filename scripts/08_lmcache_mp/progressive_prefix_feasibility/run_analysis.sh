#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CAPTURE_ROOT="${SEGMENTIA_CAPTURE_ROOT:-/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/08_lmcache_mp/cross_request_kv_capture_runs}"
DESIGN_CASE="${DESIGN_CASE:-${CAPTURE_ROOT}/20260728-rehydration-pilot10/doc_coauthoring_to_mcp_doc_coauthoring}"
HELDOUT_ROOT="${HELDOUT_ROOT:-${CAPTURE_ROOT}/20260728-m1a-actual-anchor-v1}"
THEME_CASE="${THEME_CASE:-${HELDOUT_ROOT}/web_artifact_to_launch_theme_factory}"
WEB_BUILDER_CASE="${WEB_BUILDER_CASE:-${HELDOUT_ROOT}/web_artifact_to_launch_web_artifacts_builder}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/problem_exploration/progressive_skill_prefix_recompute}"
PYTHON_BIN="${PYTHON_BIN:-python}"

COMMON_ARGS=(
  --design-case-dir "${DESIGN_CASE}"
  --heldout-case-dir "${THEME_CASE}"
  --heldout-case-dir "${WEB_BUILDER_CASE}"
  --output-dir "${OUTPUT_DIR}"
)

"${PYTHON_BIN}" \
  "${REPO_ROOT}/scripts/08_lmcache_mp/progressive_prefix_feasibility/validate_progressive_prefix.py" \
  "${COMMON_ARGS[@]}" \
  --preflight-only

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  exit 0
fi

MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/segmentia-mpl}" \
  "${PYTHON_BIN}" \
  "${REPO_ROOT}/scripts/08_lmcache_mp/progressive_prefix_feasibility/validate_progressive_prefix.py" \
  "${COMMON_ARGS[@]}"
