#!/usr/bin/env bash
# RUN_ID=h2d_01 bash scripts/07_cskcache/h2d_microbenchmark/run.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BENCH_DIR="$ROOT/scripts/07_cskcache/h2d_microbenchmark"
CSKCACHE_ROOT="$ROOT/CSKCache"
VLLM_ROOT="${VLLM_ROOT:-/home/wsh/vllm}"
KV_DIR="${CSKCACHE_KV_DIR:-/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/07_cskcache/offline_skill_kv}"
CACHE_ID="${CSKCACHE_H2D_CACHE_ID:-doc-coauthoring}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
OUTPUT_ROOT="${CSKCACHE_H2D_OUTPUT_ROOT:-/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/07_cskcache/h2d_microbenchmark}"
RUN_DIR="${CSKCACHE_H2D_RUN_DIR:-$OUTPUT_ROOT/$RUN_ID}"
WARMUP="${CSKCACHE_H2D_WARMUP:-5}"
REPETITIONS="${CSKCACHE_H2D_REPETITIONS:-30}"
DEVICE="${CSKCACHE_H2D_DEVICE:-cuda:0}"
CONDA_ENV="${CSKCACHE_CONDA_ENV:-opencode}"

OVERWRITE=0
DRY_RUN=0
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --overwrite) OVERWRITE=1 ;;
    --dry-run) DRY_RUN=1 ;;
    *) echo "usage: $0 [--overwrite] [--dry-run]" >&2; exit 2 ;;
  esac
  shift
done

set +u
source /home/wsh/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
set -u
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$CSKCACHE_ROOT:$VLLM_ROOT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$RUN_DIR/cases"
LOG_FILE="$RUN_DIR/run.log"

run_case() {
  local profiling="$1"
  local memory="$2"
  local shift="$3"
  local case_id="profile-${profiling}_memory-${memory}_shift-${shift}"
  local output="$RUN_DIR/cases/$case_id.jsonl"
  local command=(
    python "$BENCH_DIR/run_case.py"
    --case-id "$case_id"
    --kv-dir "$KV_DIR"
    --cache-id "$CACHE_ID"
    --output "$output"
    --profiling "$profiling"
    --memory "$memory"
    --position-shift "$shift"
    --warmup "$WARMUP"
    --repetitions "$REPETITIONS"
    --device "$DEVICE"
  )
  if [[ "$OVERWRITE" == "1" ]]; then
    command+=(--overwrite)
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[dry-run]'
    printf ' %q' "${command[@]}"
    printf '\n'
    return
  fi
  echo "[start] case_id=$case_id" | tee -a "$LOG_FILE"
  "${command[@]}" 2>&1 | tee -a "$LOG_FILE"
}

# Interleave profiling on/off within each memory/shift condition. Every case
# runs in a fresh Python process; a completed case file is the resume boundary.
for shift in 0 17000; do
  for memory in pageable pinned; do
    for profiling in off on; do
      run_case "$profiling" "$memory" "$shift"
    done
  done
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] summary output=$RUN_DIR"
  exit 0
fi

python "$BENCH_DIR/summarize.py" \
  --case-dir "$RUN_DIR/cases" \
  --output-dir "$RUN_DIR" \
  --expected-cases 8 \
  --expected-repetitions "$REPETITIONS" \
  2>&1 | tee -a "$LOG_FILE"

echo "[completed] run_dir=$RUN_DIR"
