#!/usr/bin/env bash
# Pure synthetic workload validation; does not start vLLM or write external SSD.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
work="$(mktemp -d /tmp/segmentia-fixed-prefix-dry-run.XXXXXX)"
trap 'rm -rf "$work"' EXIT
for arm in full direct prefix_no_correction prefix_256; do
  extra_args=()
  if [[ "$arm" != "full" ]]; then
    extra_args+=(--cpu-prefetch)
  fi
  PYTHONPATH="$SCRIPT_DIR" "$PYTHON_BIN" "$SCRIPT_DIR/run_requests.py" \
    --phase measure --arm "$arm" --replica 0 --lengths 512,768,1536,3301 \
    --warmups 1 --measurements 2 --output-dir "$work/$arm" --prepare-only \
    "${extra_args[@]}"
done
PYTHONPATH="$SCRIPT_DIR" "$PYTHON_BIN" "$SCRIPT_DIR/run_requests.py" \
  --phase source --lengths 512,768,1536,3301 --output-dir "$work/source" --prepare-only
echo "[dry-run-valid] synthetic workload prepared under $work"
