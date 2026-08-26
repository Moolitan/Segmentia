#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXAMPLE_DIR="$ROOT/CSKCache/example/synthetic_prefill_reuse_latency"
PINNED_EXAMPLE_DIR="$ROOT/CSKCache/example/pinned_kv_pipeline"

if (($#)); then
  echo "This benchmark takes no command-line arguments; edit config.py." >&2
  exit 2
fi
if [[ "${CONDA_DEFAULT_ENV:-}" != "opencode" ]]; then
  echo "Activate the opencode conda environment first." >&2
  exit 2
fi

export PYTHONPATH="$EXAMPLE_DIR:$PINNED_EXAMPLE_DIR:$ROOT/vllm:$ROOT/LMCache:$ROOT/CSKCache${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="/tmp/cskcache-synthetic-latency-matplotlib"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
python "$EXAMPLE_DIR/run.py"
