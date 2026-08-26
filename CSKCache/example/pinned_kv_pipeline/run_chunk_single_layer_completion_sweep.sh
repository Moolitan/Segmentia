#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
EXAMPLE_DIR="$ROOT/CSKCache/example/pinned_kv_pipeline"

if (($#)); then
  echo "This sweep takes no command-line arguments; edit chunk_single_layer_completion_config.py." >&2
  exit 2
fi
if [[ "${CONDA_DEFAULT_ENV:-}" != "opencode" ]]; then
  echo "Activate the opencode conda environment first." >&2
  exit 2
fi

export PYTHONPATH="$EXAMPLE_DIR:$ROOT/vllm:$ROOT/LMCache:$ROOT/CSKCache${PYTHONPATH:+:$PYTHONPATH}"
export CSKCACHE_COMPLETION_SWEEP_CONFIG="chunk_single_layer_completion_config"
export MPLCONFIGDIR="/tmp/cskcache-chunk-single-layer-completion-matplotlib"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
python "$EXAMPLE_DIR/chunk_single_layer_completion_sweep.py"
