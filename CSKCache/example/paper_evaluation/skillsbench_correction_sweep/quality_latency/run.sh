#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
EXAMPLE_DIR="$ROOT/CSKCache/example"
PYTHONPATH_VALUE="$EXAMPLE_DIR:$ROOT/CSKCache:$ROOT/LMCache:$ROOT/vllm${PYTHONPATH:+:$PYTHONPATH}"

source /home/wsh/miniconda3/etc/profile.d/conda.sh
conda activate opencode
set -u

MODULE="paper_evaluation.skillsbench_correction_sweep.quality_latency"
case "${1:-}" in
  --plan)
    if (($# != 1)); then
      echo "usage: bash run.sh [--plan | --analyze RUN_DIR | --recover RUN_DIR]" >&2
      exit 2
    fi
    PYTHONPATH="$PYTHONPATH_VALUE" python -m "$MODULE.plan"
    ;;
  --analyze)
    if (($# != 2)); then
      echo "usage: bash run.sh [--plan | --analyze RUN_DIR | --recover RUN_DIR]" >&2
      exit 2
    fi
    PYTHONPATH="$PYTHONPATH_VALUE" python -m "$MODULE.analyze" "$2"
    ;;
  --recover)
    if (($# != 2)); then
      echo "usage: bash run.sh [--plan | --analyze RUN_DIR | --recover RUN_DIR]" >&2
      exit 2
    fi
    PYTHONPATH="$PYTHONPATH_VALUE" python -m "$MODULE.recover" "$2"
    ;;
  "")
    PYTHONPATH="$PYTHONPATH_VALUE" python -m "$MODULE.preflight"
    PYTHONPATH="$PYTHONPATH_VALUE" python -m "$MODULE.run"
    ;;
  *)
    echo "usage: bash run.sh [--plan | --analyze RUN_DIR | --recover RUN_DIR]" >&2
    exit 2
    ;;
esac
