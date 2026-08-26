#!/usr/bin/env bash
set -eo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SUITE_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"
EXAMPLE_DIR="$(cd -- "$SUITE_DIR/.." && pwd)"
REPO_DIR="$(cd -- "$EXAMPLE_DIR/../.." && pwd)"
if (($#)); then
  echo "[error] this launcher accepts no arguments; edit config.py" >&2
  exit 2
fi
source /home/wsh/miniconda3/etc/profile.d/conda.sh
conda activate opencode
set -u
PYTHONPATH="$SCRIPT_DIR:$EXAMPLE_DIR:$SUITE_DIR:$SUITE_DIR/common:$REPO_DIR/vllm:${PYTHONPATH:-}" \
  python "$SCRIPT_DIR/run.py"
