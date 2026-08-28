#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHONPATH_VALUE="$ROOT/CSKCache:$ROOT/LMCache:$ROOT/vllm${PYTHONPATH:+:$PYTHONPATH}"

case "${1:-}" in
  --plan)
    PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$SCRIPT_DIR/inventory.py" --json
    exit 0
    ;;
  --verify)
    PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$SCRIPT_DIR/verify_cache.py"
    exit 0
    ;;
  "")
    ;;
  *)
    echo "usage: bash run.sh [--plan|--verify]" >&2
    exit 2
    ;;
esac

PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$SCRIPT_DIR/inventory.py"
export OFFLINE_CONFIG_PATH="$SCRIPT_DIR/config.py"
bash "$ROOT/CSKCache/example/offline_skill_kv/run.sh"
PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$SCRIPT_DIR/verify_cache.py"
