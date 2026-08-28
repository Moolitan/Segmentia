#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHONPATH_VALUE="$ROOT/CSKCache:$ROOT/LMCache:$ROOT/vllm${PYTHONPATH:+:$PYTHONPATH}"

# These settings affect only the offline staging allocator.  A 256-token
# LMCache page is about 40 MiB for all 40 Qwen3-14B layers, smaller than the
# 54.5-MiB single-layer proof-checker object.  A 512-token page fits it.  The
# local CPU backend remains an allocator but not a hot cache, so completed raw
# writes release their staging pages instead of retaining 40 pages per Skill.
export LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-512}"
export LMCACHE_LOCAL_CPU="${LMCACHE_LOCAL_CPU:-False}"

case "${1:-}" in
  --plan)
    PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$SCRIPT_DIR/prepare_sources.py" --json
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

PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$SCRIPT_DIR/prepare_sources.py" --prepare
export OFFLINE_CONFIG_PATH="$SCRIPT_DIR/config.py"

# Recover the precise failure state produced before the staging-page fix: raw
# keys exist for a prefix of the six objects, all pending records are present,
# and no Catalog was published.  The generic recovery path moves the old
# transaction to raw/.failed (never deletes it), then the build recreates the
# six pending records while reusing raw keys that are already durable.
eval "$(PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" \
  "$ROOT/CSKCache/example/offline_skill_kv/config_loader.py")"
RAW_DIR="$SKILL_SAVE_POOL_ROOT/$SKILL_POOL_MODEL_DIR_NAME/raw"
shopt -s nullglob
PENDING_RECORDS=("$RAW_DIR/.pending/"*.json)
shopt -u nullglob
if ((${#PENDING_RECORDS[@]} > 0)) && [[ ! -f "$RAW_DIR/catalog.json" ]]; then
  export FIXED_LENGTH_OFFLINE_OVERWRITE=1
  echo "[recovery] quarantining the unpublished pending transaction before resume"
fi
bash "$ROOT/CSKCache/example/offline_skill_kv/run.sh"
PYTHONPATH="$PYTHONPATH_VALUE" "$PYTHON_BIN" "$SCRIPT_DIR/verify_cache.py"
