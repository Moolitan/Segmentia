#!/usr/bin/env bash
# One-replica, four-length preflight. The user runs this; it starts vLLM.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPLICAS="${REPLICAS:-1}"
export LENGTHS="${LENGTHS:-512,768,1536,3301}"
export WARMUPS="${WARMUPS:-1}"
export MEASUREMENTS="${MEASUREMENTS:-5}"
exec bash "$SCRIPT_DIR/run_benchmark.sh"
