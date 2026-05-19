#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy ftp_proxy FTP_PROXY

PY_SCRIPT="$SCRIPT_DIR/Multi-agents/claude_style_multi_agent.py"
WORKSPACE="${MULTI_AGENT_WORKSPACE:-$ROOT/workspace/multi_agents}"
ROOT_SKILLS_DIR="${MULTI_AGENT_ROOT_SKILLS_DIR:-$ROOT/skills}"
OUTPUT_DIR="${MULTI_AGENT_OUTPUT_DIR:-$ROOT/results/multi_agents}"
VLLM_PORT="${VLLM_PORT:-8000}"

WORKSPACE_SKILLS_DIR="$WORKSPACE/.agents/skills"

mkdir -p "$WORKSPACE/.agents" "$OUTPUT_DIR"
rm -rf "$WORKSPACE_SKILLS_DIR"
cp -a "$ROOT_SKILLS_DIR"/. "$WORKSPACE_SKILLS_DIR"/

echo "[skills] synced $ROOT_SKILLS_DIR -> $WORKSPACE_SKILLS_DIR"
echo "[run] workspace: $WORKSPACE"
echo "[run] output:    $OUTPUT_DIR"
echo "[run] script:    $PY_SCRIPT"

python "$PY_SCRIPT" \
  --workspace "$WORKSPACE" \
  --output-dir "$OUTPUT_DIR" \
  --vllm-port "$VLLM_PORT" \
  "$@"
