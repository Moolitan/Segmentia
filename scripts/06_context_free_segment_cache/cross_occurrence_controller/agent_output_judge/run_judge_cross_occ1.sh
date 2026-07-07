#!/usr/bin/env bash
# 用通用判官给 cross_task_contextual_occ1_reuse 的两个 arm 打分并对比。
#
# 判官模型 = 本地 vLLM 上的 Qwen3-14B（普通聊天，不注入任何 skill KV）。
# 判官逻辑在 scripts/06_context_free_segment_cache/module/critic_reference/。
#
# 流程：起一个干净的 vLLM（不设 KV 目录）→ 等就绪 → run_judge.py 逐 case 逐 arm 打分
#       → 输出 CSV + 明细 JSONL → 停 vLLM。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$ROOT"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true

# 起服务用 opencode 环境（能 import /home/wsh/vllm 的 patched vLLM）。
VLLM_ENV_BIN="${VLLM_ENV_BIN:-/home/wsh/miniconda3/envs/opencode/bin}"
# 判官客户端（纯 HTTP + trace 读取）用有全部依赖的 env。
CLIENT_PY="${CLIENT_PY:-/home/wsh/miniconda3/envs/langgraph_vllm/bin/python}"

export VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B}"
export VLLM_SERVED_NAME="${VLLM_SERVED_NAME:-Qwen3}"
export VLLM_PORT="${VLLM_PORT:-8000}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"

RESULTS_DIR="${RESULTS_DIR:-$ROOT/results/problem_exploration/cross_task_contextual_occ1_reuse}"
ARMS=(${ARMS:-recompute cross_contextual_rope})
OUT_DIR="${OUT_DIR:-$RESULTS_DIR/judge}"
STOP_VLLM_AFTER="${STOP_VLLM_AFTER:-1}"

MODULE_DIR="$ROOT/scripts/06_context_free_segment_cache/module"
JUDGE_DIR="$MODULE_DIR/critic_reference"

cleanup() {
  if [[ "$STOP_VLLM_AFTER" == "1" ]]; then
    bash "$ROOT/scripts/vllm_stop.sh" || true
  fi
}
trap cleanup EXIT

# 判官只需普通 Qwen3，不注入 skill KV。
unset VLLM_CONTEXT_SEGMENT_KV_DIR || true
unset VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR || true

echo "[vLLM] starting plain Qwen3 judge server (no KV injection)"
bash "$ROOT/scripts/vllm_stop.sh" || true
PATH="$VLLM_ENV_BIN:$PATH" CONDA_PREFIX="/home/wsh/miniconda3/envs/opencode" \
  bash "$ROOT/scripts/vllm_start.sh"

ready=0
for _ in $(seq 1 600); do
  code="$(curl -sS -o /dev/null -w '%{http_code}' --connect-timeout 3 --max-time 10 \
    -H "Authorization: Bearer ${VLLM_API_KEY}" "http://127.0.0.1:${VLLM_PORT}/v1/models" 2>/dev/null || true)"
  if [[ "$code" == "200" ]]; then ready=1; echo "[vLLM] ready"; break; fi
  sleep 2
done
[[ "$ready" == "1" ]] || { echo "[error] vLLM start timeout" >&2; exit 1; }

echo "[judge] scoring arms: ${ARMS[*]}"
JUDGE_BASE_URL="http://127.0.0.1:${VLLM_PORT}" JUDGE_MODEL="$VLLM_SERVED_NAME" JUDGE_API_KEY="$VLLM_API_KEY" \
  "$CLIENT_PY" "$JUDGE_DIR/run_judge.py" \
    --results-dir "$RESULTS_DIR" \
    --arms "${ARMS[@]}" \
    --out "$OUT_DIR"

echo "[done] judge outputs in $OUT_DIR"
