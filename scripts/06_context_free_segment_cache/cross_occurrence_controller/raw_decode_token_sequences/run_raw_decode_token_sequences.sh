#!/usr/bin/env bash
# 导出 occurrence 3 的原始 decode token 序列（temp=0.6 采样档）。
# 一次跑完三条 run（mode / seed / run_name 一一对应）：
#   recompute_run1  seed=1111  （与 rope 同 seed，测 KV 纯效应的基准 A）
#   recompute_run2  seed=2222  （与 A 不同 seed，测模型固有方差基线）
#   rope            seed=1111  （复用臂）
# 每个 (run, task) 都重启 vLLM 并清空 prefix cache；换 mode 时自动切换是否加载 skill KV。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$ROOT"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy || true

export VLLM_MODEL_PATH="${VLLM_MODEL_PATH:-/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B}"
export VLLM_SERVED_NAME="${VLLM_SERVED_NAME:-Qwen3}"
export VLLM_PORT="${VLLM_PORT:-8000}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"

# 采样参数：Qwen3 thinking 默认（不再贪心）。可用环境变量覆盖。
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
MIN_P="${MIN_P:-0}"

# 输出路径自动按 温度档 / 复用条件 / run 分层，避免写死标签、避免覆盖别的温度档。
#   TEMP_TAG   温度档目录名，如 temp0.6（默认取 TEMPERATURE）
#   RUN_LABEL  复用条件标签：full_reuse / without_occ1 / without_occ12（occ=(3,) 即 without_occ12）
TEMP_TAG="${TEMP_TAG:-temp${TEMPERATURE}}"
RUN_LABEL="${RUN_LABEL:-without_occ12}"
RESULT_DIR="${RAW_SEQUENCE_RESULT_DIR:-$ROOT/results/problem_exploration/raw_decode_token_sequences}"

SEGMENTIA_OUTPUT_DIR="${SEGMENTIA_OUTPUT_DIR:-/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/06_context_free_segment_cache}"
KV_DIR="${KV_DIR:-$SEGMENTIA_OUTPUT_DIR/offline_skill_kv}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-1}"
RESUME="${RESUME:-0}"
TASKS="${TASKS:-internal_comms_incident_update,doc_coauthoring_design_doc,mcp_server_and_spec,web_artifact_with_theme,launch_poster_page_pack,slack_launch_pack}"

# 三条 run：下标一一对应。
RUN_MODES=(recompute recompute rope)
RUN_SEEDS=(1111 2222 1111)
RUN_NAMES=(recompute_run1 recompute_run2 rope)

start_vllm() {
  local mode="$1"
  local task="$2"

  echo ""
  echo "[vLLM] restart boundary=(mode=$mode, task=$task)"
  unset VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR || true
  if [[ "$mode" == "recompute" ]]; then
    unset VLLM_CONTEXT_SEGMENT_KV_DIR || true
  else
    export VLLM_CONTEXT_SEGMENT_KV_DIR="$KV_DIR"
  fi

  bash "$ROOT/scripts/vllm_stop.sh" || true
  bash "$ROOT/scripts/vllm_start.sh"

  local ready=0
  for _poll_i in $(seq 1 600); do
    code="$(curl -sS -o /dev/null -w '%{http_code}' \
      --connect-timeout 3 --max-time 10 \
      -H "Authorization: Bearer ${VLLM_API_KEY}" \
      "http://127.0.0.1:${VLLM_PORT}/v1/models" 2>/dev/null || true)"
    if [[ "$code" == "200" ]]; then
      ready=1
      echo "[vLLM] ready"
      break
    fi
    sleep 2
  done
  if [[ "$ready" != "1" ]]; then
    echo "[error] vLLM start timeout for mode=$mode task=$task" >&2
    exit 1
  fi
}

IFS=',' read -ra TASK_LIST <<< "$TASKS"

# run 是外层、task 是内层；每个 (run, task) 都重启服务清 prefix cache。
# occurrence 由 Python 端 OCCURRENCES=(3,) 决定，只保存 occ3。
for idx in "${!RUN_MODES[@]}"; do
  mode="${RUN_MODES[$idx]}"
  seed="${RUN_SEEDS[$idx]}"
  run_name="${RUN_NAMES[$idx]}"

  run_dir="$RESULT_DIR/$TEMP_TAG/$RUN_LABEL/$run_name"
  sequence_dir="$run_dir"
  manifest="$run_dir/sequence_manifest.jsonl"
  mkdir -p "$sequence_dir"
  if [[ "$CLEAN_OUTPUT" == "1" ]]; then
    find "$sequence_dir" -type f -name '*.txt' -delete
    rm -f "$manifest"
  fi

  echo ""
  echo "########## run=$run_name (mode=$mode, seed=$seed) -> $run_dir ##########"

  for task in "${TASK_LIST[@]}"; do
    task="${task// /}"
    [[ -z "$task" ]] && continue
    start_vllm "$mode" "$task"

    args=(
      --task "$task"
      --mode "$mode"
      --sequence-dir "$sequence_dir"
      --manifest "$manifest"
      --vllm-port "$VLLM_PORT"
      --model "$VLLM_SERVED_NAME"
      --api-key "$VLLM_API_KEY"
      --kv-dir "$KV_DIR"
      --max-tokens "$MAX_TOKENS"
      --temperature "$TEMPERATURE"
      --top-p "$TOP_P"
      --seed "$seed"
    )
    [[ -n "$TOP_K" ]] && args+=(--top-k "$TOP_K")
    [[ -n "$MIN_P" ]] && args+=(--min-p "$MIN_P")
    if [[ "$RESUME" == "1" ]]; then
      args+=(--resume)
    fi
    python "$SCRIPT_DIR/run_raw_decode_token_sequences.py" "${args[@]}"
  done
done

echo ""
echo "[done] runs: ${RUN_NAMES[*]}"
echo "[done] under: $RESULT_DIR/$TEMP_TAG/$RUN_LABEL"
