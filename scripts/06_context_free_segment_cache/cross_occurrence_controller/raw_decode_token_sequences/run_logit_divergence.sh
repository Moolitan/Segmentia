#!/usr/bin/env bash
# 方法 1+2: Token-level logit divergence + Branch point analysis（temp=0.6 采样档）。
# 一次跑完三条 run（mode / seed / run_name 一一对应）：
#   recompute_run1  seed=1111  （与 rope 同 seed，测 KV 纯效应的基准 A）
#   recompute_run2  seed=2222  （与 A 不同 seed，测模型固有方差基线）
#   rope            seed=1111  （复用臂）
# 三条 run 采完后自动做两对分析：
#   recompute_run1 vs rope           → KV 纯效应
#   recompute_run1 vs recompute_run2 → 模型固有方差基线
# 每个 (run, task) 都重启 vLLM 清 prefix cache；换 mode 自动切换是否加载 skill KV。
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

# 输出路径自动按 温度档 / 复用条件 分层；三条 run 落到 logprobs/<run_name>/ 子目录。
#   TEMP_TAG   温度档目录名，如 temp0.6（默认取 TEMPERATURE）
#   RUN_LABEL  复用条件标签：full_reuse / without_occ1 / without_occ12（occ=(3,) 即 without_occ12）
TEMP_TAG="${TEMP_TAG:-temp${TEMPERATURE}}"
RUN_LABEL="${RUN_LABEL:-without_occ12}"
export OUTPUT_DIR="${LOGIT_DIV_OUTPUT_DIR:-/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/06_context_free_segment_cache/cross_occurrence_function_vector/logit_divergence/$TEMP_TAG/$RUN_LABEL}"

SEGMENTIA_OUTPUT_DIR="${SEGMENTIA_OUTPUT_DIR:-/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/06_context_free_segment_cache}"
KV_DIR="${KV_DIR:-$SEGMENTIA_OUTPUT_DIR/offline_skill_kv}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
TASKS="${TASKS:-internal_comms_incident_update,doc_coauthoring_design_doc,mcp_server_and_spec,web_artifact_with_theme,launch_poster_page_pack,slack_launch_pack}"
# RUN_ANALYZE=0 时只采集不分析。
RUN_ANALYZE="${RUN_ANALYZE:-1}"

# 三条 run：下标一一对应。
RUN_MODES=(recompute recompute rope)
RUN_SEEDS=(1111 2222 1111)
RUN_NAMES=(recompute_run1 recompute_run2 rope)

mkdir -p "$OUTPUT_DIR"

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

# ── 阶段 1: Collect（run 外层、task 内层，每个 (run, task) 重启 vLLM）──
for idx in "${!RUN_MODES[@]}"; do
  mode="${RUN_MODES[$idx]}"
  seed="${RUN_SEEDS[$idx]}"
  run_name="${RUN_NAMES[$idx]}"

  echo ""
  echo "########## run=$run_name (mode=$mode, seed=$seed) -> $OUTPUT_DIR/logprobs/$run_name ##########"

  for task in "${TASK_LIST[@]}"; do
    task="${task// /}"
    [[ -z "$task" ]] && continue
    start_vllm "$mode" "$task"

    collect_args=(
      collect
      --task "$task"
      --mode "$mode"
      --output-dir "$OUTPUT_DIR"
      --run-name "$run_name"
      --vllm-port "$VLLM_PORT"
      --model "$VLLM_SERVED_NAME"
      --api-key "$VLLM_API_KEY"
      --max-tokens "$MAX_TOKENS"
      --temperature "$TEMPERATURE"
      --top-p "$TOP_P"
      --seed "$seed"
    )
    [[ -n "$TOP_K" ]] && collect_args+=(--top-k "$TOP_K")
    [[ -n "$MIN_P" ]] && collect_args+=(--min-p "$MIN_P")
    python "$SCRIPT_DIR/run_logit_divergence.py" "${collect_args[@]}"
  done
done

# ── 阶段 2: Analyze（两对；不需要 vLLM）──
if [[ "$RUN_ANALYZE" == "1" ]]; then
  echo ""
  echo "[analyze] recompute_run1 vs rope (KV 纯效应)"
  python "$SCRIPT_DIR/run_logit_divergence.py" analyze \
    --output-dir "$OUTPUT_DIR" --rc-name recompute_run1 --rp-name rope

  echo ""
  echo "[analyze] recompute_run1 vs recompute_run2 (固有方差基线)"
  python "$SCRIPT_DIR/run_logit_divergence.py" analyze \
    --output-dir "$OUTPUT_DIR" --rc-name recompute_run1 --rp-name recompute_run2
fi

echo ""
echo "[done] runs: ${RUN_NAMES[*]}"
echo "[done] under: $OUTPUT_DIR"
