#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy ftp_proxy FTP_PROXY

# ── 临时免密 sudo（仅 bench 运行期间生效）──────────────────────────────
_SUDOERS_TMP="/etc/sudoers.d/bench-temp-nopasswd"
echo "[sudo] 临时开启免密 sudo（脚本结束自动清理）"
echo "$(whoami) ALL=(ALL) NOPASSWD: ALL" | sudo tee "$_SUDOERS_TMP" > /dev/null
sudo chmod 440 "$_SUDOERS_TMP"
trap 'sudo rm -f "$_SUDOERS_TMP"; echo "[sudo] 已清理临时免密规则"' EXIT

VLLM_PORT="${VLLM_PORT:-8000}"
export VLLM_PORT
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"

WS_PARENT="${MULTRUN_BENCH_WORKSPACE:-$ROOT/workspace/03_8B_anthropic/multurn_bench}"
SK_PARENT="${SK_PARENT:-$ROOT/skills}"
RES_PARENT="${MULTRUN_BENCH_RESULTS:-$ROOT/results/03_8B_anthropic/multurn_bench}"
LOG_PARENT="${MULTRUN_BENCH_LOG:-$ROOT/log/03_8B_anthropic/multurn_bench}"

VLLM_READY_MAX_ATTEMPTS="${VLLM_READY_MAX_ATTEMPTS:-180}"
VLLM_READY_INTERVAL="${VLLM_READY_INTERVAL:-2}"

# 用 curl 轮询就绪(与终端手测一致)
vllm_probe_ok() {
  local code_m code_h
  code_m="$(
    curl -sS -o /dev/null -w '%{http_code}' \
      --connect-timeout 5 --max-time 15 \
      -H "Authorization: Bearer ${VLLM_API_KEY}" \
      "http://127.0.0.1:${VLLM_PORT}/v1/models" 2>/dev/null || true
  )"
  [[ -z "$code_m" ]] && code_m="000"
  if [[ "$code_m" == "200" || "$code_m" == "401" || "$code_m" == "403" ]]; then
    return 0
  fi
  code_h="$(
    curl -sS -o /dev/null -w '%{http_code}' \
      --connect-timeout 5 --max-time 15 \
      "http://127.0.0.1:${VLLM_PORT}/health" 2>/dev/null || true
  )"
  [[ -z "$code_h" ]] && code_h="000"
  [[ "$code_h" == "200" ]]
}

wait_vllm_ready() {
  local attempt=0
  if ! command -v curl &>/dev/null; then
    echo "[vLLM] 需要 curl(用于就绪探测),当前 PATH 中未找到" >&2
    return 1
  fi
  echo "[vLLM] 等待就绪(curl): /v1/models 与 /health,端口 ${VLLM_PORT}"
  echo "[vLLM] 提示: 首次加载权重与 CUDA graph 常需数分钟,日志见 ${ROOT}/log/vllm.log"
  while (( attempt < VLLM_READY_MAX_ATTEMPTS )); do
    if vllm_probe_ok; then
      echo "[vLLM] 已就绪 (port $VLLM_PORT)"
      return 0
    fi
    attempt=$((attempt + 1))
    if (( attempt % 15 == 0 )); then
      echo "[vLLM] 仍在等待... ${attempt}/${VLLM_READY_MAX_ATTEMPTS}(约 $(( attempt * VLLM_READY_INTERVAL ))s)"
    fi
    sleep "$VLLM_READY_INTERVAL"
  done
  echo "[vLLM] 超时仍未就绪,请检查 ${ROOT}/log/vllm.log" >&2
  return 1
}


RUN_MULTURN="$SCRIPT_DIR/03_8B_anthropic/run_multurn.py"
VISUALIZE_MULTURN="$SCRIPT_DIR/03_8B_anthropic/visualize_multiturn_per_request.py"

# SEQ_ARR=(
#   baseline_feature_brainstorm
#   baseline_update_polish
#   doc_coauthoring_design_doc
#   internal_comms_incident_update
#   web_artifact_with_theme
#   mcp_server_and_spec
#   launch_poster_page_pack
#   slack_launch_pack
# )

SEQ_ARR=(
  slack_launch_pack
)

CONTEXT_LENGTH=(32768)
# CONTEXT_LENGTH=(16384 32768)
SYSTEM_PROMPT="${SYSTEM_PROMPT:-system_prompt.j2}"

# ── 运行函数 ─────────────────────────────────────────────────────────
run_all() {
  local ctx_len="$1"
  shift

  local n_seqs=${#SEQ_ARR[@]}
  local seq_idx=0
  local ctx_tag="ctx_${ctx_len}"

  for repo in "${SEQ_ARR[@]}"; do
    seq_idx=$((seq_idx + 1))
    local seq_ws="$WS_PARENT/${ctx_tag}/${repo}"
    local seq_res="$RES_PARENT/${ctx_tag}/${repo}"
    local seq_log="$LOG_PARENT/${ctx_tag}/${repo}"
    mkdir -p "$seq_ws" "$seq_res" "$seq_res/figures" "$seq_log"

    # 同步 benchmark repo 的 seed_files/ 和 README.md 到工作区
    local bench_dir="$ROOT/anthropic_skill_benchmark/${repo}"
    if [[ -d "$bench_dir/seed_files" ]]; then
      cp -a "$bench_dir/seed_files" "$seq_ws/"
      echo "[bench] 已同步 $bench_dir/seed_files → $seq_ws/seed_files"
    fi
    if [[ -f "$bench_dir/README.md" ]]; then
      cp "$bench_dir/README.md" "$seq_ws/"
      echo "[bench] 已同步 $bench_dir/README.md → $seq_ws/README.md"
    fi

    # 同步 skills
    local seq_skills="$seq_ws/.agents/skills"
    mkdir -p "$seq_ws/.agents"
    rm -rf "$seq_skills"
    cp -a "$SK_PARENT"/. "$seq_skills"/
    echo "[skills] 已同步 $SK_PARENT → $seq_skills"

    echo "========== run_multurn ctx=${ctx_len}: ${repo} (${seq_idx}/${n_seqs}) =========="
    echo "  workspace: $seq_ws"
    echo "  结果目录:  $seq_res"
    echo "  日志目录:  $seq_log"

    if python "$RUN_MULTURN" \
      --benchmark-repo "$repo" \
      --vllm-port "$VLLM_PORT" \
      --workspace "$seq_ws" \
      --output "$seq_res/multiturn_sequence_traces.json" \
      --log-dir "$seq_log" \
      --system-prompt-filename "$SYSTEM_PROMPT" \
      --context-length "$ctx_len" \
      "$@"; then
      echo "========== ${repo} ctx=${ctx_len} 完成 =========="
      python "$VISUALIZE_MULTURN" \
        --input "$seq_res/multiturn_sequence_traces.json" \
        --out-dir "$seq_res/figures"
    else
      exit_code=$?
      echo "========== [ERROR] ${repo} ctx=${ctx_len} 失败(exit=${exit_code}),跳过继续下一个 =========="
      failed_seqs+=("${ctx_tag}/${repo}")
    fi

    if (( seq_idx < n_seqs )); then
      echo "[vLLM] stop / start vLLM(保持 VLLM_MAX_MODEL_LEN=${ctx_len}),就绪后继续下一序列..."
      bash "$SCRIPT_DIR/vllm_stop.sh"
      bash "$SCRIPT_DIR/vllm_start.sh"
      sleep 2
      wait_vllm_ready
    fi
  done
}

# ── 主流程 ────────────────────────────────────────────────────────────

failed_seqs=()

echo "[vLLM] 运行前 stop vLLM"
bash "$SCRIPT_DIR/vllm_stop.sh"

ctx_run=0
n_ctx=${#CONTEXT_LENGTH[@]}
for ctx in "${CONTEXT_LENGTH[@]}"; do
  ctx_run=$((ctx_run + 1))
  export VLLM_MAX_MODEL_LEN="$ctx"
  echo ""
  echo "################################################################"
  echo "#  CONTEXT_LENGTH=${ctx} → VLLM_MAX_MODEL_LEN=${ctx} (${ctx_run}/${n_ctx})"
  echo "################################################################"
  echo "[vLLM] start (max-model-len=$ctx)"
  bash "$SCRIPT_DIR/vllm_start.sh"
  sleep 2
  wait_vllm_ready

  run_all "$ctx" "$@"

  if (( ctx_run < n_ctx )); then
    echo "[vLLM] 本档 ctx=${ctx} 全部序列完成, stop vLLM 准备下一 CONTEXT_LENGTH..."
    bash "$SCRIPT_DIR/vllm_stop.sh"
  fi
done

echo ""
echo "========== bench 全部完成 =========="
total_seqs=$(( ${#CONTEXT_LENGTH[@]} * ${#SEQ_ARR[@]} ))
if (( ${#failed_seqs[@]} > 0 )); then
  echo "  失败序列 (${#failed_seqs[@]}/${total_seqs}): ${failed_seqs[*]}"
  echo "  请检查对应日志: $LOG_PARENT/<ctx>/<repo>/"
else
  echo "  全部 ${total_seqs} 个序列成功"
fi
