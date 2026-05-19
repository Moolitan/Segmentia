#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

unset LD_LIBRARY_PATH
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy ftp_proxy FTP_PROXY


VLLM_PORT="${VLLM_PORT:-8000}"
export VLLM_PORT
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"



# BENCH_ROOT="$ROOT/anthropic_skill_benchmark_8_repos_explicit_skills"
BENCH_ROOT="$ROOT/anthropic_skill_benchmark"
RUN_MULTURN="$SCRIPT_DIR/03_14B_anthropic/run_multurn3.py"
wrl="03_14B_anthropic_3"


WS_PARENT="${MULTRUN_BENCH_WORKSPACE:-$ROOT/workspace/$wrl}"
SK_PARENT="${SK_PARENT:-$ROOT/skills}"
RES_PARENT="${MULTRUN_BENCH_RESULTS:-$ROOT/results/$wrl}"
LOG_PARENT="${MULTRUN_BENCH_LOG:-$ROOT/log/$wrl}"

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
  # doc_coauthoring_design_doc
  # launch_poster_page_pack
  slack_launch_pack
)



# ── 运行函数 ─────────────────────────────────────────────────────────
run_all() {
  local n_seqs=${#SEQ_ARR[@]}
  local seq_idx=0

  for repo in "${SEQ_ARR[@]}"; do
    seq_idx=$((seq_idx + 1))
    local seq_ws="$WS_PARENT/${repo}"
    local seq_res="$RES_PARENT/${repo}"
    local seq_log="$LOG_PARENT/${repo}"
    mkdir -p "$seq_ws" "$seq_res" "$seq_res/figures" "$seq_log"

    # 同步 benchmark repo 的 seed_files/ 和 README.md 到工作区
    local bench_dir="$BENCH_ROOT/${repo}"
    if [[ -d "$bench_dir/seed_files" ]]; then
      cp -a "$bench_dir/seed_files" "$seq_ws/"
      echo "[bench] 已同步 $bench_dir/seed_files → $seq_ws/seed_files"
    fi

    # 同步 skills
    local seq_skills="$seq_ws/.agents/skills"
    mkdir -p "$seq_ws/.agents"
    rm -rf "$seq_skills"
    cp -a "$SK_PARENT"/. "$seq_skills"/
    echo "[skills] 已同步 $SK_PARENT → $seq_skills"

    echo "========== run_multurn: ${repo} (${seq_idx}/${n_seqs}) =========="
    echo "  workspace: $seq_ws"
    echo "  结果目录:  $seq_res"
    echo "  日志目录:  $seq_log"

    if python "$RUN_MULTURN" \
      --benchmark-repo "$repo" \
      --bench-root "$BENCH_ROOT" \
      --vllm-port "$VLLM_PORT" \
      --workspace "$seq_ws" \
      --output "$seq_res/multiturn_sequence_traces.json" \
      --log-dir "$seq_log" \
      "$@"; then
      echo "========== ${repo} 完成 =========="
    else
      exit_code=$?
      echo "========== [ERROR] ${repo} 失败(exit=${exit_code}),跳过继续下一个 =========="
      failed_seqs+=("${repo}")
    fi

    if (( seq_idx < n_seqs )); then
      echo "[vLLM] stop / start vLLM(保持 VLLM_MAX_MODEL_LEN=32768),就绪后继续下一序列..."
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

export VLLM_MAX_MODEL_LEN="32768"
echo "[vLLM] start (max-model-len=32768)"
bash "$SCRIPT_DIR/vllm_start.sh"
sleep 2
wait_vllm_ready

run_all "$@"

echo ""
echo "========== bench 全部完成 =========="
total_seqs=${#SEQ_ARR[@]}
if (( ${#failed_seqs[@]} > 0 )); then
  echo "  失败序列 (${#failed_seqs[@]}/${total_seqs}): ${failed_seqs[*]}"
  echo "  请检查对应日志: $LOG_PARENT/<repo>/"
else
  echo "  全部 ${total_seqs} 个序列成功"
fi
