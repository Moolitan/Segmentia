# SUDO_PASSWORD=storage-b520 nohup bash scripts/run_multurn_bench_qwen_14B_anthropic_sglang.sh
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy ftp_proxy FTP_PROXY

# 让所有子进程中的 npx 自动跳过 "Ok to proceed?" 确认，无需 agent 显式加 --yes
export npm_config_yes=true
export CI=true
export LOG_JSON=false  # 防止 CI=true 触发 SDK JSON 日志格式

# sudo 密码来源：仅从环境变量 SUDO_PASSWORD 读取
# 例如：SUDO_PASSWORD='your_password' bash scripts/run_multurn_bench_qwen_14B_anthropic.sh
SUDO_PASSWORD="${SUDO_PASSWORD:-}"
if [[ -z "$SUDO_PASSWORD" ]]; then
  echo "[sudo] 未检测到环境变量 SUDO_PASSWORD，无法在非交互场景自动输入 sudo 密码" >&2
  echo "[sudo] 请先设置：SUDO_PASSWORD='<your_sudo_password>' 后再执行脚本" >&2
  exit 1
fi

sudo_with_password() {
  sudo -S -p '' "$@" <<<"$SUDO_PASSWORD"
}

# ── 临时免密 sudo（仅 bench 运行期间生效）──────────────────────────────
_SUDOERS_TMP="/etc/sudoers.d/bench-temp-nopasswd"
echo "[sudo] 临时开启免密 sudo（脚本结束自动清理）"
_SUDOERS_LINE="$(whoami) ALL=(ALL) NOPASSWD: ALL"
_SUDOERS_LOCAL_TMP="$(mktemp)"
printf '%s\n' "$_SUDOERS_LINE" > "$_SUDOERS_LOCAL_TMP"
sudo_with_password install -m 440 "$_SUDOERS_LOCAL_TMP" "$_SUDOERS_TMP"
rm -f "$_SUDOERS_LOCAL_TMP"
trap 'sudo_with_password rm -f "$_SUDOERS_TMP"; echo "[sudo] 已清理临时免密规则"' EXIT

SGLANG_PORT="${SGLANG_PORT:-8000}"
export SGLANG_PORT
export SGLANG_API_KEY="${SGLANG_API_KEY:-EMPTY}"

WS_PARENT="${MULTRUN_BENCH_WORKSPACE:-$ROOT/workspace/03_14B_anthropic_sglang}"
SK_PARENT="${SK_PARENT:-$ROOT/skills}"
RES_PARENT="${MULTRUN_BENCH_RESULTS:-$ROOT/results/03_14B_anthropic_sglang}"
LOG_PARENT="${MULTRUN_BENCH_LOG:-$ROOT/log/03_14B_anthropic_sglang}"
  
SGLANG_READY_MAX_ATTEMPTS="${SGLANG_READY_MAX_ATTEMPTS:-180}"
SGLANG_READY_INTERVAL="${SGLANG_READY_INTERVAL:-2}"

# 用 curl 轮询就绪(与终端手测一致)
sglang_probe_ok() {
  local code_m code_h
  code_m="$(
    curl -sS -o /dev/null -w '%{http_code}' \
      --connect-timeout 5 --max-time 15 \
      -H "Authorization: Bearer ${SGLANG_API_KEY}" \
      "http://127.0.0.1:${SGLANG_PORT}/v1/models" 2>/dev/null || true
  )"
  [[ -z "$code_m" ]] && code_m="000"
  if [[ "$code_m" == "200" || "$code_m" == "401" || "$code_m" == "403" ]]; then
    return 0
  fi
  code_h="$(
    curl -sS -o /dev/null -w '%{http_code}' \
      --connect-timeout 5 --max-time 15 \
      "http://127.0.0.1:${SGLANG_PORT}/health" 2>/dev/null || true
  )"
  [[ -z "$code_h" ]] && code_h="000"
  [[ "$code_h" == "200" ]]
}

wait_sglang_ready() {
  local attempt=0
  if ! command -v curl &>/dev/null; then
    echo "[SGLang] 需要 curl(用于就绪探测),当前 PATH 中未找到" >&2
    return 1
  fi
  echo "[SGLang] 等待就绪(curl): /v1/models 与 /health,端口 ${SGLANG_PORT}"
  echo "[SGLang] 提示: 首次加载权重常需数分钟,日志见 ${ROOT}/log/sglang.log"
  while (( attempt < SGLANG_READY_MAX_ATTEMPTS )); do
    if sglang_probe_ok; then
      echo "[SGLang] 已就绪 (port $SGLANG_PORT)"
      return 0
    fi
    attempt=$((attempt + 1))
    if (( attempt % 15 == 0 )); then
      echo "[SGLang] 仍在等待... ${attempt}/${SGLANG_READY_MAX_ATTEMPTS}(约 $(( attempt * SGLANG_READY_INTERVAL ))s)"
    fi
    sleep "$SGLANG_READY_INTERVAL"
  done
  echo "[SGLang] 超时仍未就绪,请检查 ${ROOT}/log/sglang.log" >&2
  return 1
}


RUN_MULTURN="$SCRIPT_DIR/03_14B_anthropic/run_multurn.py"
VISUALIZE_MULTURN="$SCRIPT_DIR/03_14B_anthropic/visualize_multiturn_per_request.py"

SEQ_ARR=(
  doc_coauthoring_design_doc
  internal_comms_incident_update
  web_artifact_with_theme
  mcp_server_and_spec
  launch_poster_page_pack
  slack_launch_pack
)

CONTEXT_LENGTH=(32768)

SYSTEM_PROMPT="${SYSTEM_PROMPT:-system_prompt.j2}"

# ── benchmark 组定义 ──────────────────────────────────────────────────
# 每组: "目录名:标签"
# - anthropic_skill_benchmark_8_repos       → natural（自然提示词）
# - anthropic_skill_benchmark_8_repos_explicit_skills → explicit_skills（明确告知使用 skills）
BENCH_GROUPS=(
  "anthropic_skill_benchmark_8_repos:natural"
  "anthropic_skill_benchmark_8_repos_explicit_skills:explicit_skills"
)

# ── 运行函数 ─────────────────────────────────────────────────────────
# run_all <ctx_len> <bench_dir> <group_tag> [extra args...]
run_all() {
  local ctx_len="$1"
  local bench_dir="$2"
  local group_tag="$3"
  shift 3

  local n_seqs=${#SEQ_ARR[@]}
  local seq_idx=0
  local ctx_tag="ctx_${ctx_len}"

  for repo in "${SEQ_ARR[@]}"; do
    seq_idx=$((seq_idx + 1))
    local seq_ws="$WS_PARENT/${group_tag}/${ctx_tag}/${repo}"
    local seq_res="$RES_PARENT/${group_tag}/${ctx_tag}/${repo}"
    local seq_log="$LOG_PARENT/${group_tag}/${ctx_tag}/${repo}"
    mkdir -p "$seq_ws" "$seq_res" "$seq_res/figures" "$seq_log"

    # 只将 seed_files/ 目录同步到工作区间（不复制 README.md / expected_skills.json / task.json）
    local repo_bench_dir="$ROOT/$bench_dir/$repo"
    if [[ -d "$repo_bench_dir/seed_files" ]]; then
      cp -a "$repo_bench_dir/seed_files" "$seq_ws/"
      echo "[bench] 已同步 $repo_bench_dir/seed_files → $seq_ws/seed_files"
    fi

    # 同步 skills
    local seq_skills="$seq_ws/.agents/skills"
    mkdir -p "$seq_ws/.agents"
    rm -rf "$seq_skills"
    cp -a "$SK_PARENT"/. "$seq_skills"/
    echo "[skills] 已同步 $SK_PARENT → $seq_skills"

    echo "========== run_multurn group=${group_tag} ctx=${ctx_len}: ${repo} (${seq_idx}/${n_seqs}) =========="
    echo "  bench-root: $ROOT/$bench_dir"
    echo "  workspace:  $seq_ws"
    echo "  结果目录:   $seq_res"
    echo "  日志目录:   $seq_log"

    if python "$RUN_MULTURN" \
      --benchmark-repo "$repo" \
      --bench-root "$ROOT/$bench_dir" \
      --vllm-port "$SGLANG_PORT" \
      --workspace "$seq_ws" \
      --output "$seq_res/multiturn_sequence_traces.json" \
      --log-dir "$seq_log" \
      --system-prompt-filename "$SYSTEM_PROMPT" \
      --context-length "$ctx_len" \
      "$@"; then
      echo "========== ${group_tag}/${repo} ctx=${ctx_len} 完成 =========="
      python "$VISUALIZE_MULTURN" \
        --input "$seq_res/multiturn_sequence_traces.json" \
        --out-dir "$seq_res/figures"
    else
      exit_code=$?
      echo "========== [ERROR] ${group_tag}/${repo} ctx=${ctx_len} 失败(exit=${exit_code}),跳过继续下一个 =========="
      failed_seqs+=("${group_tag}/${ctx_tag}/${repo}")
    fi

    if (( seq_idx < n_seqs )); then
      echo "[SGLang] stop / start SGLang(保持 SGLANG_MAX_MODEL_LEN=${ctx_len}),就绪后继续下一序列..."
      bash "$SCRIPT_DIR/sglang_stop.sh"
      bash "$SCRIPT_DIR/sglang_start.sh"
      sleep 2
      wait_sglang_ready
    fi
  done
}

# ── 主流程 ────────────────────────────────────────────────────────────

failed_seqs=()

echo "[SGLang] 运行前 stop SGLang"
bash "$SCRIPT_DIR/sglang_stop.sh"

ctx_run=0
n_ctx=${#CONTEXT_LENGTH[@]}
for ctx in "${CONTEXT_LENGTH[@]}"; do
  ctx_run=$((ctx_run + 1))
  export SGLANG_MAX_MODEL_LEN="$ctx"
  echo ""
  echo "################################################################"
  echo "#  CONTEXT_LENGTH=${ctx} → SGLANG_MAX_MODEL_LEN=${ctx} (${ctx_run}/${n_ctx})"
  echo "################################################################"
  echo "[SGLang] start (max-model-len=$ctx)"
  bash "$SCRIPT_DIR/sglang_start.sh"
  sleep 2
  wait_sglang_ready

  for bench_entry in "${BENCH_GROUPS[@]}"; do
    local_bench_dir="${bench_entry%%:*}"
    local_group_tag="${bench_entry##*:}"
    echo ""
    echo "================================================================"
    echo "#  benchmark 组: ${local_group_tag}  (${local_bench_dir})"
    echo "================================================================"
    run_all "$ctx" "$local_bench_dir" "$local_group_tag" "$@"
  done

  if (( ctx_run < n_ctx )); then
    echo "[SGLang] 本档 ctx=${ctx} 全部序列完成, stop SGLang 准备下一 CONTEXT_LENGTH..."
    bash "$SCRIPT_DIR/sglang_stop.sh"
  fi
done

echo ""
echo "========== bench 全部完成 =========="
total_seqs=$(( ${#CONTEXT_LENGTH[@]} * ${#BENCH_GROUPS[@]} * ${#SEQ_ARR[@]} ))
if (( ${#failed_seqs[@]} > 0 )); then
  echo "  失败序列 (${#failed_seqs[@]}/${total_seqs}): ${failed_seqs[*]}"
  echo "  请检查对应日志: $LOG_PARENT/<group>/<ctx>/<repo>/"
else
  echo "  全部 ${total_seqs} 个序列成功"
fi
