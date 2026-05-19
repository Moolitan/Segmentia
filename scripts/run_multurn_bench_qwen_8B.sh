# 跑全部：对每个 CONTEXT_LENGTH 导出 VLLM_MAX_MODEL_LEN 启动 vLLM，再跑
#   (noskill + skill，若 BENCH_MODE=all) × SEQ_ARR，共 |SEQ_ARR|×|CONTEXT_LENGTH|×组数 次 run_multurn。
# bash scripts/run_multurn_bench.sh
#
# 仅跑对照组
# BENCH_MODE=noskill bash scripts/run_multurn_bench.sh
# BENCH_MODE=skill bash scripts/run_multurn_bench.sh
#
# 仅从已有数据生成对比图（默认读 results/.../ctx_<FIGURES_CONTEXT_LENGTH>/）
# FIGURES_CONTEXT_LENGTH=8192 BENCH_MODE=figures bash scripts/run_multurn_bench.sh


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

WS_PARENT="${MULTRUN_BENCH_WORKSPACE:-$ROOT/workspace/03-8B/multurn_bench}"
SK_PARENT="${SK_PARENT:-$ROOT/skills}"
DOC_PARENT="${DOC_PARENT:-$ROOT/doc}"
RES_PARENT="${MULTRUN_BENCH_RESULTS:-$ROOT/results/03-8B/multurn_bench}"
LOG_PARENT="${MULTRUN_BENCH_LOG:-$ROOT/log/03-8B/multurn_bench}"

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
  # 200 正常；401/403 表示 HTTP 已通、仅鉴权与预期不一致,仍视为「服务已起来」
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
      echo "[vLLM] 仍在等待... ${attempt}/${VLLM_READY_MAX_ATTEMPTS}(约 $(( attempt * VLLM_READY_INTERVAL ))s),大模型未监听端口前会一直重试"
    fi
    sleep "$VLLM_READY_INTERVAL"
  done
  echo "[vLLM] 超时仍未就绪,请检查 ${ROOT}/log/vllm.log" >&2
  return 1
}


RUN_MULTURN="$SCRIPT_DIR/03_8B/run_multurn.py"
VISUALIZE_MULTURN="$SCRIPT_DIR/03_8B/visualize_multiturn_per_request.py"
SEGKV_CHAR="$SCRIPT_DIR/03_8B/segkv_characterization.py"
SKILL_COMPARE="$SCRIPT_DIR/03_8B/skill_comparison.py"

# SEQ_ARR=(T4_B T6_F T8_A T10_A)
SEQ_ARR=(T8_A_SKILL)
# 与 vLLM --max-model-len 一致；实验规模为 |SEQ_ARR| × |CONTEXT_LENGTH| × {skill, noskill(若 BENCH_MODE=all)}
CONTEXT_LENGTH=(32768)
# CONTEXT_LENGTH=(16384 32768)
SKILL_SYSTEM_PROMPT="system_prompt_skill.j2"
NO_SKILL_SYSTEM_PROMPT="system_prompt.j2"
# SEQ_ARR=(T2_A)
# SEQ_ARR=(T6_F)
# SEQ_ARR=(T8_A T10_A)

# 生成对比图时使用的 results 子目录（与某次 CONTEXT_LENGTH 对齐），默认取 CONTEXT_LENGTH 最后一个
_ctx_last_idx=$(( ${#CONTEXT_LENGTH[@]} - 1 ))
FIGURES_CONTEXT_LENGTH="${FIGURES_CONTEXT_LENGTH:-${CONTEXT_LENGTH[_ctx_last_idx]}}"
COMPARE_RESULTS_DIR="$RES_PARENT/ctx_${FIGURES_CONTEXT_LENGTH}"

# ── 运行模式 ────────────────────────────────────────────────────────
# BENCH_MODE 控制运行哪些实验组:
#   "all"       — 先跑 no-skills 对照组,再跑 with-skills 实验组（默认）
#   "skill"     — 仅跑 with-skills 实验组（原有行为）
#   "noskill"   — 仅跑 no-skills 对照组
#   "figures"   — 不跑实验,仅从已有数据生成对比图
BENCH_MODE="${BENCH_MODE:-all}"

# ── 单组运行函数 ─────────────────────────────────────────────────────
# 参数: group_label extra_flag suffix ctx_len system_prompt_j2 [python 额外参数...]
run_group() {
  local group_label="$1"   # "skill" or "noskill"
  local extra_flag="$2"    # "" or "--no-skills"
  local suffix="$3"        # "" or "_noskill"
  local ctx_len="$4"
  local system_prompt="$5"
  shift 5                  # 剩余参数透传给 python

  local n_seqs=${#SEQ_ARR[@]}
  local seq_idx=0
  local ctx_tag="ctx_${ctx_len}"

  for seq in "${SEQ_ARR[@]}"; do
    seq_idx=$((seq_idx + 1))
    local seq_ws="$WS_PARENT/${ctx_tag}/${seq}${suffix}"
    local seq_res="$RES_PARENT/${ctx_tag}/${seq}${suffix}"
    local vis_res="$seq_res/figures"
    local seq_log="$LOG_PARENT/${ctx_tag}/${seq}${suffix}"
    mkdir -p "$seq_ws" "$seq_res" "$vis_res" "$seq_log"

    if [[ "$group_label" == "skill" ]]; then
      local seq_skills="$seq_ws/.agents/skills"
      mkdir -p "$seq_ws/.agents"
      rm -rf "$seq_skills"
      mkdir -p "$seq_skills"
      echo "[skills] 同步 $SK_PARENT → $seq_skills"
      cp -a "$SK_PARENT"/. "$seq_skills"/
      echo "[docs] 同步 $DOC_PARENT → $seq_ws"
      cp -a "$DOC_PARENT"/. "$seq_ws"/
    fi

    echo "========== run_multurn [${group_label}] ctx=${ctx_len}: ${seq} (${seq_idx}/${n_seqs}) =========="
    echo "  workspace 根: $seq_ws"
    echo "  结果目录:     $seq_res"
    echo "  日志目录:     $seq_log"
    echo "  system_prompt: $system_prompt"
    [[ -n "$extra_flag" ]] && echo "  额外参数:     $extra_flag"

    if python "$RUN_MULTURN" \
      --sequence "$seq" \
      --vllm-port "$VLLM_PORT" \
      --workspace "$seq_ws" \
      --output "$seq_res/multiturn_sequence_traces.json" \
      --log-dir "$seq_log" \
      --system-prompt-filename "$system_prompt" \
      --context-length "$ctx_len" \
      $extra_flag \
      "$@"; then
      echo "========== 序列 ${seq} [${group_label}] ctx=${ctx_len} 完成 =========="
      echo "========== 序列 ${seq} [${group_label}] ctx=${ctx_len} 画图 =========="
      python "$VISUALIZE_MULTURN" \
        --input "$seq_res/multiturn_sequence_traces.json" \
        --out-dir "$seq_res/figures"
    else
      exit_code=$?
      echo "========== [ERROR] 序列 ${seq} [${group_label}] ctx=${ctx_len} 失败(exit=${exit_code}),跳过继续下一个 =========="
      failed_seqs+=("${ctx_tag}/${seq}${suffix}")
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

if [[ "$BENCH_MODE" != "figures" ]]; then
  echo "[vLLM] 运行前 stop vLLM"
  bash "$SCRIPT_DIR/vllm_stop.sh"
fi

# ── 按 CONTEXT_LENGTH × (noskill/skill) × SEQ_ARR 跑实验 ──
if [[ "$BENCH_MODE" != "figures" ]]; then
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

    # ── 对照组: no-skills ──
    if [[ "$BENCH_MODE" == "all" || "$BENCH_MODE" == "noskill" ]]; then
      echo ""
      echo "################################################################"
      echo "#  对照组: NO-SKILLS ctx=${ctx} (${SEQ_ARR[*]})"
      echo "################################################################"
      run_group "noskill" "--no-skills" "_noskill" "$ctx" "$NO_SKILL_SYSTEM_PROMPT" "$@"

      # 组间重启 vLLM 清空 KV Cache（仍为同一 CONTEXT_LENGTH）
      if [[ "$BENCH_MODE" == "all" ]]; then
        echo "[vLLM] 对照组完成(ctx=${ctx}), 重启 vLLM 准备实验组..."
        bash "$SCRIPT_DIR/vllm_stop.sh"
        bash "$SCRIPT_DIR/vllm_start.sh"
        sleep 2
        wait_vllm_ready
      fi
    fi

    # ── 实验组: with-skills ──
    if [[ "$BENCH_MODE" == "all" || "$BENCH_MODE" == "skill" ]]; then
      echo ""
      echo "################################################################"
      echo "#  实验组: WITH-SKILLS ctx=${ctx} (${SEQ_ARR[*]})"
      echo "################################################################"
      run_group "skill" "" "" "$ctx" "$SKILL_SYSTEM_PROMPT" "$@"
    fi

    # 下一档上下文长度前停掉 vLLM，下一轮以新的 max-model-len 启动
    if (( ctx_run < n_ctx )); then
      echo "[vLLM] 本档 ctx=${ctx} 全部序列完成, stop vLLM 准备下一 CONTEXT_LENGTH..."
      bash "$SCRIPT_DIR/vllm_stop.sh"
    fi
  done
fi

# ── 生成对比图 ──
if [[ "$BENCH_MODE" == "all" || "$BENCH_MODE" == "figures" ]]; then
  echo ""
  echo "========== 生成 skill vs no-skill 对比图 (results: $COMPARE_RESULTS_DIR, FIGURES_CONTEXT_LENGTH=$FIGURES_CONTEXT_LENGTH) =========="
  COMPARE_OUT="$COMPARE_RESULTS_DIR/segkv_figures"
  mkdir -p "$COMPARE_OUT"
  python "$SKILL_COMPARE" \
    --results-dir "$COMPARE_RESULTS_DIR" \
    --out-dir "$COMPARE_OUT" || echo "[WARN] 对比图生成失败,请检查数据"

  # 同时重新生成原有的 segkv characterization 图
  python "$SEGKV_CHAR" \
    --results-dir "$COMPARE_RESULTS_DIR" \
    --out-dir "$COMPARE_OUT" || echo "[WARN] characterization 图生成失败"
fi

echo ""
echo "========== bench 全部完成 =========="
total_seqs=0
n_ctx=${#CONTEXT_LENGTH[@]}
n_seq=${#SEQ_ARR[@]}
if [[ "$BENCH_MODE" == "all" ]]; then
  total_seqs=$(( n_ctx * n_seq * 2 ))
elif [[ "$BENCH_MODE" == "noskill" || "$BENCH_MODE" == "skill" ]]; then
  total_seqs=$(( n_ctx * n_seq ))
fi
if (( ${#failed_seqs[@]} > 0 )); then
  echo "  失败序列 (${#failed_seqs[@]}/${total_seqs}): ${failed_seqs[*]}"
  echo "  请检查对应日志: $LOG_PARENT/<seq>/"
else
  echo "  全部 ${total_seqs} 个序列成功"
fi
